#include "cuda/gemm.cuh"
#include "cuda/qkvg_fwd.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

constexpr usize SEQ_LEN = config::n_inputs;
constexpr usize D_MODEL = config::d_model;
constexpr usize Q_ROWS = config::n_heads * config::head_dim;
constexpr usize QKVG_ROWS = 4 * Q_ROWS;

namespace QkvgFwd {
	using WeightLoader =
		SparseMatrixLoader<
			config::d_model,
			config::qkv_fan_in,
			config::head_dim,
			64, 64
		>;

	using InputLoader =
		MatrixTransLoader<
			MatrixLoader<
				config::d_model,
				128, 64
			>
		>;

	using Writer =
		MatrixQKVGWriter<
			QKVG_ROWS,
			config::d_model,
			config::qkv_fan_in,
			config::n_heads,
			config::head_dim,
			config::l2_norm_eps,
			config::rope_dim,
			config::rope_base
		>;

	using Kernel = Gemm<WeightLoader, InputLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *w,
		bf16 *inp, usize n_inputs,
		bf16 *out,
		bf16 const *qk_norm_scales,
		bf16 const *sink_k,
		f32 *sink_scores
	) {
		auto a = WeightLoader(w, QKVG_ROWS);
		auto b = InputLoader(inp, n_inputs);
		auto o = Writer(out, n_inputs, qk_norm_scales, sink_k, sink_scores);
		Kernel().run(a, b, o);
	}
}

using namespace QkvgFwd;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	if (
		QKVG_ROWS % Kernel::M_PER_BLOCK != 0 ||
		SEQ_LEN % Kernel::N_PER_BLOCK != 0
	) {
		printf("Expected M %% %u == 0 and N %% %u == 0\n", Kernel::M_PER_BLOCK, Kernel::N_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_A = load_tensor(torch_tensor_path("qkvg_weights.bin"), QKVG_ROWS, config::qkv_fan_in);
	std::vector<bf16> h_B = load_tensor(torch_tensor_path("inputs_l2.bin"), SEQ_LEN, D_MODEL);
	std::vector<bf16> h_S = load_tensor(torch_tensor_path("qk_norm_scales.bin"), 1, Q_ROWS);
	std::vector<bf16> h_sink = load_tensor(torch_tensor_path("sinks_k.bin"), config::n_heads, config::head_dim);
	if (h_A.empty() || h_B.empty() || h_S.empty() || h_sink.empty()) {
		return 1;
	}
	std::vector<bf16> h_C(SEQ_LEN * QKVG_ROWS);
	std::vector<bf16> h_Q(SEQ_LEN * Q_ROWS);
	std::vector<bf16> h_K(SEQ_LEN * Q_ROWS);
	std::vector<bf16> h_V(SEQ_LEN * Q_ROWS);
	std::vector<bf16> h_G(SEQ_LEN * Q_ROWS);
	std::vector<f32> h_sink_scores(SEQ_LEN * config::n_heads);

	bf16 *d_A, *d_B, *d_S, *d_sink, *d_C;
	f32 *d_sink_scores;
	cudaMalloc(&d_A, h_A.size() * sizeof(bf16));
	cudaMalloc(&d_B, h_B.size() * sizeof(bf16));
	cudaMalloc(&d_S, h_S.size() * sizeof(bf16));
	cudaMalloc(&d_sink, h_sink.size() * sizeof(bf16));
	cudaMalloc(&d_C, h_C.size() * sizeof(bf16));
	cudaMalloc(&d_sink_scores, h_sink_scores.size() * sizeof(f32));
	cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_S, h_S.data(), h_S.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sink, h_sink.data(), h_sink.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	dim3 grid(QKVG_ROWS / Kernel::M_PER_BLOCK, SEQ_LEN / Kernel::N_PER_BLOCK);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	int warmup = 30;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_A,
			d_B,
			SEQ_LEN,
			d_C,
			d_S,
			d_sink,
			d_sink_scores
		);
	}
	cudaDeviceSynchronize();

	int NUM_RUNS = 100;
	std::vector<cudaEvent_t> starts(NUM_RUNS), ends(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventRecord(starts[i]);
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_A,
			d_B,
			SEQ_LEN,
			d_C,
			d_S,
			d_sink,
			d_sink_scores
		);
		cudaEventRecord(ends[i]);
	}
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	std::vector<float> times_ms(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventElapsedTime(&times_ms[i], starts[i], ends[i]);
		cudaEventDestroy(starts[i]);
		cudaEventDestroy(ends[i]);
	}
	std::sort(times_ms.begin(), times_ms.end());

	float median_ms = times_ms[NUM_RUNS / 2];
	float min_ms = times_ms[0];
	double strict_flops = 2.0 * QKVG_ROWS * config::qkv_fan_in * SEQ_LEN;
	double fake_flops = 2.0 * QKVG_ROWS * D_MODEL * SEQ_LEN;
	double strict_tflops = strict_flops / (median_ms * 1e-3) / 1e12;
	double fake_tflops = fake_flops / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", NUM_RUNS, median_ms, min_ms);
	printf("Strict TFLOPS (compact A): %.2f\n", strict_tflops);
	printf("Fake TFLOPS (full d_model): %.2f\n", fake_tflops);

	cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sink_scores.data(), d_sink_scores, h_sink_scores.size() * sizeof(f32), cudaMemcpyDeviceToHost);
	for (usize row = 0; row < SEQ_LEN; ++row) {
		bf16 const *src_row = h_C.data() + row * QKVG_ROWS;
		bf16 *q_row = h_Q.data() + row * Q_ROWS;
		bf16 *k_row = h_K.data() + row * Q_ROWS;
		bf16 *v_row = h_V.data() + row * Q_ROWS;
		bf16 *g_row = h_G.data() + row * Q_ROWS;
		std::copy_n(src_row + 0 * Q_ROWS, Q_ROWS, q_row);
		std::copy_n(src_row + 1 * Q_ROWS, Q_ROWS, k_row);
		std::copy_n(src_row + 2 * Q_ROWS, Q_ROWS, v_row);
		std::copy_n(src_row + 3 * Q_ROWS, Q_ROWS, g_row);
	}

	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/q.bin", h_Q, SEQ_LEN, Q_ROWS);
	store_tensor("tmp/block_cuda/k.bin", h_K, SEQ_LEN, Q_ROWS);
	store_tensor("tmp/block_cuda/v.bin", h_V, SEQ_LEN, Q_ROWS);
	store_tensor("tmp/block_cuda/g.bin", h_G, SEQ_LEN, Q_ROWS);
	store_tensor("tmp/block_cuda/qkvg.bin", h_C, SEQ_LEN, QKVG_ROWS);
	store_f32_tensor("tmp/block_cuda/sink_scores_f32.bin", h_sink_scores, config::n_heads, SEQ_LEN);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_S);
	cudaFree(d_sink);
	cudaFree(d_C);
	cudaFree(d_sink_scores);
	return 0;
}
