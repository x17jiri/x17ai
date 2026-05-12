#include "cuda/gemm.cuh"
#include "cuda/qkvg_fwd.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

using namespace config;

namespace QkvgFwd {
	using WeightLoader =
		SparseMatrixLoader<
			D_MODEL,
			SPARSE_FAN_IN,
			SPARSE_CYCLE,
			64, 64
		>;

	using InputLoader =
		MatrixTransLoader<
			MatrixLoader<
				D_MODEL,
				128, 64
			>
		>;

	using Writer =
		MatrixQKVGWriter<
			QKVG_ROWS,
			D_MODEL,
			SPARSE_FAN_IN,
			N_HEADS,
			QK_DIM,
			VG_DIM,
			L2_NORM_EPS,
			V_SCALE_FIX
		>;

	using Kernel = Gemm<WeightLoader, InputLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *w,
		bf16 *inp, usize n_inputs,
		bf16 *out,
		bf16 const *qk_norm_scales
	) {
		auto a = WeightLoader(w, QKVG_ROWS);
		auto b = InputLoader(inp, n_inputs);
		auto o = Writer(out, qk_norm_scales);
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
		seq_len % Kernel::N_PER_BLOCK != 0
	) {
		printf("Expected M %% %u == 0 and N %% %u == 0\n", Kernel::M_PER_BLOCK, Kernel::N_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_A = load_tensor(torch_tensor_path("qkvg_weights.bin"), QKVG_ROWS, SPARSE_FAN_IN);
	std::vector<bf16> h_B = load_tensor(torch_tensor_path("inputs_l2.bin"), seq_len, D_MODEL);
	std::vector<bf16> h_S = load_tensor(torch_tensor_path("qk_norm_scales.bin"), 1, QK_SEGMENT_SIZE);
	if (h_A.empty() || h_B.empty() || h_S.empty()) {
		return 1;
	}
	std::vector<bf16> h_C(seq_len * QKVG_ROWS);
	std::vector<bf16> h_Q(seq_len * QK_SEGMENT_SIZE);
	std::vector<bf16> h_K(seq_len * QK_SEGMENT_SIZE);
	std::vector<bf16> h_V(seq_len * VG_SEGMENT_SIZE);
	std::vector<bf16> h_G(seq_len * VG_SEGMENT_SIZE);

	bf16 *d_A, *d_B, *d_S, *d_C;
	cudaMalloc(&d_A, h_A.size() * sizeof(bf16));
	cudaMalloc(&d_B, h_B.size() * sizeof(bf16));
	cudaMalloc(&d_S, h_S.size() * sizeof(bf16));
	cudaMalloc(&d_C, h_C.size() * sizeof(bf16));
	cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_S, h_S.data(), h_S.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	dim3 grid(QKVG_ROWS / Kernel::M_PER_BLOCK, seq_len / Kernel::N_PER_BLOCK);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	int warmup = 30;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_A,
			d_B,
			seq_len,
			d_C,
			d_S
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
			seq_len,
			d_C,
			d_S
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
	double strict_flops = 2.0 * QKVG_ROWS * SPARSE_FAN_IN * seq_len;
	double fake_flops = 2.0 * QKVG_ROWS * D_MODEL * seq_len;
	double strict_tflops = strict_flops / (median_ms * 1e-3) / 1e12;
	double fake_tflops = fake_flops / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", NUM_RUNS, median_ms, min_ms);
	printf("Strict TFLOPS (compact A): %.2f\n", strict_tflops);
	printf("Fake TFLOPS (full d_model): %.2f\n", fake_tflops);

	cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	for (usize row = 0; row < seq_len; ++row) {
		bf16 const *src_row = h_C.data() + row * QKVG_ROWS;
		bf16 *q_row = h_Q.data() + row * QK_SEGMENT_SIZE;
		bf16 *k_row = h_K.data() + row * QK_SEGMENT_SIZE;
		bf16 *v_row = h_V.data() + row * VG_SEGMENT_SIZE;
		bf16 *g_row = h_G.data() + row * VG_SEGMENT_SIZE;

		usize q_off = 0;
		usize k_off = q_off + QK_SEGMENT_SIZE;
		usize v_off = k_off + QK_SEGMENT_SIZE;
		usize g_off = v_off + VG_SEGMENT_SIZE;

		std::copy_n(src_row + q_off, QK_SEGMENT_SIZE, q_row);
		std::copy_n(src_row + k_off, QK_SEGMENT_SIZE, k_row);
		std::copy_n(src_row + v_off, VG_SEGMENT_SIZE, v_row);
		std::copy_n(src_row + g_off, VG_SEGMENT_SIZE, g_row);
	}

	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/q.bin", h_Q, seq_len, QK_SEGMENT_SIZE);
	store_tensor("tmp/block_cuda/k.bin", h_K, seq_len, QK_SEGMENT_SIZE);
	store_tensor("tmp/block_cuda/v.bin", h_V, seq_len, VG_SEGMENT_SIZE);
	store_tensor("tmp/block_cuda/g.bin", h_G, seq_len, VG_SEGMENT_SIZE);
	store_tensor("tmp/block_cuda/qkvg.bin", h_C, seq_len, QKVG_ROWS);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_S);
	cudaFree(d_C);
	return 0;
}
