#include "block.config.hpp"
#include "utils2.cuh"
#include "cuda/utils_b8.cuh"
#include "cuda/gemm_b8.cuh"
#include "cuda/qkvg_fwd_i8.cuh"

#include <algorithm>
#include <filesystem>

using namespace config;

namespace Attn_qg_fwd {
	static constexpr usize QG_PROJ_OUTPUTS = QK_SEGMENT_SIZE + VG_SEGMENT_SIZE;

	using InputLoader =
		b8::MatrixLoader<
			b8::FixedI8,
			D_MODEL,
			64, 128,
			SPARSE_FAN_IN,
			QG_PROJ_OUTPUTS
		>;

	using WeightLoader =
		b8::MatrixTransLoader<
			b8::MatrixLoader<
				b8::FixedI8,
				SPARSE_FAN_IN,
				128, 128
			>
		>;

	using Writer = QGMatrixWriter<
		HEAD_DIM,
		QG_PROJ_OUTPUTS,
		SPARSE_FAN_IN
	>;

	using Kernel = b8::Gemm<InputLoader, WeightLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		b8::FixedI8 *w,
		b8::FixedI8 *inp, usize n_inputs,
		b8::FixedI8 *out,
		bf16 const *qk_norm_scales
	) {
		auto a = InputLoader(inp, n_inputs);
		auto b = WeightLoader(w, QG_PROJ_OUTPUTS);
		auto o = Writer(out, qk_norm_scales);
		Kernel().run(a, b, o);
	}
}

using namespace Attn_qg_fwd;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	if (!Kernel::has_full_output_tiles(seq_len, QG_PROJ_OUTPUTS)) {
		printf(
			"Expected QG output shape [%u, %u] to align with block shape [%u, %u]\n",
			seq_len,
			QG_PROJ_OUTPUTS,
			Kernel::M_PER_BLOCK,
			Kernel::N_PER_BLOCK
		);
		return 1;
	}

	std::vector<b8::FixedI8> h_q_weights = load_i8_tensor(
		torch_tensor_path("attn_q_weights_i8.bin"),
		QK_SEGMENT_SIZE,
		SPARSE_FAN_IN
	);
	std::vector<b8::FixedI8> h_g_weights = load_i8_tensor(
		torch_tensor_path("attn_g_weights_i8.bin"),
		VG_SEGMENT_SIZE,
		SPARSE_FAN_IN
	);
	std::vector<bf16> h_qk_norm_scales = load_tensor(
		torch_tensor_path("qk_norm_scales.bin"),
		1,
		QK_SEGMENT_SIZE
	);
	std::vector<b8::FixedI8> h_inputs = load_i8_tensor(
		tensor_path(cli.input_dir, "x_i8.bin"),
		seq_len,
		D_MODEL
	);
	if (h_q_weights.empty() || h_g_weights.empty() || h_qk_norm_scales.empty() || h_inputs.empty()) {
		return 1;
	}

	std::vector<b8::FixedI8> h_weights(QG_PROJ_OUTPUTS * SPARSE_FAN_IN);
	for (usize head = 0; head < N_HEADS; ++head) {
		usize src_off = head * HEAD_DIM * SPARSE_FAN_IN;
		usize dst_off = (2 * head) * HEAD_DIM * SPARSE_FAN_IN;
		std::copy_n(
			h_q_weights.data() + src_off,
			HEAD_DIM * SPARSE_FAN_IN,
			h_weights.data() + dst_off
		);
		std::copy_n(
			h_g_weights.data() + src_off,
			HEAD_DIM * SPARSE_FAN_IN,
			h_weights.data() + dst_off + HEAD_DIM * SPARSE_FAN_IN
		);
	}

	std::vector<b8::FixedI8> h_out(seq_len * QG_PROJ_OUTPUTS);

	b8::FixedI8 *d_weights = nullptr;
	b8::FixedI8 *d_inputs = nullptr;
	b8::FixedI8 *d_out = nullptr;
	bf16 *d_qk_norm_scales = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_inputs, h_inputs.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_out, h_out.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_qk_norm_scales, h_qk_norm_scales.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inputs, h_inputs.data(), h_inputs.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_qk_norm_scales, h_qk_norm_scales.data(), h_qk_norm_scales.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid = Kernel::output_grid(seq_len, QG_PROJ_OUTPUTS);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_inputs,
			seq_len,
			d_out,
			d_qk_norm_scales
		);
	}

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	int num_runs = 100;
	std::vector<cudaEvent_t> starts(num_runs), ends(num_runs);
	for (int i = 0; i < num_runs; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < num_runs; ++i) {
		cudaEventRecord(starts[i]);
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_inputs,
			seq_len,
			d_out,
			d_qk_norm_scales
		);
		cudaEventRecord(ends[i]);
	}
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	std::vector<float> times_ms(num_runs);
	for (int i = 0; i < num_runs; ++i) {
		cudaEventElapsedTime(&times_ms[i], starts[i], ends[i]);
		cudaEventDestroy(starts[i]);
		cudaEventDestroy(ends[i]);
	}
	std::sort(times_ms.begin(), times_ms.end());

	float median_ms = times_ms[num_runs / 2];
	float min_ms = times_ms[0];
	double tflops = 2.0 * QG_PROJ_OUTPUTS * SPARSE_FAN_IN * seq_len / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(b8::FixedI8), cudaMemcpyDeviceToHost);

	std::filesystem::create_directories("tmp/block_cuda");
	store_i8_tensor("tmp/block_cuda/qg_i8.bin", h_out, seq_len, QG_PROJ_OUTPUTS);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_inputs);
	cudaFree(d_out);
	cudaFree(d_qk_norm_scales);
	return 0;
}
