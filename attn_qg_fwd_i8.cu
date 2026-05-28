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

	using QInputLoader =
		b8::MatrixLoader<
			b8::FixedI8,
			D_MODEL,
			64, 128,
			SPARSE_FAN_IN,
			QK_SEGMENT_SIZE
		>;

	using GInputLoader =
		b8::MatrixLoader<
			b8::FixedI8,
			D_MODEL,
			64, 128,
			SPARSE_FAN_IN,
			VG_SEGMENT_SIZE
		>;

	using QWeightLoader =
		b8::MatrixTransLoader<
			b8::MatrixLoader<
				b8::FixedI8,
				SPARSE_FAN_IN,
				128, 128
			>
		>;

	using GWeightLoader =
		b8::MatrixTransLoader<
			b8::MatrixLoader<
				b8::FixedI8,
				SPARSE_FAN_IN,
				128, 128
			>
		>;

	using QWriter = QKVGMatrixWriter<
		HEAD_DIM,
		QK_SEGMENT_SIZE,
		SPARSE_FAN_IN,
		true
	>;
	using GWriter = b8::FixedI8MatrixWriter<
		VG_SEGMENT_SIZE,
		math::constexpr_inv_sqrt(SPARSE_FAN_IN)
	>;

	using QKernel = b8::Gemm<QInputLoader, QWeightLoader, QWriter>;
	using GKernel = b8::Gemm<GInputLoader, GWeightLoader, GWriter>;

	X17_KERNEL(QKernel::THREADS_PER_BLOCK)
	void q_kernel(
		b8::FixedI8 *w,
		b8::FixedI8 *inp, usize n_inputs,
		b8::FixedI8 *out,
		bf16 const *qk_norm_scales
	) {
		auto a = QInputLoader(inp, n_inputs);
		auto b = QWeightLoader(w, QK_SEGMENT_SIZE);
		auto o = QWriter(out, qk_norm_scales);
		QKernel().run(a, b, o);
	}

	X17_KERNEL(GKernel::THREADS_PER_BLOCK)
	void g_kernel(
		b8::FixedI8 *w,
		b8::FixedI8 *inp, usize n_inputs,
		b8::FixedI8 *out
	) {
		auto a = GInputLoader(inp, n_inputs);
		auto b = GWeightLoader(w, VG_SEGMENT_SIZE);
		auto o = GWriter(out);
		GKernel().run(a, b, o);
	}
}

using namespace Attn_qg_fwd;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	if (
		!QKernel::has_full_output_tiles(seq_len, QK_SEGMENT_SIZE)
		|| !GKernel::has_full_output_tiles(seq_len, VG_SEGMENT_SIZE)
	) {
		printf(
			"Expected Q [%u, %u] and G [%u, %u] outputs to align with block shapes [%u, %u] and [%u, %u]\n",
			seq_len,
			QK_SEGMENT_SIZE,
			seq_len,
			VG_SEGMENT_SIZE,
			QKernel::M_PER_BLOCK,
			QKernel::N_PER_BLOCK,
			GKernel::M_PER_BLOCK,
			GKernel::N_PER_BLOCK
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

	std::vector<b8::FixedI8> h_out(seq_len * QG_PROJ_OUTPUTS);
	std::vector<b8::FixedI8> h_q(seq_len * QK_SEGMENT_SIZE);
	std::vector<b8::FixedI8> h_g(seq_len * VG_SEGMENT_SIZE);

	b8::FixedI8 *d_q_weights = nullptr;
	b8::FixedI8 *d_g_weights = nullptr;
	b8::FixedI8 *d_inputs = nullptr;
	b8::FixedI8 *d_q = nullptr;
	b8::FixedI8 *d_g = nullptr;
	bf16 *d_qk_norm_scales = nullptr;

	cudaMalloc(&d_q_weights, h_q_weights.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_g_weights, h_g_weights.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_inputs, h_inputs.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_q, h_q.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_g, h_g.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_qk_norm_scales, h_qk_norm_scales.size() * sizeof(bf16));

	cudaMemcpy(d_q_weights, h_q_weights.data(), h_q_weights.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_g_weights, h_g_weights.data(), h_g_weights.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inputs, h_inputs.data(), h_inputs.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_qk_norm_scales, h_qk_norm_scales.data(), h_qk_norm_scales.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(q_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, QKernel::SMEM_BYTES);
	cudaFuncSetAttribute(q_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
	cudaFuncSetAttribute(g_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, GKernel::SMEM_BYTES);
	cudaFuncSetAttribute(g_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 q_grid = QKernel::output_grid(seq_len, QK_SEGMENT_SIZE);
	dim3 g_grid = GKernel::output_grid(seq_len, VG_SEGMENT_SIZE);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		q_kernel<<<q_grid, QKernel::THREADS_PER_BLOCK, QKernel::SMEM_BYTES>>>(
			d_q_weights,
			d_inputs,
			seq_len,
			d_q,
			d_qk_norm_scales
		);
		g_kernel<<<g_grid, GKernel::THREADS_PER_BLOCK, GKernel::SMEM_BYTES>>>(
			d_g_weights,
			d_inputs,
			seq_len,
			d_g
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
		q_kernel<<<q_grid, QKernel::THREADS_PER_BLOCK, QKernel::SMEM_BYTES>>>(
			d_q_weights,
			d_inputs,
			seq_len,
			d_q,
			d_qk_norm_scales
		);
		g_kernel<<<g_grid, GKernel::THREADS_PER_BLOCK, GKernel::SMEM_BYTES>>>(
			d_g_weights,
			d_inputs,
			seq_len,
			d_g
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

	cudaMemcpy(h_q.data(), d_q, h_q.size() * sizeof(b8::FixedI8), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g.data(), d_g, h_g.size() * sizeof(b8::FixedI8), cudaMemcpyDeviceToHost);
	for (usize row = 0; row < seq_len; ++row) {
		auto *dst_row = h_out.data() + row * QG_PROJ_OUTPUTS;
		auto *q_row = h_q.data() + row * QK_SEGMENT_SIZE;
		auto *g_row = h_g.data() + row * VG_SEGMENT_SIZE;
		std::copy_n(q_row, QK_SEGMENT_SIZE, dst_row);
		std::copy_n(g_row, VG_SEGMENT_SIZE, dst_row + QK_SEGMENT_SIZE);
	}

	std::filesystem::create_directories("tmp/block_cuda");
	store_i8_tensor("tmp/block_cuda/qg_i8.bin", h_out, seq_len, QG_PROJ_OUTPUTS);
	store_i8_tensor("tmp/block_cuda/q_i8.bin", h_q, seq_len, QK_SEGMENT_SIZE);
	store_i8_tensor("tmp/block_cuda/g_i8.bin", h_g, seq_len, VG_SEGMENT_SIZE);

	printf("Used SMEM per Q kernel: %u\n", QKernel::SMEM_BYTES);
	printf("Used SMEM per G kernel: %u\n", GKernel::SMEM_BYTES);

	cudaFree(d_q_weights);
	cudaFree(d_g_weights);
	cudaFree(d_inputs);
	cudaFree(d_q);
	cudaFree(d_g);
	cudaFree(d_qk_norm_scales);
	return 0;
}
