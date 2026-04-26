#include "cuda/o_proj.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

int main() {
	constexpr usize SEQ_LEN = config::n_inputs;
	constexpr usize D_MODEL = config::d_model;
	constexpr usize ATTN_WIDTH = config::n_heads * config::head_dim;
	constexpr usize F_WIDTH = config::f_width;
	constexpr usize O_PROJ_INPUT_ROWS = ATTN_WIDTH + F_WIDTH;

	static_assert(config::d_model == ATTN_WIDTH);

	using OP = OProj<D_MODEL, O_PROJ_INPUT_ROWS>;

	if (SEQ_LEN % OP::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", OP::N_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_weights = load_tensor("tmp/block_torch/o_weights.bin", D_MODEL, O_PROJ_INPUT_ROWS);
	std::vector<bf16> h_attn_out = load_tensor("tmp/block_torch/attn_out.bin", SEQ_LEN, ATTN_WIDTH);
	std::vector<bf16> h_f = load_tensor("tmp/block_torch/f.bin", SEQ_LEN, F_WIDTH);
	if (h_weights.empty() || h_attn_out.empty() || h_f.empty()) {
		return 1;
	}

	std::vector<bf16> h_inputs(SEQ_LEN * O_PROJ_INPUT_ROWS);
	for (usize row = 0; row < SEQ_LEN; ++row) {
		bf16 *dst = h_inputs.data() + row * O_PROJ_INPUT_ROWS;
		std::copy_n(h_attn_out.data() + row * ATTN_WIDTH, ATTN_WIDTH, dst);
		std::copy_n(h_f.data() + row * F_WIDTH, F_WIDTH, dst + ATTN_WIDTH);
	}

	std::vector<bf16> h_out(SEQ_LEN * D_MODEL);

	bf16 *d_weights = nullptr;
	bf16 *d_inputs = nullptr;
	bf16 *d_out = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(bf16));
	cudaMalloc(&d_inputs, h_inputs.size() * sizeof(bf16));
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inputs, h_inputs.data(), h_inputs.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(o_proj<OP>, cudaFuncAttributeMaxDynamicSharedMemorySize, OP::SMEM_BYTES);
	cudaFuncSetAttribute(o_proj<OP>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(D_MODEL / OP::M_PER_BLOCK, SEQ_LEN / OP::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		o_proj<OP><<<grid, OP::THREADS_PER_BLOCK, OP::SMEM_BYTES>>>(
			SEQ_LEN,
			d_weights,
			d_inputs,
			d_out
		);
	}
	cudaDeviceSynchronize();

	int num_runs = 100;
	std::vector<cudaEvent_t> starts(num_runs), ends(num_runs);
	for (int i = 0; i < num_runs; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < num_runs; ++i) {
		cudaEventRecord(starts[i]);
		o_proj<OP><<<grid, OP::THREADS_PER_BLOCK, OP::SMEM_BYTES>>>(
			SEQ_LEN,
			d_weights,
			d_inputs,
			d_out
		);
		cudaEventRecord(ends[i]);
	}
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
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
	double tflops = OP::flops(SEQ_LEN) / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/o.bin", h_out, SEQ_LEN, D_MODEL);

	printf("Used SMEM per kernel: %u\n", OP::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_inputs);
	cudaFree(d_out);
	return 0;
}
