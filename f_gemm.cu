#include "cuda/f_proj.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

int main() {
	constexpr usize SEQ_LEN = config::n_inputs;
	constexpr usize D_MODEL = config::d_model;
	constexpr usize F_WIDTH = config::f_width;
	constexpr usize F_PROJ_OUTPUTS = 2 * F_WIDTH;

	static_assert(config::d_model == config::n_heads * config::head_dim);

	using FP = FProj<F_WIDTH, D_MODEL>;

	if (SEQ_LEN % FP::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", FP::N_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_weights = load_tensor("tmp/block_torch/f_weights.bin", F_PROJ_OUTPUTS, D_MODEL);
	std::vector<bf16> h_inputs = load_tensor("tmp/block_torch/inputs_l2.bin", SEQ_LEN, D_MODEL);
	if (h_weights.empty() || h_inputs.empty()) {
		return 1;
	}

	std::vector<bf16> h_out(SEQ_LEN * F_WIDTH);

	bf16 *d_weights = nullptr;
	bf16 *d_inputs = nullptr;
	bf16 *d_out = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(bf16));
	cudaMalloc(&d_inputs, h_inputs.size() * sizeof(bf16));
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inputs, h_inputs.data(), h_inputs.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(f_proj<FP>, cudaFuncAttributeMaxDynamicSharedMemorySize, FP::SMEM_BYTES);
	cudaFuncSetAttribute(f_proj<FP>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(F_PROJ_OUTPUTS / FP::M_PER_BLOCK, SEQ_LEN / FP::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		f_proj<FP><<<grid, FP::THREADS_PER_BLOCK, FP::SMEM_BYTES>>>(
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
		f_proj<FP><<<grid, FP::THREADS_PER_BLOCK, FP::SMEM_BYTES>>>(
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
	double tflops = FP::flops(SEQ_LEN) / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/f.bin", h_out, SEQ_LEN, F_WIDTH);

	printf("Used SMEM per kernel: %u\n", FP::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_inputs);
	cudaFree(d_out);
	return 0;
}
