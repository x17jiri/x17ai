#include "cuda/qkv_proj.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>

int main(int argc, char *argv[]) {
	constexpr usize A_ROWS = 4 * config::n_heads * config::head_dim;
	constexpr usize A_COLS = config::qkv_fan_in;
	constexpr usize B_ROWS = config::d_model;
	constexpr usize G_ROWS = config::n_heads;
	constexpr usize G_COLS = config::head_dim;
	usize B_COLS = config::n_inputs;

	using Proj = QKVProj<
		A_ROWS, A_COLS,
		B_ROWS,
		config::n_heads,
		config::head_dim,
		config::rope_dim,
		config::l2_norm_eps,
		config::rope_base
	>;

	printf("K_ITERS = %d\n", Proj::K_ITERS);

	constexpr usize C_ROWS = A_ROWS;
	usize C_COLS = B_COLS;

	if (
		C_ROWS % Proj::M_PER_BLOCK != 0 ||
		C_COLS % Proj::N_PER_BLOCK != 0
	) {
		printf("Expected M %% %u == 0 and N %% %u == 0\n", Proj::M_PER_BLOCK, Proj::N_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_A = load_tensor("tmp/block_torch/qkv_weights.bin", A_ROWS, A_COLS);
	std::vector<bf16> h_B = load_tensor("tmp/block_torch/inputs.bin", B_COLS, B_ROWS);
	std::vector<bf16> h_G = load_tensor("tmp/block_torch/g_weights.bin", G_ROWS, G_COLS);
	if (h_A.empty() || h_B.empty() || h_G.empty()) {
		return 1;
	}
	std::vector<bf16> h_C(C_ROWS * C_COLS);

	bf16 *d_A, *d_B, *d_G, *d_C;
	cudaMalloc(&d_A, h_A.size() * sizeof(bf16));
	cudaMalloc(&d_B, h_B.size() * sizeof(bf16));
	cudaMalloc(&d_G, h_G.size() * sizeof(bf16));
	cudaMalloc(&d_C, h_C.size() * sizeof(bf16));
	cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_G, h_G.data(), h_G.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	dim3 grid(C_ROWS / Proj::M_PER_BLOCK, C_COLS / Proj::N_PER_BLOCK);

	cudaFuncSetAttribute(qkv_proj<Proj>, cudaFuncAttributeMaxDynamicSharedMemorySize, Proj::SMEM_BYTES);
	cudaFuncSetAttribute(qkv_proj<Proj>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	int warmup = 30;
	for (int i = 0; i < warmup; ++i) {
		qkv_proj<Proj><<<grid, Proj::THREADS_PER_BLOCK, Proj::SMEM_BYTES>>>(d_A, d_B, d_G, d_C);
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
		qkv_proj<Proj><<<grid, Proj::THREADS_PER_BLOCK, Proj::SMEM_BYTES>>>(d_A, d_B, d_G, d_C);
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
	double strict_flops = 2.0 * A_ROWS * A_COLS * B_COLS;
	double fake_flops = 2.0 * A_ROWS * B_ROWS * B_COLS;
	double strict_tflops = strict_flops / (median_ms * 1e-3) / 1e12;
	double fake_tflops = fake_flops / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", NUM_RUNS, median_ms, min_ms);
	printf("Strict TFLOPS (compact A): %.2f\n", strict_tflops);
	printf("Fake TFLOPS (full d_model): %.2f\n", fake_tflops);

	cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(bf16), cudaMemcpyDeviceToHost);

	store_tensor("tmp/block_cuda/qkvg.bin", h_C, C_ROWS, C_COLS);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_G);
	cudaFree(d_C);
	return 0;
}
