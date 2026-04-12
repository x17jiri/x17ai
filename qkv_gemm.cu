#include "cuda/qkv_proj.cuh"
#include "block.config.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <vector>
#include <fstream>

std::vector<bf16> load_tensor(std::string const &filename, usize rows, usize cols) {
	std::vector<bf16> data(rows * cols);
	std::ifstream a_in(filename, std::ios::binary);
	if (!a_in) {
		printf("Failed to open %s\n", filename.c_str());
		return {};
	}
	if (!a_in.read(
		reinterpret_cast<char *>(data.data()),
		static_cast<std::streamsize>(data.size() * sizeof(bf16))
	)) {
		printf("Failed to read %s as [%u, %u]\n", filename.c_str(), rows, cols);
		return {};
	}
	return data;
}

void store_tensor(
	std::string const &filename,
	std::vector<bf16> const &data,
	[[maybe_unused]] usize rows, [[maybe_unused]] usize cols
) {
	std::ofstream out(filename, std::ios::binary);
	if (!out) {
		printf("Failed to open %s for writing\n", filename.c_str());
		return;
	}
	if (!out.write(
		reinterpret_cast<const char *>(data.data()),
		static_cast<std::streamsize>(data.size() * sizeof(bf16))
	)) {
		printf("Failed to write data to %s\n", filename.c_str());
	}
	printf("Wrote output to %s\n", filename.c_str());
}

int main(int argc, char *argv[]) {
	constexpr usize A_ROWS = 3 * config::n_heads * config::head_dim;
	constexpr usize A_COLS = config::qkv_fan_in;
	constexpr usize B_ROWS = config::d_model;
	usize B_COLS = config::n_inputs;

	using Proj = QKVProj<
		A_ROWS, A_COLS,
		B_ROWS,
		config::n_heads,
		config::head_dim,
		config::rope_dim, config::rope_base
	>;

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
	if (h_A.empty() || h_B.empty()) {
		return 1;
	}
	std::vector<bf16> h_C(C_ROWS * C_COLS);

	bf16 *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, h_A.size() * sizeof(bf16));
	cudaMalloc(&d_B, h_B.size() * sizeof(bf16));
	cudaMalloc(&d_C, h_C.size() * sizeof(bf16));
	cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	dim3 grid(C_ROWS / Proj::M_PER_BLOCK, C_COLS / Proj::N_PER_BLOCK);

	cudaFuncSetAttribute(qkv_proj<Proj>, cudaFuncAttributeMaxDynamicSharedMemorySize, Proj::SMEM_BYTES);
	cudaFuncSetAttribute(qkv_proj<Proj>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	int warmup = 30;
	for (int i = 0; i < warmup; ++i) {
		qkv_proj<Proj><<<grid, Proj::THREADS_PER_BLOCK, Proj::SMEM_BYTES>>>(d_A, d_B, d_C);
	}
	cudaDeviceSynchronize();

	GPU_Clock timer;
	timer.start();
	int NUM_RUNS = 100;
	for (int i = 0; i < NUM_RUNS; ++i) {
		qkv_proj<Proj><<<grid, Proj::THREADS_PER_BLOCK, Proj::SMEM_BYTES>>>(d_A, d_B, d_C);
	}
	cudaDeviceSynchronize();
	double elapsed = timer.seconds() / NUM_RUNS;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	double strict_flops = 2.0 * A_ROWS * A_COLS * B_COLS;
	double logical_flops = 2.0 * A_ROWS * B_ROWS * B_COLS;
	double strict_tflops = strict_flops / elapsed / 1e12;
	double logical_tflops = logical_flops / elapsed / 1e12;
	printf("Average kernel time over %d runs: %.3f ms\n", NUM_RUNS, elapsed * 1e3);
	printf("Strict TFLOPS (compact A): %.2f\n", strict_tflops);
	printf("Logical TFLOPS (full d_model): %.2f\n", logical_tflops);

	cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(bf16), cudaMemcpyDeviceToHost);

	store_tensor("tmp/block_cuda/qkv.bin", h_C, C_ROWS, C_COLS);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
