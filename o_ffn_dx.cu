#include "cuda/dense_matmul.cuh"
#include "cuda/dense_matmul_dx.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	constexpr usize SEQ_LEN = config::n_inputs;
	constexpr usize D_MODEL = config::d_model;
	constexpr usize F_WIDTH = config::f_width;

	using MatMul = DenseMatMul_dx<F_WIDTH, D_MODEL>;

	if (SEQ_LEN % MatMul::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", MatMul::N_PER_BLOCK);
		return 1;
	}
	if (F_WIDTH % MatMul::M_PER_BLOCK != 0) {
		printf("Expected f_width %% %u == 0\n", MatMul::M_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_weights = load_tensor(torch_tensor_path("w_ffn.bin"), D_MODEL, F_WIDTH);
	std::vector<bf16> h_d_o_ffn = load_tensor(tensor_path(cli.input_dir, "d_o_ffn.bin"), SEQ_LEN, D_MODEL);
	if (h_weights.empty() || h_d_o_ffn.empty()) {
		return 1;
	}

	std::vector<bf16> h_d_f(SEQ_LEN * F_WIDTH);

	bf16 *d_weights = nullptr;
	bf16 *d_d_o_ffn = nullptr;
	bf16 *d_d_f = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(bf16));
	cudaMalloc(&d_d_o_ffn, h_d_o_ffn.size() * sizeof(bf16));
	cudaMalloc(&d_d_f, h_d_f.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d_o_ffn, h_d_o_ffn.data(), h_d_o_ffn.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(matmul_dx<MatMul>, cudaFuncAttributeMaxDynamicSharedMemorySize, MatMul::SMEM_BYTES);
	cudaFuncSetAttribute(matmul_dx<MatMul>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(F_WIDTH / MatMul::M_PER_BLOCK, SEQ_LEN / MatMul::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		matmul_dx<MatMul><<<grid, MatMul::THREADS_PER_BLOCK, MatMul::SMEM_BYTES>>>(
			d_weights,
			d_d_o_ffn,
			d_d_f
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
		matmul_dx<MatMul><<<grid, MatMul::THREADS_PER_BLOCK, MatMul::SMEM_BYTES>>>(
			d_weights,
			d_d_o_ffn,
			d_d_f
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
	double tflops = MatMul::flops(F_WIDTH, SEQ_LEN) / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_d_f.data(), d_d_f, h_d_f.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/d_f.bin", h_d_f, SEQ_LEN, F_WIDTH);

	printf("Used SMEM per kernel: %u\n", MatMul::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_d_o_ffn);
	cudaFree(d_d_f);
	return 0;
}
