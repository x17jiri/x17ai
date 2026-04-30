#include "cuda/gemm.cuh"
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

	using OP = Gemm<F_WIDTH, D_MODEL>;

	if (SEQ_LEN % OP::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", OP::N_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_weights = load_tensor(torch_tensor_path("w_ffn.bin"), D_MODEL, F_WIDTH);
	std::vector<bf16> h_f = load_tensor(tensor_path(cli.input_dir, "f.bin"), SEQ_LEN, F_WIDTH);
	if (h_weights.empty() || h_f.empty()) {
		return 1;
	}

	std::vector<bf16> h_out(SEQ_LEN * D_MODEL);

	bf16 *d_weights = nullptr;
	bf16 *d_f = nullptr;
	bf16 *d_out = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(bf16));
	cudaMalloc(&d_f, h_f.size() * sizeof(bf16));
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f, h_f.data(), h_f.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(matmul<OP>, cudaFuncAttributeMaxDynamicSharedMemorySize, OP::SMEM_BYTES);
	cudaFuncSetAttribute(matmul<OP>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(D_MODEL / OP::M_PER_BLOCK, SEQ_LEN / OP::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		matmul<OP><<<grid, OP::THREADS_PER_BLOCK, OP::SMEM_BYTES>>>(
			SEQ_LEN,
			d_weights,
			d_f,
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
		matmul<OP><<<grid, OP::THREADS_PER_BLOCK, OP::SMEM_BYTES>>>(
			SEQ_LEN,
			d_weights,
			d_f,
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
	store_tensor("tmp/block_cuda/o_ffn.bin", h_out, SEQ_LEN, D_MODEL);

	printf("Used SMEM per kernel: %u\n", OP::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_f);
	cudaFree(d_out);
	return 0;
}
