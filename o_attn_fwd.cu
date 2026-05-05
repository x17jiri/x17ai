//#include "cuda/dense_matmul.cuh"
#include "cuda/base_matmul.cuh"

#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

constexpr usize SEQ_LEN = config::n_inputs;
constexpr usize D_MODEL = config::d_model;
constexpr usize ATTN_WIDTH = config::n_heads * config::head_dim;

using ALoader = MatrixLoader<ATTN_WIDTH, 128, 64>;
using BLoader = MatrixTransLoader<
	MatrixLoader<ATTN_WIDTH, 64, 64>
>;
using CWriter = MatrixWriter<config::d_model>;

using MyGemm = Gemm<ALoader, BLoader, CWriter>;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}


	static_assert(config::d_model == ATTN_WIDTH);

	if (SEQ_LEN % MyGemm::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", MyGemm::N_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_weights = load_tensor(torch_tensor_path("w_attn.bin"), D_MODEL, ATTN_WIDTH);
	std::vector<bf16> h_attn_out = load_tensor(tensor_path(cli.input_dir, "attn_out.bin"), SEQ_LEN, ATTN_WIDTH);
	if (h_weights.empty() || h_attn_out.empty()) {
		return 1;
	}

	std::vector<bf16> h_out(SEQ_LEN * D_MODEL);

	bf16 *d_weights = nullptr;
	bf16 *d_attn_out = nullptr;
	bf16 *d_out = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(bf16));
	cudaMalloc(&d_attn_out, h_attn_out.size() * sizeof(bf16));
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_attn_out, h_attn_out.data(), h_attn_out.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(gemm<MyGemm>, cudaFuncAttributeMaxDynamicSharedMemorySize, MyGemm::SMEM_BYTES);
	cudaFuncSetAttribute(gemm<MyGemm>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(D_MODEL / MyGemm::M_PER_BLOCK, SEQ_LEN / MyGemm::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		gemm<MyGemm><<<grid, MyGemm::THREADS_PER_BLOCK, MyGemm::SMEM_BYTES>>>(
			d_weights, D_MODEL,
			d_attn_out, SEQ_LEN,
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
		gemm<MyGemm><<<grid, MyGemm::THREADS_PER_BLOCK, MyGemm::SMEM_BYTES>>>(
			d_weights, D_MODEL,
			d_attn_out, SEQ_LEN,
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
	//double tflops = MyGemm::flops(D_MODEL, SEQ_LEN) / (median_ms * 1e-3) / 1e12;
	double tflops = 2.0 * config::d_model * config::n_heads * config::head_dim * config::n_inputs / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/o_attn.bin", h_out, SEQ_LEN, D_MODEL);

	printf("Used SMEM per kernel: %u\n", MyGemm::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_attn_out);
	cudaFree(d_out);
	return 0;
}
