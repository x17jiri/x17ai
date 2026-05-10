#include "cuda/gemm.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

constexpr usize SEQ_LEN = config::n_inputs;
constexpr usize D_MODEL = config::d_model;
constexpr usize F_WIDTH = config::f_width;

namespace Ffn_d_y_weights {
	using FLoader =
		MatrixTransLoader<
			MatrixLoader<
				config::f_width,
				64, 128
			>
		>;

	using DOutLoader =
		MatrixLoader<
			config::d_model,
			64, 64
		>;

	using Writer = MatrixWriter<config::f_width>;

	using Kernel = Gemm<FLoader, DOutLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *f,
		bf16 *d_out, usize n_inputs,
		bf16 *d_y_weights
	) {
		auto a = FLoader(f, n_inputs);
		auto b = DOutLoader(d_out, n_inputs);
		auto o = Writer(d_y_weights);
		Kernel().run(a, b, o);
	}
}

using namespace Ffn_d_y_weights;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	if (SEQ_LEN % Kernel::K_STEP != 0) {
		printf("Expected n_inputs %% %u == 0\n", Kernel::K_STEP);
		return 1;
	}
	if (F_WIDTH % Kernel::M_PER_BLOCK != 0) {
		printf("Expected f_width %% %u == 0\n", Kernel::M_PER_BLOCK);
		return 1;
	}
	if (D_MODEL % Kernel::N_PER_BLOCK != 0) {
		printf("Expected d_model %% %u == 0\n", Kernel::N_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_f = load_tensor(tensor_path(cli.input_dir, "ffn_f.bin"), SEQ_LEN, F_WIDTH);
	std::vector<bf16> h_d_out = load_tensor(torch_tensor_path("d_out.bin"), SEQ_LEN, D_MODEL);
	if (h_f.empty() || h_d_out.empty()) {
		return 1;
	}

	std::vector<bf16> h_d_y_weights(D_MODEL * F_WIDTH);

	bf16 *d_f = nullptr;
	bf16 *d_d_out = nullptr;
	bf16 *d_d_y_weights = nullptr;

	cudaMalloc(&d_f, h_f.size() * sizeof(bf16));
	cudaMalloc(&d_d_out, h_d_out.size() * sizeof(bf16));
	cudaMalloc(&d_d_y_weights, h_d_y_weights.size() * sizeof(bf16));

	cudaMemcpy(d_f, h_f.data(), h_f.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d_out, h_d_out.data(), h_d_out.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(F_WIDTH / Kernel::M_PER_BLOCK, D_MODEL / Kernel::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_f,
			d_d_out,
			SEQ_LEN,
			d_d_y_weights
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
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_f,
			d_d_out,
			SEQ_LEN,
			d_d_y_weights
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
	double tflops = 2.0 * D_MODEL * F_WIDTH * SEQ_LEN / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_d_y_weights.data(), d_d_y_weights, h_d_y_weights.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/ffn_d_y_weights.bin", h_d_y_weights, D_MODEL, F_WIDTH);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_f);
	cudaFree(d_d_out);
	cudaFree(d_d_y_weights);
	return 0;
}
