#include "cuda/gemm.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

constexpr usize SEQ_LEN = config::n_inputs;
constexpr usize D_MODEL = config::d_model;
constexpr usize F_WIDTH = config::f_width;

namespace OFfnFwd {
	using WeightLoader =
		MatrixLoader<
			config::f_width,
			128, 64
		>;

	using InputLoader =
		MatrixTransLoader<
			MatrixLoader<
				config::f_width,
				64, 64
			>
		>;

	using Writer = MatrixWriter<config::d_model>;

	using Kernel = Gemm<WeightLoader, InputLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *w,
		bf16 *inp, usize n_inputs,
		bf16 *out
	) {
		auto a = WeightLoader(w, config::d_model);
		auto b = InputLoader(inp, n_inputs);
		auto o = Writer(out);
		Kernel().run(a, b, o);
	}
}

using namespace OFfnFwd;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	if (SEQ_LEN % Kernel::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", Kernel::N_PER_BLOCK);
		return 1;
	}
	if (D_MODEL % Kernel::M_PER_BLOCK != 0) {
		printf("Expected d_model %% %u == 0\n", Kernel::M_PER_BLOCK);
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

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(D_MODEL / Kernel::M_PER_BLOCK, SEQ_LEN / Kernel::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_f,
			SEQ_LEN,
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
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_f,
			SEQ_LEN,
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
	double tflops = 2.0 * D_MODEL * F_WIDTH * SEQ_LEN / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/o_ffn.bin", h_out, SEQ_LEN, D_MODEL);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_f);
	cudaFree(d_out);
	return 0;
}
