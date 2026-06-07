#include "cuda/gemm_b16.cuh"

#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

using namespace config;

namespace OAttnFwd {
	using InputLoader =
		b16::MatrixTransLoader<
			b16::MatrixLoader<
				bf16,
				ATTN_WIDTH,
				64, 128
			>
		>;

	using WeightLoader =
		b16::MatrixLoader<
			bf16,
			ATTN_WIDTH,
			64, 64
		>;

	using Writer = b16::Bf16MatrixWriter<
		MODEL_DIM,
		InputLoader::M,
		WeightLoader::K
	>;

	using Kernel = b16::Gemm<InputLoader, WeightLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *w,
		bf16 *inp, usize n_inputs,
		bf16 *out
	) {
		auto a = InputLoader(inp, n_inputs);
		auto b = WeightLoader(w, MODEL_DIM);
		auto o = Writer(out);
		Kernel().run(a, b, o);
	}
}

using namespace OAttnFwd;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	if (seq_len % Kernel::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", Kernel::N_PER_BLOCK);
		return 1;
	}
	if (MODEL_DIM % Kernel::M_PER_BLOCK != 0) {
		printf("Expected d_model %% %u == 0\n", Kernel::M_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_weights = load_tensor(torch_tensor_path("attn_y_weights.bin"), MODEL_DIM, ATTN_WIDTH);
	std::vector<bf16> h_attn_out = load_tensor(tensor_path(cli.input_dir, "attn_out.bin"), seq_len, ATTN_WIDTH);
	if (h_weights.empty() || h_attn_out.empty()) {
		return 1;
	}

	std::vector<bf16> h_out(seq_len * MODEL_DIM);

	bf16 *d_weights = nullptr;
	bf16 *d_attn_out = nullptr;
	bf16 *d_out = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(bf16));
	cudaMalloc(&d_attn_out, h_attn_out.size() * sizeof(bf16));
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_attn_out, h_attn_out.data(), h_attn_out.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(MODEL_DIM / Kernel::M_PER_BLOCK, seq_len / Kernel::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_attn_out,
			seq_len,
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
			d_attn_out,
			seq_len,
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
	double tflops = 2.0 * MODEL_DIM * ATTN_WIDTH * seq_len / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/o_attn.bin", h_out, seq_len, MODEL_DIM);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_attn_out);
	cudaFree(d_out);
	return 0;
}
