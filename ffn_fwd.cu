#include "cuda/gemm.cuh"

#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

constexpr usize SEQ_LEN = config::n_inputs;
constexpr usize D_MODEL = config::d_model;
constexpr usize F_WIDTH = config::f_width;
constexpr usize F_PROJ_OUTPUTS = 2 * F_WIDTH;

namespace FfnFwd {
	using WeightLoader =
		SparseMatrixLoader<
			config::d_model, // d_input,
			config::qkv_fan_in,
			config::head_dim, // cycle
			64, 64 // tile size
		>;

	using InputLoader =
		MatrixTransLoader<
			MatrixLoader<
				config::d_model, // d_input
				128, 64 // tile size
			>
		>;

	using Writer =
		MatrixGeGluWriter<
			config::f_width, // d_output
			config::d_model, // d_input
			config::qkv_fan_in
		>;

	using Kernel = Gemm<WeightLoader, InputLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *w,
		bf16 *inp, usize n_inputs,
		bf16 *out,
		bf16 *grad
	) {
		auto a = WeightLoader(w, 2*config::f_width);
		auto b = InputLoader(inp, n_inputs);
		auto o = Writer(out, grad);
		Kernel().run(a, b, o);
	}
}

using namespace FfnFwd;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	static_assert(config::d_model == config::n_heads * config::head_dim);

	if (SEQ_LEN % Kernel::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", Kernel::N_PER_BLOCK);
		return 1;
	}
	if (F_PROJ_OUTPUTS % Kernel::M_PER_BLOCK != 0) {
		printf("Expected 2 * f_width %% %u == 0\n", Kernel::M_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_weights = load_tensor(torch_tensor_path("f_weights.bin"), F_PROJ_OUTPUTS, config::qkv_fan_in);
	std::vector<bf16> h_inputs = load_tensor(torch_tensor_path("inputs_l2.bin"), SEQ_LEN, D_MODEL);
	if (h_weights.empty() || h_inputs.empty()) {
		return 1;
	}

	std::vector<bf16> h_out(SEQ_LEN * F_WIDTH);
	std::vector<bf16> h_backvec(SEQ_LEN * F_PROJ_OUTPUTS);

	bf16 *d_weights = nullptr;
	bf16 *d_inputs = nullptr;
	bf16 *d_out = nullptr;
	bf16 *d_backvec = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(bf16));
	cudaMalloc(&d_inputs, h_inputs.size() * sizeof(bf16));
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));
	cudaMalloc(&d_backvec, h_backvec.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inputs, h_inputs.data(), h_inputs.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(F_PROJ_OUTPUTS / Kernel::M_PER_BLOCK, SEQ_LEN / Kernel::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_inputs,
			SEQ_LEN,
			d_out,
			d_backvec
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
			d_inputs,
			SEQ_LEN,
			d_out,
			d_backvec
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
	double tflops = 2.0 * F_PROJ_OUTPUTS * D_MODEL * SEQ_LEN / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_backvec.data(), d_backvec, h_backvec.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/f.bin", h_out, SEQ_LEN, F_WIDTH);
	store_tensor("tmp/block_cuda/f_backvec.bin", h_backvec, SEQ_LEN, F_PROJ_OUTPUTS);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_inputs);
	cudaFree(d_out);
	cudaFree(d_backvec);
	return 0;
}
