#include "block.config.hpp"
#include "utils2.cuh"
#include "cuda/utils_b8.cuh"
#include "cuda/gemm_b8.cuh"

#include <algorithm>
#include <filesystem>

using namespace config;

namespace Ffn_y_fwd {
	static constexpr usize Y_PROJ_OUTPUTS = 2 * MODEL_DIM;

	using InputLoader =
		b8::MatrixLoader<
			b8::FixedI8,
			F_WIDTH,
			64, 128
		>;

	using WeightLoader =
		b8::MatrixTransLoader<
			b8::MatrixLoader<
				b8::FixedI8,
				F_WIDTH,
				128, 128
			>
		>;

	static constexpr f64 Y_SCALE = math::constexpr_sqrt(math::fast::GELU_VAR_FIX_2 / f64(F_WIDTH));
	using Writer = b8::FixedI8MatrixResidualWriter<
		InputLoader::M,
		WeightLoader::K,
		Y_SCALE
	>;

	using Kernel = b8::Gemm<InputLoader, WeightLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		b8::FixedI8 *w,
		b8::FixedI8 *inp, usize n_inputs,
		b8::FixedI8 *residual,
		b8::FixedI8 *out
	) {
		auto a = InputLoader(inp, n_inputs);
		auto b = WeightLoader(w, Y_PROJ_OUTPUTS);
		auto o = Writer(out, residual, MODEL_DIM);
		Kernel().run(a, b, o);
	}
}

using namespace Ffn_y_fwd;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	if (!Kernel::has_full_output_tiles(seq_len, Y_PROJ_OUTPUTS)) {
		printf(
			"Expected pregate output shape [%u, %u] to align with block shape [%u, %u]\n",
			seq_len,
			Y_PROJ_OUTPUTS,
			Kernel::M_PER_BLOCK,
			Kernel::N_PER_BLOCK
		);
		return 1;
	}

	std::vector<b8::FixedI8> h_weights_compact = load_i8_tensor(
		torch_tensor_path("ffn_y_weights_i8.bin"),
		Y_PROJ_OUTPUTS,
		F_WIDTH
	);
	std::vector<b8::FixedI8> h_f = load_i8_tensor(tensor_path(cli.input_dir, "ffn_f_i8.bin"), seq_len, F_WIDTH);
	std::vector<b8::FixedI8> h_x = load_i8_tensor(tensor_path(cli.input_dir, "x_i8.bin"), seq_len, MODEL_DIM);
	if (h_weights_compact.empty() || h_f.empty() || h_x.empty()) {
		return 1;
	}

	std::vector<b8::FixedI8> h_out(seq_len * MODEL_DIM);

	b8::FixedI8 *d_weights = nullptr;
	b8::FixedI8 *d_f = nullptr;
	b8::FixedI8 *d_x = nullptr;
	b8::FixedI8 *d_out = nullptr;

	cudaMalloc(&d_weights, h_weights_compact.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_f, h_f.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_x, h_x.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_out, h_out.size() * sizeof(b8::FixedI8));

	cudaMemcpy(d_weights, h_weights_compact.data(), h_weights_compact.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f, h_f.data(), h_f.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid = Kernel::output_grid(seq_len, Y_PROJ_OUTPUTS);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_f,
			seq_len,
			d_x,
			d_out
		);
	}

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	int num_runs = 100;
	std::vector<cudaEvent_t> starts(num_runs), ends(num_runs);
	for (int i = 0; i < num_runs; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < num_runs; ++i) {
		//printf("run i = %d: %s\n", i, cudaGetErrorString(cudaGetLastError()));
		cudaEventRecord(starts[i]);
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_f,
			seq_len,
			d_x,
			d_out
		);
		cudaEventRecord(ends[i]);
	}
	cudaDeviceSynchronize();

	err = cudaGetLastError();
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
	double tflops = 2.0 * Y_PROJ_OUTPUTS * F_WIDTH * seq_len / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(b8::FixedI8), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_i8_tensor("tmp/block_cuda/ffn_y_i8.bin", h_out, seq_len, MODEL_DIM);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_f);
	cudaFree(d_x);
	cudaFree(d_out);
	return 0;
}
