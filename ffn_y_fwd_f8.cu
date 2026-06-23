#include "block.config.hpp"
#include "utils2.cuh"
#include "cuda/gemm_b16.cuh"
#include "cuda/gemm_b8.cuh"

#include <algorithm>
#include <filesystem>

using namespace config;

namespace Ffn_y_fwd_f8 {
	static constexpr usize Y_PROJ_OUTPUTS = 2 * MODEL_DIM;
	static constexpr f64 OUTPUT_SCALE =
		math::constexpr_sqrt(math::fast::GELU_VAR_FIX_2 / f64(F_WIDTH))
		/ f64(b8::FIXED_I8_SCALE);

	using InputLoader =
		b16::E4m3MatrixLoader<
			F_WIDTH,
			64, 64
		>;

	using WeightLoader =
		b16::MatrixTransLoader<
			b16::FixedI8MatrixLoader<
				F_WIDTH,
				128, 64
			>
	>;

	using Writer = b8::ResidualMatrixWriter<
		b8::Int8Store,
		InputLoader::M,
		WeightLoader::K,
		OUTPUT_SCALE
	>;

	using Kernel = b16::Gemm<InputLoader, WeightLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		b8::FixedI8 *w,
		b8::E4m3 *inp, usize n_inputs,
		FixedI8 *residual,
		FixedI8 *out
	) {
		auto a = InputLoader(inp, n_inputs);
		auto b = WeightLoader(w, Y_PROJ_OUTPUTS);
		auto o = Writer(out, residual, MODEL_DIM);
		Kernel().run(a, b, o);
	}
}

using namespace Ffn_y_fwd_f8;

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

	std::vector<b8::FixedI8> h_weights_i8 = load_i8_tensor(
		torch_tensor_path("ffn_y_weights_i8.bin"),
		Y_PROJ_OUTPUTS,
		F_WIDTH
	);
	std::vector<b8::E4m3> h_f = load_e4m3_tensor(tensor_path(cli.input_dir, "ffn_f_f8.bin"), seq_len, F_WIDTH);
	std::vector<b8::FixedI8> h_x = load_i8_tensor(tensor_path(cli.input_dir, "x_i8.bin"), seq_len, MODEL_DIM);
	if (h_weights_i8.empty() || h_f.empty() || h_x.empty()) {
		return 1;
	}

	std::vector<FixedI8> h_out(seq_len * MODEL_DIM);

	b8::FixedI8 *d_weights = nullptr;
	b8::E4m3 *d_f = nullptr;
	FixedI8 *d_x = nullptr;
	FixedI8 *d_out = nullptr;

	cudaMalloc(&d_weights, h_weights_i8.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_f, h_f.size() * sizeof(b8::E4m3));
	cudaMalloc(&d_x, h_x.size() * sizeof(FixedI8));
	cudaMalloc(&d_out, h_out.size() * sizeof(FixedI8));

	cudaMemcpy(d_weights, h_weights_i8.data(), h_weights_i8.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f, h_f.data(), h_f.size() * sizeof(b8::E4m3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(FixedI8), cudaMemcpyHostToDevice);

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

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(FixedI8), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_i8_tensor("tmp/block_cuda/ffn_y_i8.bin", h_out, seq_len, MODEL_DIM);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_f);
	cudaFree(d_x);
	cudaFree(d_out);
	return 0;
}
