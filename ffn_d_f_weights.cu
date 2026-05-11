#include "cuda/gemm.cuh"

#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

constexpr usize SEQ_LEN = config::n_inputs;
constexpr usize D_MODEL = config::d_model;
constexpr usize F_WIDTH = config::f_width;
constexpr usize F_PROJ_OUTPUTS = 2 * F_WIDTH;
constexpr usize FAN_IN = config::qkv_fan_in;
constexpr usize INPUT_STEP = D_MODEL / config::head_dim;

namespace Ffn_d_f_weights {
	using XLoader =
		MatrixTransLoader<
			MatrixLoader<
				config::d_model,
				64, 64
			>
		>;

	using DtLoader = GeGluBackwardLoader<F_PROJ_OUTPUTS, 64, 64>;

	template<const usize D_IN, const usize _FAN_IN, const usize STEP>
	struct SparseWriter {
		bf16 *gC;

		static_assert(D_IN % 16 == 0);
		static_assert(_FAN_IN % 16 == 0);
		static_assert(STEP % 16 == 0);

		X17_DEVICE SparseWriter(bf16 *gC):
			gC(gC)
		{}

		X17_DEVICE void write_8x8(
			usize proj_row_base,
			usize dense_col_base,
			Fragment_8x8<f32> const &frag
		) {
			usize lane = threadIdx.x % WARP_SIZE;
			usize local_row = lane / 4;
			usize local_col_pair = lane % 4;

			usize proj_row = proj_row_base + local_row;
			usize dense_col = dense_col_base + 2 * local_col_pair;
			usize dense_col_start = (proj_row * STEP) % D_IN;
			usize compact_col = (dense_col + D_IN - dense_col_start) % D_IN;
			if (compact_col >= _FAN_IN) {
				return;
			}

			usize out_idx = proj_row * _FAN_IN + compact_col;
			FragmentReg<bf16> packed;
			packed.set(
				round_cast<bf16>(frag.first()),
				round_cast<bf16>(frag.second())
			);
			reinterpret_cast<u32 *>(gC)[out_idx / 2] = packed.val;
		}

		template<const usize ROW_TILES, const usize COL_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			Fragment_16x16<f32> (&acc)[ROW_TILES][COL_TILES]
		) {
			X17_UNROLL for (usize row_tile = 0; row_tile < ROW_TILES; ++row_tile) {
				usize proj_row = row + 16 * row_tile;
				X17_UNROLL for (usize col_tile = 0; col_tile < COL_TILES; ++col_tile) {
					usize dense_col = col + 16 * col_tile;
					auto const &tile = acc[row_tile][col_tile];

					write_8x8(proj_row + 0, dense_col + 0, tile.sub[0][0]);
					write_8x8(proj_row + 0, dense_col + 8, tile.sub[0][1]);
					write_8x8(proj_row + 8, dense_col + 0, tile.sub[1][0]);
					write_8x8(proj_row + 8, dense_col + 8, tile.sub[1][1]);
				}
			}
		}
	};

	using Writer = SparseWriter<config::d_model, config::qkv_fan_in, INPUT_STEP>;
	using Kernel = Gemm<XLoader, DtLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *x, usize n_inputs,
		bf16 *d_f,
		bf16 *backvec,
		bf16 *d_f_weights
	) {
		auto a = XLoader(x, n_inputs);
		auto b = DtLoader(d_f, backvec, n_inputs);
		auto o = Writer(d_f_weights);
		Kernel().run(a, b, o);
	}
}

using namespace Ffn_d_f_weights;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	static_assert(D_MODEL % config::head_dim == 0);
	static_assert(FAN_IN <= D_MODEL);

	if (SEQ_LEN % Kernel::K_STEP != 0) {
		printf("Expected n_inputs %% %u == 0\n", Kernel::K_STEP);
		return 1;
	}
	if (F_PROJ_OUTPUTS % Kernel::N_PER_BLOCK != 0) {
		printf("Expected 2 * f_width %% %u == 0\n", Kernel::N_PER_BLOCK);
		return 1;
	}
	if (D_MODEL % Kernel::M_PER_BLOCK != 0) {
		printf("Expected d_model %% %u == 0\n", Kernel::M_PER_BLOCK);
		return 1;
	}

	std::string d_f_path = tensor_path(cli.input_dir, "ffn_d_f.bin");
	std::string backvec_path = tensor_path(cli.input_dir, "ffn_f_backvec.bin");
	if (!std::filesystem::exists(backvec_path)) {
		backvec_path = tensor_path("tmp/block_cuda", "ffn_f_backvec.bin");
	}

	std::vector<bf16> h_x = load_tensor(torch_tensor_path("inputs_l2.bin"), SEQ_LEN, D_MODEL);
	std::vector<bf16> h_d_f = load_tensor(d_f_path, SEQ_LEN, F_WIDTH);
	std::vector<bf16> h_backvec = load_tensor(backvec_path, SEQ_LEN, F_PROJ_OUTPUTS);
	if (h_x.empty() || h_d_f.empty() || h_backvec.empty()) {
		if (h_backvec.empty()) {
			printf(
				"Expected %s (or tmp/block_cuda/ffn_f_backvec.bin) before running ffn_d_f_weights.cu\n",
				tensor_path(cli.input_dir, "ffn_f_backvec.bin").c_str()
			);
		}
		return 1;
	}

	std::vector<bf16> h_out(F_PROJ_OUTPUTS * FAN_IN);

	bf16 *d_x = nullptr;
	bf16 *d_d_f = nullptr;
	bf16 *d_backvec = nullptr;
	bf16 *d_out = nullptr;

	cudaMalloc(&d_x, h_x.size() * sizeof(bf16));
	cudaMalloc(&d_d_f, h_d_f.size() * sizeof(bf16));
	cudaMalloc(&d_backvec, h_backvec.size() * sizeof(bf16));
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));

	cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d_f, h_d_f.data(), h_d_f.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_backvec, h_backvec.data(), h_backvec.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(D_MODEL / Kernel::M_PER_BLOCK, F_PROJ_OUTPUTS / Kernel::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_x,
			SEQ_LEN,
			d_d_f,
			d_backvec,
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
			d_x,
			SEQ_LEN,
			d_d_f,
			d_backvec,
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
	double strict_flops = 2.0 * F_PROJ_OUTPUTS * FAN_IN * SEQ_LEN;
	double expanded_flops = 2.0 * F_PROJ_OUTPUTS * D_MODEL * SEQ_LEN;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("Strict TFLOPS (compact weights): %.2f\n", strict_flops / (median_ms * 1e-3) / 1e12);
	printf("Expanded TFLOPS (dense intermediate): %.2f\n", expanded_flops / (median_ms * 1e-3) / 1e12);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/ffn_d_f_weights.bin", h_out, F_PROJ_OUTPUTS, FAN_IN);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_x);
	cudaFree(d_d_f);
	cudaFree(d_backvec);
	cudaFree(d_out);
	return 0;
}
