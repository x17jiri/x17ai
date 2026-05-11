#include "cuda/gemm.cuh"

#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

constexpr usize SEQ_LEN = config::n_inputs;
constexpr usize D_MODEL = config::d_model;
constexpr usize F_WIDTH = config::f_width;
constexpr usize F_PROJ_OUTPUTS = 2 * F_WIDTH;

namespace Ffn_d_x {
	namespace detail {
		std::vector<bf16> expand_sparse_weights_transposed(std::vector<bf16> const &compact_weights) {
			std::vector<bf16> dense_weights(D_MODEL * F_PROJ_OUTPUTS);
			constexpr usize INPUT_STEP = D_MODEL / config::head_dim;

			for (usize row = 0; row < F_PROJ_OUTPUTS; ++row) {
				usize row_offset = (row * INPUT_STEP) % D_MODEL;
				for (usize col = 0; col < config::qkv_fan_in; ++col) {
					usize d_model_col = (row_offset + col) % D_MODEL;
					dense_weights[d_model_col * F_PROJ_OUTPUTS + row] = compact_weights[row * config::qkv_fan_in + col];
				}
			}

			return dense_weights;
		}
	}

	template<
		const usize _GN,
		const usize _M,
		const usize _N,
		const usize _GMEM_PRELOAD = 2
	>
	struct GeGluBackwardLoader {
		static constexpr usize GN = _GN;
		static constexpr usize M = _M;
		static constexpr usize N = _N;
		static constexpr usize DF_GN = GN / 2;
		static constexpr usize DF_TILE_N = N / 2;
		static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

		static_assert(GN % N == 0);
		static_assert(DF_GN % DF_TILE_N == 0);
		static_assert(M % 32 == 0);
		static_assert(N % 16 == 0);

		static constexpr usize BACKVEC_SMEM_BYTES = N * M * GMEM_PRELOAD * sizeof(bf16);
		static constexpr usize DF_SMEM_BYTES = M * DF_TILE_N * GMEM_PRELOAD * sizeof(bf16);
		static constexpr usize SMEM_BYTES = BACKVEC_SMEM_BYTES + DF_SMEM_BYTES;

		using GBackvec = GMatrixDynSize<bf16, GN>;
		using GDf = GMatrixDynSize<bf16, DF_GN>;
		using SBackvec = SMatrix<bf16, M * GMEM_PRELOAD, N>;
		using SDf = SMatrix<bf16, M, DF_TILE_N * GMEM_PRELOAD>;

		GBackvec gBackvec;
		GDf gDf;
		SBackvec sBackvec;
		SDf sDf;

		X17_DEVICE usize m_rows() const { return gBackvec.m_rows(); }
		X17_DEVICE usize n_cols() const { return gBackvec.n_cols(); }

		X17_DEVICE GeGluBackwardLoader(bf16 *d_f, bf16 *backvec, usize m_rows):
			gBackvec(backvec, m_rows),
			gDf(d_f, m_rows),
			sBackvec(),
			sDf()
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			sBackvec._ptr = smem_alloc.alloc(BACKVEC_SMEM_BYTES);
			sDf._ptr = smem_alloc.alloc(DF_SMEM_BYTES);
		}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize step, usize m, usize n) {
			cp_async_gmem_to_smem<THREADS_PER_BLOCK, M, N>(
				threadIdx.x,
				gBackvec.template tile_m<M>(m).template slice_n<N>(n * N),
				sBackvec.template tile_m<M>(step % GMEM_PRELOAD),
				0, 0, 0, 0
			);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK, M, DF_TILE_N>(
				threadIdx.x,
				gDf.template tile_m<M>(m).template slice_n<DF_TILE_N>(n * DF_TILE_N),
				sDf,
				0, 0, 0, (step % GMEM_PRELOAD) * DF_TILE_N
			);
		}

		X17_DEVICE void load_fused_fragment(
			usize step,
			usize row_tile,
			usize col_tile,
			Fragment_16x16<bf16> &frag
		) {
			auto backvec_tile = sBackvec.template tile_m<M>(step % GMEM_PRELOAD);
			smem_tile_to_fragment(backvec_tile, row_tile * 16, col_tile * 16, frag);

			Fragment_16x16<bf16> d_f_tile;
			usize slot_col = (step % GMEM_PRELOAD) * DF_TILE_N;
			usize d_f_tile_col = slot_col + (col_tile / 2) * 16;
			smem_tile_to_fragment(sDf, row_tile * 16, d_f_tile_col, d_f_tile);

			usize half = col_tile & 1;
			Fragment_8x8<bf16> d_f_top = d_f_tile.sub[0][half];
			Fragment_8x8<bf16> d_f_bot = d_f_tile.sub[1][half];
			geglu_backward_(d_f_top, frag.sub[0][0], frag.sub[0][1]);
			geglu_backward_(d_f_bot, frag.sub[1][0], frag.sub[1][1]);
		}

		X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
			load_fused_fragment(step, m, n, frag);
		}

		X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
			load_fused_fragment(step, m, n, frag);
			frag.transpose_();
		}
	};

	using WeightLoader = MatrixLoader<F_PROJ_OUTPUTS, 128, 64>;

	using InputLoader = MatrixTransLoader<GeGluBackwardLoader<F_PROJ_OUTPUTS, 64, 64>>;

	using Writer = MatrixWriter<config::d_model>;
	using Kernel = Gemm<WeightLoader, InputLoader, Writer>;

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *w,
		bf16 *d_f,
		bf16 *backvec, usize n_inputs,
		bf16 *d_x
	) {
		auto a = WeightLoader(w, config::d_model);
		auto b = InputLoader(d_f, backvec, n_inputs);
		auto o = Writer(d_x);
		Kernel().run(a, b, o);
	}
}

using namespace Ffn_d_x;

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
	if (F_PROJ_OUTPUTS % Kernel::K_STEP != 0) {
		printf("Expected 2 * f_width %% %u == 0\n", Kernel::K_STEP);
		return 1;
	}

	std::string d_f_path = tensor_path(cli.input_dir, "ffn_d_f.bin");
	std::string backvec_path = tensor_path(cli.input_dir, "ffn_f_backvec.bin");
	if (!std::filesystem::exists(backvec_path)) {
		backvec_path = tensor_path("tmp/block_cuda", "ffn_f_backvec.bin");
	}

	std::vector<bf16> h_compact_weights = load_tensor(torch_tensor_path("ffn_f_weights.bin"), F_PROJ_OUTPUTS, config::qkv_fan_in);
	std::vector<bf16> h_d_f = load_tensor(d_f_path, SEQ_LEN, F_WIDTH);
	std::vector<bf16> h_backvec = load_tensor(backvec_path, SEQ_LEN, F_PROJ_OUTPUTS);
	if (h_compact_weights.empty() || h_d_f.empty() || h_backvec.empty()) {
		if (h_backvec.empty()) {
			printf("Expected %s (or tmp/block_cuda/ffn_f_backvec.bin) before running ffn_d_x.cu\n", tensor_path(cli.input_dir, "ffn_f_backvec.bin").c_str());
		}
		return 1;
	}

	std::vector<bf16> h_weights = detail::expand_sparse_weights_transposed(h_compact_weights);
	std::vector<bf16> h_out(SEQ_LEN * D_MODEL);

	bf16 *d_weights = nullptr;
	bf16 *d_d_f = nullptr;
	bf16 *d_backvec = nullptr;
	bf16 *d_out = nullptr;

	cudaMalloc(&d_weights, h_weights.size() * sizeof(bf16));
	cudaMalloc(&d_d_f, h_d_f.size() * sizeof(bf16));
	cudaMalloc(&d_backvec, h_backvec.size() * sizeof(bf16));
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));

	cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d_f, h_d_f.data(), h_d_f.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_backvec, h_backvec.data(), h_backvec.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SMEM_BYTES);
	cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(D_MODEL / Kernel::M_PER_BLOCK, SEQ_LEN / Kernel::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_d_f,
			d_backvec,
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
			d_d_f,
			d_backvec,
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
	double tflops = 2.0 * D_MODEL * F_PROJ_OUTPUTS * SEQ_LEN / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/ffn_d_x.bin", h_out, SEQ_LEN, D_MODEL);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_d_f);
	cudaFree(d_backvec);
	cudaFree(d_out);
	return 0;
}
