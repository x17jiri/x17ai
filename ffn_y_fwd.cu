#include "cuda/gemm.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

#define USE_F8

using namespace config;

template<typename _ALoader, typename _BLoader, typename _Writer>
struct XGemm {
	using ALoader = _ALoader;
	using BLoader = _BLoader;
	using Writer = _Writer;

	static constexpr usize M_PER_BLOCK = ALoader::M;
	static constexpr usize N_PER_BLOCK = BLoader::N;
	static constexpr usize K_STEP = ALoader::N;
	static_assert(BLoader::M == K_STEP);

	static constexpr usize M_WARPS = 2;
	static constexpr usize N_WARPS = 2;
	static constexpr usize WARPS_PER_BLOCK = M_WARPS * N_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
	static constexpr usize M_PER_WARP = M_PER_BLOCK / M_WARPS;
	static constexpr usize N_PER_WARP = N_PER_BLOCK / N_WARPS;

	static constexpr usize M_TILES = M_PER_WARP / 16;
	static constexpr usize N_TILES = N_PER_WARP / 16;
	static constexpr usize K_TILES = K_STEP / 16;
	static_assert(M_TILES * M_WARPS * 16 == M_PER_BLOCK);
	static_assert(N_TILES * N_WARPS * 16 == N_PER_BLOCK);

	static constexpr usize SMEM_BYTES = ALoader::SMEM_BYTES + BLoader::SMEM_BYTES;
	static constexpr usize GMEM_PRELOAD = ALoader::GMEM_PRELOAD;
	static_assert(ALoader::GMEM_PRELOAD == BLoader::GMEM_PRELOAD);

	static_assert(M_TILES == 2);
	static_assert(N_TILES == 2);
	static_assert(K_TILES == 8);

	X17_DEVICE void run(
		ALoader &A,
		BLoader &B,
		Writer &C
	) {
		usize K_ITERS = std::min<usize>(A.n_cols(), B.m_rows()) / K_STEP;

		usize block_m = blockIdx.x;
		usize block_n = blockIdx.y;
		usize tid = threadIdx.x;
		usize warp_idx = tid / WARP_SIZE;
		usize warp_m = (warp_idx / N_WARPS);
		usize warp_n = (warp_idx % N_WARPS);

		SMemAllocator<SMEM_BYTES> smem_alloc;
		A.alloc_smem(smem_alloc);
		B.alloc_smem(smem_alloc);
		smem_alloc.finish();

		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			if (p < K_ITERS) {
				A.template cp_async<THREADS_PER_BLOCK>(p, block_m, p);
				B.template cp_async<THREADS_PER_BLOCK>(p, p, block_n);
			}
			cp_async_commit();
		}

		Fragment_16x16<bf16> rA_0_0, rA_0_1, rA_0_2, rA_0_3, rA_0_4, rA_0_5, rA_0_6, rA_0_7;
		Fragment_16x16<bf16> rA_1_0, rA_1_1, rA_1_2, rA_1_3, rA_1_4, rA_1_5, rA_1_6, rA_1_7;
		Fragment_16x16<bf16> rBT_0_0, rBT_0_1;
		Fragment_16x16<bf16> rBT_1_0, rBT_1_1;
		Fragment_16x16<bf16> rBT_2_0, rBT_2_1;
		Fragment_16x16<bf16> rBT_3_0, rBT_3_1;
		Fragment_16x16<bf16> rBT_4_0, rBT_4_1;
		Fragment_16x16<bf16> rBT_5_0, rBT_5_1;
		Fragment_16x16<bf16> rBT_6_0, rBT_6_1;
		Fragment_16x16<bf16> rBT_7_0, rBT_7_1;
		Fragment_16x16<f32> acc[N_TILES][M_TILES];
		zero_(acc);

		usize a_row = warp_m * M_TILES;
		usize b_col = warp_n * N_TILES;

		cp_async_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		A.load_fragment(0, a_row + 0, 0, rA_0_0);
		A.load_fragment(0, a_row + 1, 0, rA_1_0);
		B.load_fragment_trans(0, 0, b_col + 0, rBT_0_0);
		B.load_fragment_trans(0, 0, b_col + 1, rBT_0_1);

		A.load_fragment(0, a_row + 0, 1, rA_0_1);
		A.load_fragment(0, a_row + 1, 1, rA_1_1);
		B.load_fragment_trans(0, 1, b_col + 0, rBT_1_0);
		B.load_fragment_trans(0, 1, b_col + 1, rBT_1_1);

		A.load_fragment(0, a_row + 0, 2, rA_0_2);
		A.load_fragment(0, a_row + 1, 2, rA_1_2);
		B.load_fragment_trans(0, 2, b_col + 0, rBT_2_0);
		B.load_fragment_trans(0, 2, b_col + 1, rBT_2_1);

		A.load_fragment(0, a_row + 0, 3, rA_0_3);
		A.load_fragment(0, a_row + 1, 3, rA_1_3);
		B.load_fragment_trans(0, 3, b_col + 0, rBT_3_0);
		B.load_fragment_trans(0, 3, b_col + 1, rBT_3_1);

		A.load_fragment(0, a_row + 0, 4, rA_0_4);
		A.load_fragment(0, a_row + 1, 4, rA_1_4);
		B.load_fragment_trans(0, 4, b_col + 0, rBT_4_0);
		B.load_fragment_trans(0, 4, b_col + 1, rBT_4_1);

		A.load_fragment(0, a_row + 0, 5, rA_0_5);
		A.load_fragment(0, a_row + 1, 5, rA_1_5);
		B.load_fragment_trans(0, 5, b_col + 0, rBT_5_0);
		B.load_fragment_trans(0, 5, b_col + 1, rBT_5_1);

		A.load_fragment(0, a_row + 0, 6, rA_0_6);
		A.load_fragment(0, a_row + 1, 6, rA_1_6);
		B.load_fragment_trans(0, 6, b_col + 0, rBT_6_0);
		B.load_fragment_trans(0, 6, b_col + 1, rBT_6_1);

		A.load_fragment(0, a_row + 0, 7, rA_0_7);
		A.load_fragment(0, a_row + 1, 7, rA_1_7);
		B.load_fragment_trans(0, 7, b_col + 0, rBT_7_0);
		B.load_fragment_trans(0, 7, b_col + 1, rBT_7_1);

		X17_UNROLL for (usize k_step = 0; k_step < K_ITERS; ++k_step) {
			{ // Get more data from GMEM
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();

				usize p = k_step + GMEM_PRELOAD;
				if (p < K_ITERS) {
					A.template cp_async<THREADS_PER_BLOCK>(p, block_m, p);
					B.template cp_async<THREADS_PER_BLOCK>(p, p, block_n);
				}
				cp_async_commit();
			}

			mma_a_bt(rBT_0_0, rA_0_0, acc[0][0]);
			mma_a_bt(rBT_0_1, rA_0_0, acc[1][0]);
			mma_a_bt(rBT_0_0, rA_1_0, acc[0][1]);
			mma_a_bt(rBT_0_1, rA_1_0, acc[1][1]);
			A.load_fragment(k_step + 1, a_row + 0, 0, rA_0_0);
			A.load_fragment(k_step + 1, a_row + 1, 0, rA_1_0);
			B.load_fragment_trans(k_step + 1, 0, b_col + 0, rBT_0_0);
			B.load_fragment_trans(k_step + 1, 0, b_col + 1, rBT_0_1);

			mma_a_bt(rBT_1_0, rA_0_1, acc[0][0]);
			mma_a_bt(rBT_1_1, rA_0_1, acc[1][0]);
			mma_a_bt(rBT_1_0, rA_1_1, acc[0][1]);
			mma_a_bt(rBT_1_1, rA_1_1, acc[1][1]);
			A.load_fragment(k_step + 1, a_row + 0, 1, rA_0_1);
			A.load_fragment(k_step + 1, a_row + 1, 1, rA_1_1);
			B.load_fragment_trans(k_step + 1, 1, b_col + 0, rBT_1_0);
			B.load_fragment_trans(k_step + 1, 1, b_col + 1, rBT_1_1);

			mma_a_bt(rBT_2_0, rA_0_2, acc[0][0]);
			mma_a_bt(rBT_2_1, rA_0_2, acc[1][0]);
			mma_a_bt(rBT_2_0, rA_1_2, acc[0][1]);
			mma_a_bt(rBT_2_1, rA_1_2, acc[1][1]);
			A.load_fragment(k_step + 1, a_row + 0, 2, rA_0_2);
			A.load_fragment(k_step + 1, a_row + 1, 2, rA_1_2);
			B.load_fragment_trans(k_step + 1, 2, b_col + 0, rBT_2_0);
			B.load_fragment_trans(k_step + 1, 2, b_col + 1, rBT_2_1);

			mma_a_bt(rBT_3_0, rA_0_3, acc[0][0]);
			mma_a_bt(rBT_3_1, rA_0_3, acc[1][0]);
			mma_a_bt(rBT_3_0, rA_1_3, acc[0][1]);
			mma_a_bt(rBT_3_1, rA_1_3, acc[1][1]);
			A.load_fragment(k_step + 1, a_row + 0, 3, rA_0_3);
			A.load_fragment(k_step + 1, a_row + 1, 3, rA_1_3);
			B.load_fragment_trans(k_step + 1, 3, b_col + 0, rBT_3_0);
			B.load_fragment_trans(k_step + 1, 3, b_col + 1, rBT_3_1);

			mma_a_bt(rBT_4_0, rA_0_4, acc[0][0]);
			mma_a_bt(rBT_4_1, rA_0_4, acc[1][0]);
			mma_a_bt(rBT_4_0, rA_1_4, acc[0][1]);
			mma_a_bt(rBT_4_1, rA_1_4, acc[1][1]);
			A.load_fragment(k_step + 1, a_row + 0, 4, rA_0_4);
			A.load_fragment(k_step + 1, a_row + 1, 4, rA_1_4);
			B.load_fragment_trans(k_step + 1, 4, b_col + 0, rBT_4_0);
			B.load_fragment_trans(k_step + 1, 4, b_col + 1, rBT_4_1);

			mma_a_bt(rBT_5_0, rA_0_5, acc[0][0]);
			mma_a_bt(rBT_5_1, rA_0_5, acc[1][0]);
			mma_a_bt(rBT_5_0, rA_1_5, acc[0][1]);
			mma_a_bt(rBT_5_1, rA_1_5, acc[1][1]);
			A.load_fragment(k_step + 1, a_row + 0, 5, rA_0_5);
			A.load_fragment(k_step + 1, a_row + 1, 5, rA_1_5);
			B.load_fragment_trans(k_step + 1, 5, b_col + 0, rBT_5_0);
			B.load_fragment_trans(k_step + 1, 5, b_col + 1, rBT_5_1);

			mma_a_bt(rBT_6_0, rA_0_6, acc[0][0]);
			mma_a_bt(rBT_6_1, rA_0_6, acc[1][0]);
			mma_a_bt(rBT_6_0, rA_1_6, acc[0][1]);
			mma_a_bt(rBT_6_1, rA_1_6, acc[1][1]);
			A.load_fragment(k_step + 1, a_row + 0, 6, rA_0_6);
			A.load_fragment(k_step + 1, a_row + 1, 6, rA_1_6);
			B.load_fragment_trans(k_step + 1, 6, b_col + 0, rBT_6_0);
			B.load_fragment_trans(k_step + 1, 6, b_col + 1, rBT_6_1);

			mma_a_bt(rBT_7_0, rA_0_7, acc[0][0]);
			mma_a_bt(rBT_7_1, rA_0_7, acc[1][0]);
			mma_a_bt(rBT_7_0, rA_1_7, acc[0][1]);
			mma_a_bt(rBT_7_1, rA_1_7, acc[1][1]);
			A.load_fragment(k_step + 1, a_row + 0, 7, rA_0_7);
			A.load_fragment(k_step + 1, a_row + 1, 7, rA_1_7);
			B.load_fragment_trans(k_step + 1, 7, b_col + 0, rBT_7_0);
			B.load_fragment_trans(k_step + 1, 7, b_col + 1, rBT_7_1);
		}

		C.write(
			block_n * N_PER_BLOCK + warp_n * N_PER_WARP,
			block_m * M_PER_BLOCK + warp_m * M_PER_WARP,
			acc
		);
	}
};

namespace Ffn_y_fwd {
	using WeightLoader =
#ifdef USE_F8
		MatrixF8Loader<
			F_WIDTH,
			64, 128
		>;
#else
		MatrixLoader<
			F_WIDTH,
			128, 64
		>;
#endif

	using InputLoader =
#ifdef USE_F8
		MatrixTransLoader<
			MatrixLoader<
				F_WIDTH,
				64, 128
			>
		>;
#else
		MatrixTransLoader<
			MatrixLoader<
				F_WIDTH,
				64, 64
			>
		>;
#endif

	using Writer = MatrixWriter<D_MODEL>;

#ifdef USE_F8
	using Kernel = Gemm<WeightLoader, InputLoader, Writer>;
#else
	using Kernel = Gemm<WeightLoader, InputLoader, Writer>;
#endif

	X17_KERNEL(Kernel::THREADS_PER_BLOCK)
	void kernel(
		bf16 *w,
		bf16 *inp, usize n_inputs,
		bf16 *out
	) {
		auto a = WeightLoader(w, D_MODEL);
		auto b = InputLoader(inp, n_inputs);
		auto o = Writer(out);
		Kernel().run(a, b, o);
	}
}

using namespace Ffn_y_fwd;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, false, cli)) {
		return 1;
	}

	if (seq_len % Kernel::N_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", Kernel::N_PER_BLOCK);
		return 1;
	}
	if (D_MODEL % Kernel::M_PER_BLOCK != 0) {
		printf("Expected d_model %% %u == 0\n", Kernel::M_PER_BLOCK);
		return 1;
	}

#ifdef USE_F8
	std::vector<bf16> h_weights = load_f8_tensor(torch_tensor_path("ffn_y_weights_f8.bin"), D_MODEL, F_WIDTH);
#else
	std::vector<bf16> h_weights = load_tensor(torch_tensor_path("ffn_y_weights.bin"), D_MODEL, F_WIDTH);
#endif
	std::vector<bf16> h_f = load_tensor(tensor_path(cli.input_dir, "ffn_f.bin"), seq_len, F_WIDTH);
	if (h_weights.empty() || h_f.empty()) {
		return 1;
	}

	std::vector<bf16> h_out(seq_len * D_MODEL);

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

	dim3 grid(D_MODEL / Kernel::M_PER_BLOCK, seq_len / Kernel::N_PER_BLOCK);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, Kernel::THREADS_PER_BLOCK, Kernel::SMEM_BYTES>>>(
			d_weights,
			d_f,
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
			d_f,
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
	double tflops = 2.0 * D_MODEL * F_WIDTH * seq_len / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/ffn_y.bin", h_out, seq_len, D_MODEL);

	printf("Used SMEM per kernel: %u\n", Kernel::SMEM_BYTES);

	cudaFree(d_weights);
	cudaFree(d_f);
	cudaFree(d_out);
	return 0;
}
