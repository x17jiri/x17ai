#include "utils2.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include "cutlass/util/GPU_Clock.hpp"

constexpr usize M_WARPS = 2;
constexpr usize N_WARPS = 2;
constexpr usize M_PER_WARP = 32;
constexpr usize N_PER_WARP = 64;
constexpr usize M_PER_BLOCK = M_WARPS * M_PER_WARP;
constexpr usize N_PER_BLOCK = N_WARPS * N_PER_WARP;
constexpr usize WARPS_PER_BLOCK = M_WARPS * N_WARPS;
constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr usize K_STEP = 64;
constexpr usize GMEM_PRELOAD = 4;
constexpr usize M_TILES = M_PER_WARP / 16;
constexpr usize N_TILES = N_PER_WARP / 16;

constexpr usize HEAD_DIM = 32;
constexpr usize ROPE_DIM = HEAD_DIM;
constexpr f32 ROPE_BASE = 10000.0f;
constexpr f32 ROPE_LOG_SCALE = -2.0 * math::fast::constexpr_logb(f64(ROPE_BASE)) / f64(ROPE_DIM);

static_assert(WARPS_PER_BLOCK == 4);
static_assert(K_STEP % 16 == 0);
static_assert(HEAD_DIM <= N_PER_WARP);
static_assert(HEAD_DIM % 16 == 0);
static_assert(N_PER_WARP % HEAD_DIM == 0);
static_assert(ROPE_DIM <= HEAD_DIM);
static_assert(ROPE_DIM % 16 == 0);

template<usize K>
X17_DEVICE void cp_async_ab(
	GMatrix<bf16, M_PER_BLOCK, K> gA_block,
	GMatrix<bf16, N_PER_BLOCK, K> gB_block,
	SMatrix<bf16, M_PER_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload,
	SMatrix<bf16, N_PER_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload,
	usize p,
	usize k_end
) {
	if (p < k_end) {
		SMatrix<bf16, M_PER_BLOCK, K_STEP> sA_tile = tile_m<M_PER_BLOCK>(sA_preload, p % GMEM_PRELOAD);
		SMatrix<bf16, N_PER_BLOCK, K_STEP> sB_tile = tile_m<N_PER_BLOCK>(sB_preload, p % GMEM_PRELOAD);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
			threadIdx.x,
			gA_block.template slice_n<K_STEP>(p * K_STEP),
			sA_tile
		);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
			threadIdx.x,
			gB_block.template slice_n<K_STEP>(p * K_STEP),
			sB_tile
		);
	}
}

X17_DEVICE void rope_rotate_pair(Fragment_8x8<f32> &frag, f32 c, f32 s) {
	f32 even = frag.first();
	f32 odd = frag.second();
	frag.set(
		math::fma(-odd, s, even * c),
		math::fma(even, s, odd * c)
	);
}

template<usize N_TILE_CNT>
X17_DEVICE void apply_rope(
	Fragment_16x16<f32> (&acc)[M_TILES][N_TILE_CNT],
	usize block_m,
	usize warp_m
) {
	constexpr usize GROUP_TILE_CNT = HEAD_DIM / 16;
	constexpr usize GROUP_CNT = N_PER_WARP / HEAD_DIM;
	constexpr usize ROPE_TILE_CNT = ROPE_DIM / 16;
	usize tid = threadIdx.x % WARP_SIZE;
	usize row_in_half = tid / 4;
	usize pair_in_quad = tid % 4;

	X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
		usize top_row = block_m + warp_m + mi * 16 + row_in_half;
		usize bot_row = top_row + 8;
		X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
			X17_UNROLL for (usize tile = 0; tile < ROPE_TILE_CNT; ++tile) {
				Fragment_16x16<f32> &frag = acc[mi][group * GROUP_TILE_CNT + tile];
				usize pair_base = tile * 8;

				f32 left_freq_recip = math::fast::expb(ROPE_LOG_SCALE * f32(pair_base + pair_in_quad));
				f32 right_freq_recip = math::fast::expb(ROPE_LOG_SCALE * f32(pair_base + 4 + pair_in_quad));

				f32 top_left_theta = f32(top_row) * left_freq_recip;
				f32 top_right_theta = f32(top_row) * right_freq_recip;
				f32 bot_left_theta = f32(bot_row) * left_freq_recip;
				f32 bot_right_theta = f32(bot_row) * right_freq_recip;

				f32 top_left_cos = math::fast::cos(top_left_theta);
				f32 top_left_sin = math::fast::sin(top_left_theta);
				f32 top_right_cos = math::fast::cos(top_right_theta);
				f32 top_right_sin = math::fast::sin(top_right_theta);

				f32 bot_left_cos = math::fast::cos(bot_left_theta);
				f32 bot_left_sin = math::fast::sin(bot_left_theta);
				f32 bot_right_cos = math::fast::cos(bot_right_theta);
				f32 bot_right_sin = math::fast::sin(bot_right_theta);

				rope_rotate_pair(frag.sub[0][0], top_left_cos, top_left_sin);
				rope_rotate_pair(frag.sub[0][1], top_right_cos, top_right_sin);
				rope_rotate_pair(frag.sub[1][0], bot_left_cos, bot_left_sin);
				rope_rotate_pair(frag.sub[1][1], bot_right_cos, bot_right_sin);
			}
		}
	}
}

template<usize N_TILE_CNT>
X17_DEVICE void l2_norm(
	Fragment_16x16<f32> (&acc)[M_TILES][N_TILE_CNT]
) {
	constexpr usize GROUP_TILE_CNT = HEAD_DIM / 16;
	constexpr usize GROUP_CNT = N_PER_WARP / HEAD_DIM;

	X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
		X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
			f32 top_sum_sq = 0.0f;
			f32 bot_sum_sq = 0.0f;

			X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
				Fragment_16x16<f32> &frag = acc[mi][group * GROUP_TILE_CNT + tile];
				top_sum_sq = math::fma(frag.sub[0][0].val0, frag.sub[0][0].val0, top_sum_sq);
				top_sum_sq = math::fma(frag.sub[0][0].val1, frag.sub[0][0].val1, top_sum_sq);
				top_sum_sq = math::fma(frag.sub[0][1].val0, frag.sub[0][1].val0, top_sum_sq);
				top_sum_sq = math::fma(frag.sub[0][1].val1, frag.sub[0][1].val1, top_sum_sq);

				bot_sum_sq = math::fma(frag.sub[1][0].val0, frag.sub[1][0].val0, bot_sum_sq);
				bot_sum_sq = math::fma(frag.sub[1][0].val1, frag.sub[1][0].val1, bot_sum_sq);
				bot_sum_sq = math::fma(frag.sub[1][1].val0, frag.sub[1][1].val0, bot_sum_sq);
				bot_sum_sq = math::fma(frag.sub[1][1].val1, frag.sub[1][1].val1, bot_sum_sq);
			}

			top_sum_sq += shuffle_xor_sync(top_sum_sq, 1);
			top_sum_sq += shuffle_xor_sync(top_sum_sq, 2);
			bot_sum_sq += shuffle_xor_sync(bot_sum_sq, 1);
			bot_sum_sq += shuffle_xor_sync(bot_sum_sq, 2);

			f32 top_inv_norm = top_sum_sq > 0.0f ? math::fast::recip(sqrtf(top_sum_sq)) : 0.0f;
			f32 bot_inv_norm = bot_sum_sq > 0.0f ? math::fast::recip(sqrtf(bot_sum_sq)) : 0.0f;

			X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
				acc[mi][group * GROUP_TILE_CNT + tile].scale_(top_inv_norm, bot_inv_norm);
			}
		}
	}
}

// GEMM: C[M,N] = A[M,K] * B^T[N,K]
// A is [M, K] row-major
// B is [N, K] row-major (transposed)
// C is [M, N] row-major
template<usize K>
__global__ void gemm_kernel(
	usize M, usize N,
	bf16 *A,
	bf16 *B,
	bf16 *C
) {
	static_assert(K % K_STEP == 0);
	constexpr usize K_ITERS = K / K_STEP;
	constexpr usize K_TILES = K_STEP / 16;

	usize tid = threadIdx.x;
	usize warp_idx = tid / WARP_SIZE;
	usize warp_m = (warp_idx / N_WARPS) * M_PER_WARP;
	usize warp_n = (warp_idx % N_WARPS) * N_PER_WARP;
	usize block_m = blockIdx.x * M_PER_BLOCK;

	GMatrixDynSize<bf16, K> gA{A, M};
	GMatrixDynSize<bf16, K> gB{B, N};
	GMatrix<bf16, M_PER_BLOCK, K> gA_block = tile_m<M_PER_BLOCK>(gA, blockIdx.x);
	GMatrix<bf16, N_PER_BLOCK, K> gB_block = tile_m<N_PER_BLOCK>(gB, blockIdx.y);

	u32 smem = 0;
	SMatrix<bf16, M_PER_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload{smem};
	SMatrix<bf16, N_PER_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload{sA_preload._ptr + sA_preload.bytes()};

	X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD - 1; ++p) {
		cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, p, K_ITERS);
		cp_async_commit();
	}

	Fragment_16x16<f32> acc[M_TILES][N_TILES];
	zero_(acc);

	SMatrix<bf16, M_PER_BLOCK, K_STEP> sA;
	SMatrix<bf16, N_PER_BLOCK, K_STEP> sB;

	Fragment_16x16<bf16> rA[K_TILES][M_TILES];
	Fragment_16x16<bf16> rB[K_TILES][N_TILES];

	X17_NO_UNROLL for (usize k_step = 0; k_step < K_ITERS; ++k_step) {
		{ // Get more data from GMEM
			cp_async_wait<GMEM_PRELOAD - 2>();
			sync_threads();
			sA = tile_m<M_PER_BLOCK>(sA_preload, k_step % GMEM_PRELOAD);
			sB = tile_m<N_PER_BLOCK>(sB_preload, k_step % GMEM_PRELOAD);

			X17_UNROLL for (usize k_tile = 0; k_tile < K_TILES; ++k_tile) {
				X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
					smem_tile_to_fragment(sA, warp_m + mi * 16, k_tile * 16, rA[k_tile][mi]);
				}
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					smem_tile_to_fragment(sB, warp_n + ni * 16, k_tile * 16, rB[k_tile][ni]);
				}
			}

			cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, k_step + GMEM_PRELOAD - 1, K_ITERS);
			cp_async_commit();
		}

		X17_UNROLL for (usize k_tile = 0; k_tile < K_TILES; ++k_tile) {
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					mma_a_bt(rA[k_tile][mi], rB[k_tile][ni], acc[mi][ni]);
				}
			}
		}
	}

	l2_norm(acc);
	apply_rope(acc, block_m, warp_m);

	bf16 *c_ptr = C + blockIdx.x * M_PER_BLOCK * N + blockIdx.y * N_PER_BLOCK;
	GMatrix<bf16, M_PER_BLOCK, N_PER_BLOCK> gC_block{c_ptr, N};
	X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
		store(acc[mi], gC_block, warp_m + mi * 16, warp_n);
	}
}

int main(int argc, char *argv[]) {
	constexpr usize K = 1024;
	{
		f64 ref_logb = log(f64(ROPE_BASE)) / log(math::fast::b);
		f64 diff = fabs(ref_logb - math::fast::constexpr_logb(f64(ROPE_BASE)));
		printf("logb=%e, constexpr_logb=%e, diff=%e\n",
			ref_logb, math::fast::constexpr_logb(f64(ROPE_BASE)), diff);
		if (diff > 1e-12) {
			return 1;
		}
	}
	bool use_real_data = argc <= 1;
	usize M = use_real_data ? 256 : 32768;
	usize N = use_real_data ? 256 : 4096;

	if (M % M_PER_BLOCK != 0 || N % N_PER_BLOCK != 0 || N % HEAD_DIM != 0) {
		printf("Expected M %% %u == 0, N %% %u == 0, and N %% %u == 0\n", M_PER_BLOCK, N_PER_BLOCK, HEAD_DIM);
		return 1;
	}

	{
		std::ofstream config_file("tmp/gemm.config.json", std::ios::binary);
		config_file << "{\n"
			<< "  \"K\": " << K << ",\n"
			<< "  \"HEAD_DIM\": " << HEAD_DIM << ",\n"
			<< "  \"ROPE_DIM\": " << ROPE_DIM << ",\n"
			<< "  \"ROPE_BASE\": " << ROPE_BASE << "\n"
			<< "}\n";
	}

	srand(42);
	std::vector<bf16> h_A(M * K), h_B(N * K), h_C(M * N);
	if (use_real_data) {
		std::ifstream a_in("tmp/a.bin", std::ios::binary);
		if (!a_in) {
			printf("Failed to open tmp/a.bin\n");
			return 1;
		}
		a_in.read(
			reinterpret_cast<char *>(h_A.data()),
			static_cast<std::streamsize>(h_A.size() * sizeof(bf16))
		);

		std::ifstream b_in("tmp/b.bin", std::ios::binary);
		if (!b_in) {
			printf("Failed to open tmp/b.bin\n");
			return 1;
		}
		b_in.read(
			reinterpret_cast<char *>(h_B.data()),
			static_cast<std::streamsize>(h_B.size() * sizeof(bf16))
		);
	} else {
		for (bf16 &x : h_A) x = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
		for (bf16 &x : h_B) x = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);

		std::ofstream a_out("tmp/large_a.bin", std::ios::binary);
		a_out.write(
			reinterpret_cast<char *>(h_A.data()),
			static_cast<std::streamsize>(h_A.size() * sizeof(bf16))
		);
		std::ofstream b_out("tmp/large_b.bin", std::ios::binary);
		b_out.write(
			reinterpret_cast<char *>(h_B.data()),
			static_cast<std::streamsize>(h_B.size() * sizeof(bf16))
		);
	}

	bf16 *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, M * K * sizeof(bf16));
	cudaMalloc(&d_B, N * K * sizeof(bf16));
	cudaMalloc(&d_C, M * N * sizeof(bf16));
	cudaMemcpy(d_A, h_A.data(), M * K * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), N * K * sizeof(bf16), cudaMemcpyHostToDevice);

	dim3 grid(M / M_PER_BLOCK, N / N_PER_BLOCK);
	usize smem_bytes = GMEM_PRELOAD * K_STEP * (M_PER_BLOCK + N_PER_BLOCK) * sizeof(bf16);

	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	int warmup = use_real_data ? 0 : 50;
	for (int i = 0; i < warmup; ++i) {
		gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(M, N, d_A, d_B, d_C);
	}
	cudaDeviceSynchronize();

	GPU_Clock timer;
	timer.start();
	int NUM_RUNS = use_real_data ? 1 : 100;
	for (int i = 0; i < NUM_RUNS; ++i) {
		gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(M, N, d_A, d_B, d_C);
	}
	cudaDeviceSynchronize();
	double elapsed = timer.seconds() / NUM_RUNS;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	double flops = 2.0 * M * N * K;
	double tflops = flops / elapsed / 1e12;
	printf("M=%u, N=%u, K=%u\n", M, N, K);
	printf("Average kernel time over %d runs: %.3f ms\n", NUM_RUNS, elapsed * 1e3);
	printf("%.2f TFLOPS\n", tflops);

	cudaMemcpy(h_C.data(), d_C, M * N * sizeof(bf16), cudaMemcpyDeviceToHost);

	const char *out_path = use_real_data ? "tmp/out_cpu.bin" : "tmp/large_out_cpu.bin";
	std::ofstream out_file(out_path, std::ios::binary);
	out_file.write(
		reinterpret_cast<char *>(h_C.data()),
		static_cast<std::streamsize>(h_C.size() * sizeof(bf16))
	);

	printf("\nFirst 4x4 (GPU):\n");
	for (usize m = 0; m < 4; m++) {
		for (usize n = 0; n < 4; n++)
			printf(" %10.4f", float(h_C[m * N + n]));
		printf("\n");
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
