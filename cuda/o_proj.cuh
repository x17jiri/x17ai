#pragma once

#include "utils.cuh"

#pragma nv_diag_suppress 186

template<
	const usize _D_MODEL,
	const usize _O_PROJ_INPUT_ROWS
>
struct OProj {
	static constexpr usize D_MODEL = _D_MODEL;
	static constexpr usize O_PROJ_INPUT_ROWS = _O_PROJ_INPUT_ROWS;

	static constexpr usize A_ROWS = D_MODEL;
	static constexpr usize K = O_PROJ_INPUT_ROWS;

	static constexpr usize M_WARPS = 2;
	static constexpr usize N_WARPS = 2;
	static constexpr usize M_PER_WARP = 32;
	static constexpr usize N_PER_WARP = 64;
	static constexpr usize M_PER_BLOCK = M_WARPS * M_PER_WARP;
	static constexpr usize N_PER_BLOCK = N_WARPS * N_PER_WARP;
	static constexpr usize WARPS_PER_BLOCK = M_WARPS * N_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
	static constexpr usize K_STEP = 64;
	static constexpr usize GMEM_PRELOAD = 2;
	static constexpr usize M_TILES = M_PER_WARP / 16;
	static constexpr usize N_TILES = N_PER_WARP / 16;

	static constexpr usize SMEM_BYTES =
		GMEM_PRELOAD * K_STEP * (M_PER_BLOCK + N_PER_BLOCK) * sizeof(bf16);

	static_assert(WARPS_PER_BLOCK == 4, "current kernel layout expects 4 warps");
	static_assert(D_MODEL % M_PER_BLOCK == 0, "D_MODEL must be divisible by M_PER_BLOCK");
	static_assert(K_STEP % 16 == 0, "K_STEP must be divisible by 16");
	static_assert(K % K_STEP == 0, "O_PROJ_INPUT_ROWS must be divisible by K_STEP");
	static_assert(K % 16 == 0, "O_PROJ_INPUT_ROWS must be divisible by 16");
	static_assert(M_PER_WARP % 2 == 0, "M_PER_WARP must be even");
	static_assert((K * sizeof(bf16)) % 16 == 0, "O_PROJ_INPUT_ROWS rows must be 16B aligned");
	static_assert(N_TILES % 2 == 0, "N_TILES must be even");

	static constexpr double flops(size_t seq_len) {
		return 2.0 * double(A_ROWS) * double(seq_len) * double(K);
	}

	static X17_DEVICE void cp_async_ab(
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
			cp_async_gmem_to_smem<THREADS_PER_BLOCK, M_PER_BLOCK, K_STEP>(
				threadIdx.x,
				gA_block.template slice_n<K_STEP>(p * K_STEP),
				sA_tile,
				0, 0, 0, 0
			);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK, N_PER_BLOCK, K_STEP>(
				threadIdx.x,
				gB_block.template slice_n<K_STEP>(p * K_STEP),
				sB_tile,
				0, 0, 0, 0
			);
		}
	}

	X17_DEVICE void run(
		usize seq_len,
		bf16 *A,
		bf16 *B,
		bf16 *C
	) {
		constexpr usize K_ITERS = K / K_STEP;
		constexpr usize K_TILES = K_STEP / 16;

		usize tid = threadIdx.x;
		usize warp_idx = tid / WARP_SIZE;
		usize warp_m = (warp_idx / N_WARPS) * M_PER_WARP;
		usize warp_n = (warp_idx % N_WARPS) * N_PER_WARP;
		GMatrixDynSize<bf16, K> gA{A, A_ROWS};
		GMatrixDynSize<bf16, K> gB{B, seq_len};
		GMatrix<bf16, M_PER_BLOCK, K> gA_block = tile_m<M_PER_BLOCK>(gA, blockIdx.x);
		GMatrix<bf16, N_PER_BLOCK, K> gB_block = tile_m<N_PER_BLOCK>(gB, blockIdx.y);

		u32 smem = 0;
		SMatrix<bf16, M_PER_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload{smem};
		SMatrix<bf16, N_PER_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload{sA_preload._ptr + sA_preload.bytes()};

		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, p, K_ITERS);
			cp_async_commit();
		}

		Fragment_16x16<f32> acc_t[N_TILES][M_TILES];
		zero_(acc_t);

		Fragment_16x16<bf16> rA[K_TILES][M_TILES];
		Fragment_16x16<bf16> rB[K_TILES][N_TILES];

		SMatrix<bf16, M_PER_BLOCK, K_STEP> sA = tile_m<M_PER_BLOCK>(sA_preload, 0);
		SMatrix<bf16, N_PER_BLOCK, K_STEP> sB = tile_m<N_PER_BLOCK>(sB_preload, 0);

		cp_async_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		X17_UNROLL for (usize k_tile = 0; k_tile < K_TILES; ++k_tile) {
			X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
				smem_tile_to_fragment(sB, warp_n + ni * 16, k_tile * 16, rB[k_tile][ni]);
			}
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				smem_tile_to_fragment(sA, warp_m + mi * 16, k_tile * 16, rA[k_tile][mi]);
			}
		}

		X17_UNROLL for (usize k_step = 0; k_step < K_ITERS; ++k_step) {
			{ // Get more data from GMEM
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sA = tile_m<M_PER_BLOCK>(sA_preload, (k_step + 1) % GMEM_PRELOAD);
				sB = tile_m<N_PER_BLOCK>(sB_preload, (k_step + 1) % GMEM_PRELOAD);

				cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, k_step + GMEM_PRELOAD, K_ITERS);
				cp_async_commit();
			}

			X17_UNROLL for (usize k_tile = 0; k_tile < K_TILES; ++k_tile) {
				X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
					X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
						mma_a_bt(rB[k_tile][ni], rA[k_tile][mi], acc_t[ni][mi]);
					}
					smem_tile_to_fragment(sA, warp_m + mi * 16, k_tile * 16, rA[k_tile][mi]);
				}
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					smem_tile_to_fragment(sB, warp_n + ni * 16, k_tile * 16, rB[k_tile][ni]);
				}
			}
		}

		bf16 *c_ptr = C + blockIdx.y * N_PER_BLOCK * D_MODEL + blockIdx.x * M_PER_BLOCK;
		GMatrix<bf16, N_PER_BLOCK, M_PER_BLOCK> gC_block{c_ptr, D_MODEL};
		X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
			store(acc_t[ni], gC_block, warp_n + ni * 16, warp_m);
		}
	}
};

template<typename OProj>
__global__ __launch_bounds__(OProj::THREADS_PER_BLOCK) void
o_proj(
	usize seq_len,
	bf16 *gA_ptr,
	bf16 *gB_ptr,
	bf16 *gC_ptr
) {
	OProj o_proj_op = OProj();
	o_proj_op.run(seq_len, gA_ptr, gB_ptr, gC_ptr);
}
