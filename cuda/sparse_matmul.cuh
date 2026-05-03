#pragma once

#include "utils.cuh"

#pragma nv_diag_suppress 186

template<const usize _D_IN, const usize _D_OUT, const usize _FAN_IN, const usize _CYCLE>
struct SparseMatMul {
	static constexpr usize D_IN = _D_IN;
	static constexpr usize D_OUT = _D_OUT;
	static constexpr usize FAN_IN = _FAN_IN; // A_COLS
	static constexpr usize CYCLE = _CYCLE;
	static constexpr f64 SPARSE_SCALE = math::constexpr_sqrt(f64(D_IN) / f64(FAN_IN));

	static constexpr usize M = D_OUT; // A_ROWS
	static constexpr usize K = D_IN; // B_ROWS

	static constexpr usize M_WARPS = 2;
	static constexpr usize N_WARPS = 2;
	static constexpr usize M_PER_WARP = 32;
	static constexpr usize N_PER_WARP = 64;
	static constexpr usize M_PER_BLOCK = M_WARPS * M_PER_WARP;
	static constexpr usize N_PER_BLOCK = N_WARPS * N_PER_WARP;
	static constexpr usize WARPS_PER_BLOCK = M_WARPS * N_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
	static constexpr usize K_STEP = 64;
	static constexpr usize K_TILES = K_STEP / 16;
	static constexpr usize K_ITERS = K / K_STEP;
	static constexpr usize GMEM_PRELOAD = 2;
	static constexpr usize M_TILES = M_PER_WARP / 16;
	static constexpr usize N_TILES = N_PER_WARP / 16;

	static constexpr usize INPUT_STEP = K / CYCLE;
	static constexpr usize GROUP_TILE_CNT = CYCLE / 16;
	static constexpr usize GROUP_CNT = M_PER_WARP / CYCLE;

	static constexpr usize SMEM_BYTES =
		GMEM_PRELOAD * K_STEP * (M_PER_BLOCK + N_PER_BLOCK) * sizeof(bf16);

	static_assert(WARPS_PER_BLOCK == 4);
	static_assert(K_STEP % 16 == 0);
	static_assert(M_PER_WARP % CYCLE == 0);
	static_assert(CYCLE <= N_PER_WARP);
	static_assert(CYCLE % 16 == 0);
	static_assert(N_PER_WARP % CYCLE == 0);
	static_assert(K % CYCLE == 0);
	static_assert(K % K_STEP == 0);
	static_assert((INPUT_STEP * sizeof(bf16)) % 16 == 0);
	static_assert(FAN_IN <= K);
	static_assert(K % 16 == 0);

	static constexpr double flops(size_t seq_len) {
		return 2.0 * double(M) * double(seq_len) * double(K);
	}

	X17_DEVICE void cp_async_ab(
		GMatrix<bf16, M_PER_BLOCK, FAN_IN> gA_block,
		GMatrix<bf16, N_PER_BLOCK, K> gB_block,
		SMatrix<bf16, M_PER_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload,
		SMatrix<bf16, N_PER_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload,
		usize first_b_col,
		usize p,
		usize k_end
	) {
		if (p < k_end) {
			SMatrix<bf16, M_PER_BLOCK, K_STEP> sA_tile = tile_m<M_PER_BLOCK>(sA_preload, p % GMEM_PRELOAD);
			SMatrix<bf16, N_PER_BLOCK, K_STEP> sB_tile = tile_m<N_PER_BLOCK>(sB_preload, p % GMEM_PRELOAD);

			{
				auto src = gA_block.template slice_n<K_STEP>(p * K_STEP);
				auto dst = sA_tile;
				static constexpr usize GM = src.m_rows();
				static constexpr usize GN = src.n_cols();
				[[maybe_unused]] static constexpr usize DST_ROWS = dst.m_rows();
				static constexpr usize DST_COLS = dst.n_cols();
				static constexpr usize ROW_BYTES = dst.ROW_BYTES;
				using T = bf16;
				usize tid = threadIdx.x;
				usize dst_row = 0;
				usize dst_col = 0;

				constexpr usize SRC_ROW_BYTES = GN * sizeof(T);
				constexpr usize CP_BYTES = 16;
				constexpr usize CP_PER_ROW = SRC_ROW_BYTES / CP_BYTES;
				constexpr usize ROWS_PER_STEP = THREADS_PER_BLOCK / CP_PER_ROW;
				constexpr usize STEPS = GM / ROWS_PER_STEP;

				static_assert(CP_BYTES % sizeof(T) == 0);
				static_assert((GN * sizeof(T)) % CP_BYTES == 0);
				static_assert((DST_COLS * sizeof(T)) % CP_BYTES == 0);
				static_assert(THREADS_PER_BLOCK % CP_PER_ROW == 0);
				if constexpr (STEPS == 0) {
					if constexpr (GM % ROWS_PER_STEP == 0) {
						return;
					}
					if (tid >= (GM % ROWS_PER_STEP) * CP_PER_ROW) {
						return;
					}
				}

				// Thread's position within a step is fixed
				usize off_in_row = dst_col * sizeof(T) + (tid % CP_PER_ROW) * CP_BYTES;
				usize col_in_row = off_in_row / sizeof(T);
				usize row_in_step = tid / CP_PER_ROW;
				usize src_col = (tid % CP_PER_ROW) * CP_BYTES;

				constexpr usize REPEAT_AFTER = least_common_multiple(8, ROWS_PER_STEP) / ROWS_PER_STEP;
				usize off[REPEAT_AFTER];
				X17_UNROLL for (usize i = 0; i < REPEAT_AFTER; ++i) {
					usize row = dst_row + i * ROWS_PER_STEP + row_in_step;
					off[i] = off_in_row ^ ((row & 7) << 4);
				}

				usize first_col = row_in_step * INPUT_STEP;
				usize first_col_step = ROWS_PER_STEP * INPUT_STEP;

				u8 const * first_src_ptr =
					reinterpret_cast<u8 const *>(src._ptr)
					+ row_in_step * src.stride_bytes()
					+ src_col;
				usize src_step = ROWS_PER_STEP * src.stride_bytes();

				u32 first_dst_ptr = dst._ptr + (dst_row + row_in_step) * ROW_BYTES;
				u32 dst_step = ROWS_PER_STEP * ROW_BYTES;

				if constexpr (STEPS > 0) {
					X17_UNROLL for (usize step = 0; step < STEPS; ++step) {
						u8 const * src_ptr = first_src_ptr + step * src_step;
						u32 dst_ptr = first_dst_ptr + step * dst_step;
						u32 dst_ptr_swizzled = dst_ptr + off[step % REPEAT_AFTER];
						if (usize(p * K_STEP + col_in_row - first_col) < FAN_IN) {
							sm80::cp_async(src_ptr - first_col * sizeof(T), dst_ptr_swizzled);
						} else if (usize(p * K_STEP + col_in_row + K - first_col) < FAN_IN) {
							sm80::cp_async(src_ptr + (K - first_col) * sizeof(T), dst_ptr_swizzled);
						} else {
							store_shared_4x32b(dst_ptr_swizzled, 0.0f, 0.0f, 0.0f, 0.0f);
						}
						first_col = (first_col + first_col_step) % K;
					}
				}
				if constexpr (GM % ROWS_PER_STEP != 0) {
					usize step = STEPS;
					if (tid < (GM % ROWS_PER_STEP) * CP_PER_ROW) {
						u8 const *src_ptr = first_src_ptr + step * src_step;
						u32 const dst_ptr = first_dst_ptr + step * dst_step;
						u32 dst_ptr_swizzled = dst_ptr + off[step % REPEAT_AFTER];
						if (usize(p * K_STEP + col_in_row - first_col) < FAN_IN) {
							sm80::cp_async(src_ptr - first_col * sizeof(T), dst_ptr_swizzled);
						} else if (usize(p * K_STEP + col_in_row + K - first_col) < FAN_IN) {
							sm80::cp_async(src_ptr + (K - first_col) * sizeof(T), dst_ptr_swizzled);
						} else {
							store_shared_4x32b(dst_ptr_swizzled, 0.0f, 0.0f, 0.0f, 0.0f);
						}
					}
				}
			}
			cp_async_gmem_to_smem<THREADS_PER_BLOCK, N_PER_BLOCK, K_STEP>(
				threadIdx.x,
				gB_block.template slice_n<K_STEP>(first_b_col + p * K_STEP),
				sB_tile,
				0, 0, 0, 0
			);
		}
	}

	X17_DEVICE usize warp_m() const {
		usize tid = threadIdx.x;
		usize warp_idx = tid / WARP_SIZE;
		return (warp_idx / N_WARPS) * M_PER_WARP;
	}

	X17_DEVICE usize warp_n() const {
		usize tid = threadIdx.x;
		usize warp_idx = tid / WARP_SIZE;
		return (warp_idx % N_WARPS) * N_PER_WARP;
	}

	X17_DEVICE void run(
		bf16 *A,
		bf16 *B,
		Fragment_16x16<f32> (&acc_t)[N_TILES][M_TILES]
	) {
		usize tid = threadIdx.x;
		usize warp_idx = tid / WARP_SIZE;
		usize warp_m = (warp_idx / N_WARPS) * M_PER_WARP;
		usize warp_n = (warp_idx % N_WARPS) * N_PER_WARP;

		GMatrixDynSize<bf16, FAN_IN> gA{A, M};
		GMatrixDynSize<bf16, K> gB{B, usize(-1)};
		GMatrix<bf16, M_PER_BLOCK, FAN_IN> gA_block = tile_m<M_PER_BLOCK>(gA, blockIdx.x);
		GMatrix<bf16, N_PER_BLOCK, K> gB_block = tile_m<N_PER_BLOCK>(gB, blockIdx.y);
		usize first_b_col = 0;

		u32 smem = 0;
		SMatrix<bf16, M_PER_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload{smem};
		SMatrix<bf16, N_PER_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload{sA_preload._ptr + sA_preload.bytes()};

		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, first_b_col, p, K_ITERS);
			cp_async_commit();
		}

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

				cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, first_b_col, k_step + GMEM_PRELOAD, K_ITERS);
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
	}
};
