#include "utils.cuh"

template<
	const usize A_ROWS,
	const usize A_COLS,
	const usize B_ROWS,
	const usize N_HEADS,
	const usize HEAD_DIM,
	const usize ROPE_DIM,
	const f64 L2_NORM_EPS,
	const f64 ROPE_BASE
>
struct QKVProj {
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
	static constexpr usize GMEM_PRELOAD = 2;
	static constexpr usize M_TILES = M_PER_WARP / 16;
	static constexpr usize N_TILES = N_PER_WARP / 16;
	static constexpr usize INPUT_STEP = B_ROWS / N_HEADS;
	static constexpr usize SMEM_BYTES = GMEM_PRELOAD * K_STEP * (M_PER_BLOCK + N_PER_BLOCK) * sizeof(bf16);

	static constexpr usize K_ITERS = B_ROWS / K_STEP;
	static constexpr usize PACKED_DIM = N_HEADS * HEAD_DIM;

	static_assert(WARPS_PER_BLOCK == 4);
	static_assert(K_STEP % 16 == 0);
	static_assert(M_PER_WARP % HEAD_DIM == 0);
	static_assert(HEAD_DIM <= N_PER_WARP);
	static_assert(HEAD_DIM % 16 == 0);
	static_assert(N_PER_WARP % HEAD_DIM == 0);
	static_assert(ROPE_DIM <= HEAD_DIM);
	static_assert(ROPE_DIM % 16 == 0);
	static_assert(B_ROWS % N_HEADS == 0);
	static_assert(B_ROWS % K_STEP == 0);
	static_assert((INPUT_STEP * sizeof(bf16)) % 16 == 0);
	static_assert(A_COLS <= B_ROWS);
	static_assert(A_COLS % K_STEP == 0);
	static_assert(A_ROWS == 3 * PACKED_DIM);
	static_assert(PACKED_DIM % M_PER_WARP == 0);
	static_assert(M_PER_BLOCK % N_HEADS == 0);
	static_assert((B_ROWS * sizeof(bf16)) % 16 == 0);

	X17_DEVICE void scale_pair(Fragment_8x8<f32> &frag, f32 first_scale, f32 second_scale) {
		frag.set(
			frag.first() * first_scale,
			frag.second() * second_scale
		);
	}

	X17_DEVICE bool is_q(usize block_m, usize warp_m) {
		usize warp_col = block_m + warp_m;
		return warp_col < PACKED_DIM;
	}

	X17_DEVICE bool is_k(usize block_m, usize warp_m) {
		usize warp_col = block_m + warp_m;
		return warp_col >= PACKED_DIM && warp_col < 2 * PACKED_DIM;
	}

	X17_DEVICE bool is_v(usize block_m, usize warp_m) {
		usize warp_col = block_m + warp_m;
		return warp_col >= 2 * PACKED_DIM;
	}

	template<usize N_TILE_CNT>
	X17_DEVICE void apply_norm_scales(
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES],
		bf16 const *gQKNormScale_ptr,
		usize block_m,
		usize warp_m
	) {
		usize pair_in_quad = threadIdx.x % 4;

		X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
			usize col_base = block_m + warp_m + mi * 16;
			usize left_col = col_base + pair_in_quad * 2;
			usize right_col = left_col + 8;

			union {
				u32 packed;
				bf16 values[2];
			} left_scale, right_scale;

			left_scale.packed = __float_as_uint(load_gmem_1x32b(reinterpret_cast<f32 const *>(gQKNormScale_ptr + left_col)));
			right_scale.packed = __float_as_uint(load_gmem_1x32b(reinterpret_cast<f32 const *>(gQKNormScale_ptr + right_col)));

			f32 left0 = f32(left_scale.values[0]);
			f32 left1 = f32(left_scale.values[1]);
			f32 right0 = f32(right_scale.values[0]);
			f32 right1 = f32(right_scale.values[1]);

			X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
				Fragment_16x16<f32> &frag = acc[ni][mi];
				scale_pair(frag.sub[0][0], left0, left1);
				scale_pair(frag.sub[1][0], left0, left1);
				scale_pair(frag.sub[0][1], right0, right1);
				scale_pair(frag.sub[1][1], right0, right1);
			}
		}
	}

	template<usize N_TILE_CNT>
	X17_DEVICE void scale_v_to_preserve_variance(
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES]
	) {
		constexpr f32 V_SCALE = math::constexpr_rsqrt(f64(A_COLS));
		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				acc[ni][mi].scale_(V_SCALE);
			}
		}
	}

	X17_DEVICE void cp_async_ab(
		GMatrix<bf16, M_PER_BLOCK, A_COLS> gA_block,
		GMatrix<bf16, N_PER_BLOCK, B_ROWS> gB_block,
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
				[[maybe_unused]] static constexpr usize M = dst.m_rows();
				static constexpr usize N = dst.n_cols();
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
				static_assert((N * sizeof(T)) % CP_BYTES == 0);
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
						if (usize(p * K_STEP + col_in_row - first_col) < A_COLS) {
							sm80::cp_async(src_ptr - first_col * sizeof(T), dst_ptr_swizzled);
						} else if (usize(p * K_STEP + col_in_row + B_ROWS - first_col) < A_COLS) {
							sm80::cp_async(src_ptr + (B_ROWS - first_col) * sizeof(T), dst_ptr_swizzled);
						} else {
							store_shared_4x32b(dst_ptr_swizzled, 0.0f, 0.0f, 0.0f, 0.0f);
						}
						first_col = (first_col + first_col_step) % B_ROWS;
					}
				}
				if constexpr (GM % ROWS_PER_STEP != 0) {
					usize step = STEPS;
					if (tid < (GM % ROWS_PER_STEP) * CP_PER_ROW) {
						u8 const *src_ptr = first_src_ptr + step * src_step;
						u32 const dst_ptr = first_dst_ptr + step * dst_step;
						u32 dst_ptr_swizzled = dst_ptr + off[step % REPEAT_AFTER];
						if (usize(p * K_STEP + col_in_row - first_col) < A_COLS) {
							sm80::cp_async(src_ptr - first_col * sizeof(T), dst_ptr_swizzled);
						} else if (usize(p * K_STEP + col_in_row + B_ROWS - first_col) < A_COLS) {
							sm80::cp_async(src_ptr + (B_ROWS - first_col) * sizeof(T), dst_ptr_swizzled);
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

	template<usize N_TILE_CNT>
	X17_DEVICE void apply_rope(
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES],
		usize block_n,
		usize warp_n
	) {
		constexpr usize GROUP_TILE_CNT = HEAD_DIM / 16;
		constexpr usize GROUP_CNT = M_PER_WARP / HEAD_DIM;
		constexpr usize ROPE_TILE_CNT = ROPE_DIM / 16;

		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			X17_UNROLL for (usize tile = 0; tile < ROPE_TILE_CNT; ++tile) {
				Fragment_16x16<f32> coefs;
				X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
					Fragment_16x16<f32> &frag = acc[ni][group * GROUP_TILE_CNT + tile];
					rope_coefs<ROPE_DIM, ROPE_BASE>(coefs, block_n + warp_n + ni * 16, tile * 16);
					apply_rope_(frag, coefs);
				}
			}
		}
	}

	template<usize N_TILE_CNT>
	X17_DEVICE void l2_norm(
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES]
	) {
		constexpr usize GROUP_TILE_CNT = HEAD_DIM / 16;
		constexpr usize GROUP_CNT = M_PER_WARP / HEAD_DIM;
		constexpr f32 EPS = f32(L2_NORM_EPS);

		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
				f32 top_sum_sq = 0.0f;
				f32 bot_sum_sq = 0.0f;

				X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
					Fragment_16x16<f32> &frag = acc[ni][group * GROUP_TILE_CNT + tile];
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

				f32 top_inv_norm = math::fast::recip(sqrtf(top_sum_sq) + EPS);
				f32 bot_inv_norm = math::fast::recip(sqrtf(bot_sum_sq) + EPS);

				X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
					acc[ni][group * GROUP_TILE_CNT + tile].scale_(top_inv_norm, bot_inv_norm);
				}
			}
		}
	}

	X17_DEVICE void run(bf16 *A, bf16 *B, bf16 const *gQKNormScale_ptr, bf16 *C) {
		static_assert(A_COLS % K_STEP == 0);
		static_assert(A_COLS <= B_ROWS);

		usize tid = threadIdx.x;
		usize warp_idx = tid / WARP_SIZE;
		usize block_m = blockIdx.x * M_PER_BLOCK;
		usize block_n = blockIdx.y * N_PER_BLOCK;
		usize warp_m = (warp_idx / N_WARPS) * M_PER_WARP;
		usize warp_n = (warp_idx % N_WARPS) * N_PER_WARP;

		GMatrixDynSize<bf16, A_COLS> gA{A, A_ROWS};
		GMatrixDynSize<bf16, B_ROWS> gB{B, /*B_COLS*/ 1000 /* TODO */};
		GMatrix<bf16, M_PER_BLOCK, A_COLS> gA_block = tile_m<M_PER_BLOCK>(gA, blockIdx.x);
		GMatrix<bf16, N_PER_BLOCK, B_ROWS> gB_block = tile_m<N_PER_BLOCK>(gB, blockIdx.y);
		usize first_b_col = 0;

		u32 smem = 0;
		SMatrix<bf16, M_PER_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload{smem};
		SMatrix<bf16, N_PER_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload{sA_preload._ptr + sA_preload.bytes()};

		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, first_b_col, p, K_ITERS);
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

		if (is_v(block_m, warp_m)) {
			scale_v_to_preserve_variance(acc_t);
		} else {
			l2_norm(acc_t);
			apply_norm_scales(acc_t, gQKNormScale_ptr, block_m, warp_m);
			if (is_k(block_m, warp_m)) {
				apply_rope(acc_t, block_n, warp_n);
			}
		}

		bf16 *c_ptr = C + blockIdx.y * N_PER_BLOCK * A_ROWS + blockIdx.x * M_PER_BLOCK;
		GMatrix<bf16, N_PER_BLOCK, M_PER_BLOCK> gC_block{c_ptr, A_ROWS};
		X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
			store(acc_t[ni], gC_block, warp_n + ni * 16, warp_m);
		}
	}
};

template<typename QKVProj>
__global__ __launch_bounds__(QKVProj::THREADS_PER_BLOCK) void qkv_proj(bf16 *A, bf16 *B, bf16 const *gQKNormScale_ptr, bf16 *C) {
	QKVProj qkv_proj = QKVProj();
	qkv_proj.run(A, B, gQKNormScale_ptr, C);
}
