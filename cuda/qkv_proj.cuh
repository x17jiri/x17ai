#include "utils.cuh"

template<
	const usize A_ROWS,
	const usize A_COLS,
	const usize B_ROWS,
	const usize N_HEADS,
	const usize HEAD_DIM,
	const usize ROPE_DIM,
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
	static constexpr usize GMEM_PRELOAD = 2;
	static constexpr usize M_TILES = M_PER_WARP / 16;
	static constexpr usize N_TILES = N_PER_WARP / 16;
	static constexpr usize INPUT_STEP = B_ROWS / N_HEADS;
	static constexpr usize SMEM_BYTES = GMEM_PRELOAD * K_STEP * (M_PER_BLOCK + N_PER_BLOCK) * sizeof(bf16);

	static constexpr f32 ROPE_LOG_SCALE = -2.0 * math::fast::constexpr_logb(ROPE_BASE) / f64(ROPE_DIM);

	static_assert(WARPS_PER_BLOCK == 4);
	static_assert(K_STEP % 16 == 0);
	static_assert(HEAD_DIM <= N_PER_WARP);
	static_assert(HEAD_DIM % 16 == 0);
	static_assert(N_PER_WARP % HEAD_DIM == 0);
	static_assert(ROPE_DIM <= HEAD_DIM);
	static_assert(ROPE_DIM % 16 == 0);
	static_assert(B_ROWS % N_HEADS == 0);
	static_assert((INPUT_STEP * sizeof(bf16)) % 16 == 0);
	static_assert(A_COLS <= B_ROWS);
	static_assert(A_COLS % K_STEP == 0);
	static_assert(M_PER_BLOCK % N_HEADS == 0);
	static_assert((B_ROWS * sizeof(bf16)) % 16 == 0);

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
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				gA_block.template slice_n<K_STEP>(p * K_STEP),
				sA_tile
			);
			cp_async_gmem_to_smem_modulo<THREADS_PER_BLOCK>(
				threadIdx.x,
				gB_block,
				sB_tile,
				0,
				first_b_col + p * K_STEP,
				0
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
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES],
		usize block_n,
		usize warp_n
	) {
		constexpr usize GROUP_TILE_CNT = HEAD_DIM / 16;
		constexpr usize GROUP_CNT = M_PER_WARP / HEAD_DIM;
		constexpr usize ROPE_TILE_CNT = ROPE_DIM / 16;
		usize tid = threadIdx.x % WARP_SIZE;
		usize row_in_half = tid / 4;
		usize pair_in_quad = tid % 4;

		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			usize top_row = block_n + warp_n + ni * 16 + row_in_half;
			usize bot_row = top_row + 8;
			X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
				X17_UNROLL for (usize tile = 0; tile < ROPE_TILE_CNT; ++tile) {
					Fragment_16x16<f32> &frag = acc[ni][group * GROUP_TILE_CNT + tile];
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
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES]
	) {
		constexpr usize GROUP_TILE_CNT = HEAD_DIM / 16;
		constexpr usize GROUP_CNT = M_PER_WARP / HEAD_DIM;

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

				f32 top_inv_norm = top_sum_sq > 0.0f ? math::fast::recip(sqrtf(top_sum_sq)) : 0.0f;
				f32 bot_inv_norm = bot_sum_sq > 0.0f ? math::fast::recip(sqrtf(bot_sum_sq)) : 0.0f;

				X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
					acc[ni][group * GROUP_TILE_CNT + tile].scale_(top_inv_norm, bot_inv_norm);
				}
			}
		}
	}

	// GEMM: C[A_ROWS,B_COLS] = A[A_ROWS,A_COLS] * windowed(B[B_ROWS,B_COLS])
	// A is the weight matrix [A_ROWS, A_COLS] row-major.
	// B is the input matrix stored transposed as [B_COLS, B_ROWS] row-major so B_ROWS is contiguous.
	// Each output column reads A_COLS values from the corresponding B row, starting at
	// (output_col * INPUT_STEP) modulo B_ROWS.
	X17_DEVICE void run(bf16 *A, bf16 *B, bf16 *C) {
		static_assert(A_COLS % K_STEP == 0);
		static_assert(A_COLS <= B_ROWS);
		constexpr usize K_ITERS = (A_COLS + (M_PER_BLOCK - 1)*INPUT_STEP + K_STEP - 1) / K_STEP;
		constexpr usize K_TILES = K_STEP / 16;

		usize tid = threadIdx.x;
		usize warp_idx = tid / WARP_SIZE;
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

		//l2_norm(acc_t);
		//apply_rope(acc_t, block_n, warp_n);

		bf16 *c_ptr = C + blockIdx.y * N_PER_BLOCK * A_ROWS + blockIdx.x * M_PER_BLOCK;
		GMatrix<bf16, N_PER_BLOCK, M_PER_BLOCK> gC_block{c_ptr, A_ROWS};
		X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
			store(acc_t[ni], gC_block, warp_n + ni * 16, warp_m);
		}
	}
};

template<typename QKVProj>
__global__ __launch_bounds__(QKVProj::THREADS_PER_BLOCK) void qkv_proj(bf16 *A, bf16 *B, bf16 *C) {
	QKVProj qkv_proj = QKVProj();
	qkv_proj.run(A, B, C);
}
