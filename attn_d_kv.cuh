#pragma once

#include "utils2.cuh"

#pragma nv_diag_suppress 186

template<typename Attn_forward>
struct Attn_d_kv {
	static constexpr usize HEAD_CNT = Attn_forward::HEAD_CNT;
	static constexpr usize NONROPE_DIM = Attn_forward::NONROPE_DIM;
	static constexpr usize ROPE_DIM = Attn_forward::ROPE_DIM;
	static constexpr usize V_DIM = Attn_forward::V_DIM;
	static constexpr bool V_EQUALS_K = Attn_forward::V_EQUALS_K;
	static constexpr usize GMEM_PRELOAD = Attn_forward::GMEM_PRELOAD;

	static_assert(V_DIM <= NONROPE_DIM, "V_DIM must be <= NONROPE_DIM");

	static constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	static constexpr usize NONROPE_TILES = NONROPE_DIM / 16;
	static constexpr usize ROPE_TILES = ROPE_DIM / 16;
	static constexpr usize QK_TILES = NONROPE_TILES + ROPE_TILES;
	static constexpr usize V_TILES = V_DIM / 16;
	static constexpr usize PRELOAD_DIM = QK_DIM + V_DIM; // preload region holds both Q and dO
	static constexpr usize KV_SMEM_DIM = QK_DIM + (V_EQUALS_K ? 0 : V_DIM);
	static constexpr usize V_SMEM_COL = V_EQUALS_K ? 0 : QK_DIM;

	static constexpr usize Q_WARPS = 1;
	static constexpr usize KV_WARPS = 4;
	static constexpr usize Q_PER_WARP = 16;
	static constexpr usize Q_PER_STEP = Q_PER_WARP * Q_WARPS;
	static constexpr usize KV_PER_WARP = 16;
	static constexpr usize KV_PER_BLOCK = KV_PER_WARP * KV_WARPS;

	static constexpr usize WARPS_PER_BLOCK = Q_WARPS * KV_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

	// TODO - other matrices (dO, O, ...) should have their stride
	static constexpr usize KC_STRIDE = NONROPE_DIM * HEAD_CNT;
	static constexpr usize KR_STRIDE = ROPE_DIM * HEAD_CNT;
	static constexpr usize V_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize Q_STRIDE = QK_DIM * HEAD_CNT;

	static constexpr usize SMEM_BYTES =
		sizeof(bf16) * (
			Q_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
			+ KV_PER_BLOCK * KV_SMEM_DIM
		);

	static X17_DEVICE void cp_async_q_do(
		GMatrixDynSize<bf16, QK_DIM> gQ,
		GMatrixDynSize<bf16, V_DIM> gDO,
		SMatrix<bf16, Q_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
		usize p, usize q_steps
	) {
		if (p < q_steps) {
			auto slot = tile_m<Q_PER_STEP>(preload, p % GMEM_PRELOAD);
			// Load Q into columns [0, QK_DIM)
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<Q_PER_STEP>(gQ, p),
				slot, 0, 0
			);
			// Load dO into columns [QK_DIM, QK_DIM+V_DIM)
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<Q_PER_STEP>(gDO, p),
				slot, 0, QK_DIM
			);
		}
	}

	X17_DEVICE void run(
		usize seq_len, bf16 *gQ_ptr,
		bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
		bf16 *gDO_ptr, bf16 *gDK_ptr, bf16 *gDV_ptr,
		f32 *gL_ptr, f32 *gD_ptr,
		f32 *sink
	) {
		static_assert(Q_WARPS == 1, "current algorithm doesn't reduce over Q warps");

		// GMEM Matrices
		GMatrixDynSize<bf16, QK_DIM> gQ{gQ_ptr, seq_len, Q_STRIDE};
		GMatrixDynSize<bf16, NONROPE_DIM> gKc{gKc_ptr, seq_len, KC_STRIDE};
		GMatrixDynSize<bf16, ROPE_DIM> gKr{gKr_ptr, seq_len, KR_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gV{gV_ptr, seq_len, V_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gDO{gDO_ptr, seq_len};
		GMatrixDynSize<bf16, QK_DIM> gDK{gDK_ptr, seq_len};
		GMatrixDynSize<bf16, V_DIM> gDV{gDV_ptr, seq_len};

		// SMEM layout: Q + dO preload region + KV
		u32 smem = 0;
		usize kv_warp_idx = threadIdx.x / WARP_SIZE;
		usize tid = threadIdx.x % WARP_SIZE;
		SMatrix<bf16, Q_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		SMatrix<bf16, KV_PER_BLOCK, KV_SMEM_DIM> sKV{sPreload._ptr + sPreload.bytes()};

		// Load KV from GMEM to SMEM (no commit — piggyback on first KV commit)
		usize kv_block_idx = blockIdx.x;
		usize kv_block_start = kv_block_idx * KV_PER_BLOCK;
		usize kv_start = kv_block_start + kv_warp_idx * KV_PER_WARP;
		GMatrix<bf16, KV_PER_BLOCK, NONROPE_DIM> gKc_block = tile_m<KV_PER_BLOCK>(gKc, kv_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gKc_block, sKV, 0, 0);
		if constexpr (ROPE_DIM > 0) {
			GMatrix<bf16, KV_PER_BLOCK, ROPE_DIM> gKr_block = tile_m<KV_PER_BLOCK>(gKr, kv_block_idx);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gKr_block, sKV, 0, NONROPE_DIM);
		}
		if constexpr (!V_EQUALS_K) {
			GMatrix<bf16, KV_PER_BLOCK, V_DIM> gV_block = tile_m<KV_PER_BLOCK>(gV, kv_block_idx);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gV_block, sKV, 0, QK_DIM);
		}

		usize q_first_step = kv_block_start / Q_PER_STEP;
		usize q_steps = (seq_len + Q_PER_STEP - 1) / Q_PER_STEP;

		// Start preloading first Q+dO blocks (first commit also commits KV)
		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			cp_async_q_do(gQ, gDO, sPreload, q_first_step + p, q_steps);
			cp_async_commit();
		}

		// Sink: a virtual token with no V contribution - it only adds to the
		// softmax denominator, stealing probability from real tokens.
		// sink[0] = raw score, sink[1] = output gate
		f32 sink_score = -INFINITY;
		f32 gate = 1.0f;
		if (sink != nullptr) {
			load_gmem_2x32b(sink, sink_score, gate);
		}

		// dK, dV accumulators
		Fragment_16x16<f32> rDK[QK_TILES];
		zero_(rDK);
		Fragment_16x16<f32> rDV[V_TILES];
		zero_(rDV);

		cp_async_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		// Load K from SMEM into registers
		SMatrix<bf16, KV_PER_WARP, KV_SMEM_DIM> sKV_warp = tile_m<KV_PER_WARP>(sKV, kv_warp_idx);
		Fragment_16x16<bf16> rK[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sKV_warp, 0, i * 16, rK[i]);
		}
		// Load V from SMEM into registers
		// When V_EQUALS_K, V starts at col 0; otherwise at col QK_DIM
		Fragment_16x16<bf16> rV[V_TILES];
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			smem_tile_to_fragment(sKV_warp, 0, V_SMEM_COL + i * 16, rV[i]);
		}
		// Load first Q + dO tile from SMEM to registers
		auto sSlot = tile_m<Q_PER_STEP>(sPreload, q_first_step % GMEM_PRELOAD);
		Fragment_16x16<bf16> rQ[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sSlot, 0, i * 16, rQ[i]);
		}
		Fragment_16x16<bf16> rDO[V_TILES];
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			smem_tile_to_fragment(sSlot, 0, QK_DIM + i * 16, rDO[i]);
		}

		// Sequential loop over Q
		X17_NO_UNROLL for (usize q_step = q_first_step; q_step < q_steps; ++q_step) {
			// S = Q * K^T
			Fragment_16x16<f32> rS_f32;
			zero_(rS_f32);
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				mma_a_bt(rQ[i], rK[i], rS_f32);
			}

			// Per-Q-row score_scale: score_scale = logb(n) / sqrt(QK_DIM)
			// where n = q_pos + 2 (1 for 0-index, 1 for sink)
			// In the MMA fragment layout, tid/4 gives the row within the 8-row sub-tile
			usize q_start = q_step * Q_PER_STEP;
			f32 top_n = q_start + tid / 4 + 1 + 1;
			f32 bot_n = q_start + tid / 4 + 9 + 1;
			f32 top_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(top_n);
			f32 bot_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(bot_n);

			scale_top_(rS_f32, top_score_scale);
			scale_bottom_(rS_f32, bot_score_scale);

			// Causal mask: only needed for early Q steps
			if (q_step - q_first_step < KV_WARPS) {
				if (q_step - q_first_step == kv_warp_idx) {
					Attn_forward::causal_mask_diagonal(rS_f32);
				} else if (q_step - q_first_step > kv_warp_idx) {
					fill_(rS_f32, -INFINITY);
				}
			}

			/*
			TODO: Need to load L in the loop

			// P = expb(S - L)
			Fragment_16x16<f32> rP_f32;
			rP_f32.sub[0][0].val0 = math::fast::expb(rS_f32.sub[0][0].val0 - top_L);
			rP_f32.sub[0][0].val1 = math::fast::expb(rS_f32.sub[0][0].val1 - top_L);
			rP_f32.sub[0][1].val0 = math::fast::expb(rS_f32.sub[0][1].val0 - top_L);
			rP_f32.sub[0][1].val1 = math::fast::expb(rS_f32.sub[0][1].val1 - top_L);

			rP_f32.sub[1][0].val0 = math::fast::expb(rS_f32.sub[1][0].val0 - bot_L);
			rP_f32.sub[1][0].val1 = math::fast::expb(rS_f32.sub[1][0].val1 - bot_L);
			rP_f32.sub[1][1].val0 = math::fast::expb(rS_f32.sub[1][1].val0 - bot_L);
			rP_f32.sub[1][1].val1 = math::fast::expb(rS_f32.sub[1][1].val1 - bot_L);
			*/

			// dP = dO @ V^T (must compute before rDO is overwritten with next iteration)
			Fragment_16x16<f32> rDP;
			zero_(rDP);
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				mma_a_bt(rDO[i], rV[i], rDP);
			}

			{ // Get more data from GMEM
				// Wait for the next batch of GMEM -> SMEM preloads to complete
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sSlot = tile_m<Q_PER_STEP>(sPreload, (q_step + 1) % GMEM_PRELOAD);

				// Preload next Q+dO tiles from GMEM
				cp_async_q_do(gQ, gDO, sPreload, q_step + GMEM_PRELOAD, q_steps);
				cp_async_commit();
			}

			// Load next Q + dO from SMEM (for next iteration)
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sSlot, 0, i * 16, rQ[i]);
			}
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				smem_tile_to_fragment(sSlot, 0, QK_DIM + i * 16, rDO[i]);
			}

			// TODO: P, dS computation and dK/dV accumulation
			(void)rS_f32;
			(void)rDP;
			(void)rDK;
			(void)rDV;
			(void)gate;
			(void)sink_score;
		}

		// TODO: store dK, dV
	}
};

template<typename Attn_d_kv>
__global__ __launch_bounds__(Attn_d_kv::THREADS_PER_BLOCK) void
attn_d_kv(
	usize seq_len, bf16 *gQ_ptr,
	bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
	bf16 *gDO_ptr, bf16 *gDK_ptr, bf16 *gDV_ptr,
	f32 *gL_ptr, f32 *gD_ptr,
	f32 *sink
) {
	auto attn_d_kv = Attn_d_kv();
	attn_d_kv.run(seq_len, gQ_ptr, gKc_ptr, gKr_ptr, gV_ptr, gDO_ptr, gDK_ptr, gDV_ptr, gL_ptr, gD_ptr, sink);
}
