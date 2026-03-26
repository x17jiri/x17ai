#pragma once

#include "utils2.cuh"

#pragma nv_diag_suppress 186

template<typename Attn_forward>
struct Attn_d_kv {
	static constexpr usize HEAD_CNT = Attn_forward::HEAD_CNT;
	static constexpr usize NONROPE_DIM = Attn_forward::NONROPE_DIM;
	static constexpr usize ROPE_DIM = Attn_forward::ROPE_DIM;
	static constexpr usize V_DIM = Attn_forward::V_DIM;
	static constexpr usize Q_WARPS = 1;
	static constexpr usize KV_WARPS = 4;
	static constexpr bool V_EQUALS_K = Attn_forward::V_EQUALS_K;
	static constexpr usize GMEM_PRELOAD = Attn_forward::GMEM_PRELOAD;

	static_assert(V_DIM <= NONROPE_DIM, "V_DIM must be <= NONROPE_DIM");

	static constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	static constexpr usize NONROPE_TILES = NONROPE_DIM / 16;
	static constexpr usize ROPE_TILES = ROPE_DIM / 16;
	static constexpr usize QK_TILES = NONROPE_TILES + ROPE_TILES;
	static constexpr usize V_TILES = V_DIM / 16;
	static constexpr usize PRELOAD_DIM = QK_DIM + V_DIM; // preload region holds both Q and dO

	static constexpr usize Q_PER_WARP = 16;
	static constexpr usize Q_PER_STEP = Q_PER_WARP * Q_WARPS;
	static constexpr usize KV_PER_WARP = 16;
	static constexpr usize KV_PER_BLOCK = KV_PER_WARP * KV_WARPS;

	static constexpr usize WARPS_PER_BLOCK = Q_WARPS * KV_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
	static constexpr bool SMEM_OVERLAP_K_WITH_Q = KV_PER_BLOCK <= Q_PER_STEP;
	static constexpr usize EARLY_PRELOAD = SMEM_OVERLAP_K_WITH_Q ? GMEM_PRELOAD - 1 : GMEM_PRELOAD;

	// TODO - other matrices (dO, O, ...) should have their stride
	static constexpr usize KC_STRIDE = NONROPE_DIM * HEAD_CNT;
	static constexpr usize KR_STRIDE = ROPE_DIM * HEAD_CNT;
	static constexpr usize V_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize Q_STRIDE = QK_DIM * HEAD_CNT;

	static constexpr usize SMEM_BYTES =
		sizeof(bf16) * (
			(SMEM_OVERLAP_K_WITH_Q
				?
					Q_PER_STEP * PRELOAD_DIM * (GMEM_PRELOAD - 1)
					+ std::max(
						Q_PER_STEP * PRELOAD_DIM,
						KV_PER_BLOCK * QK_DIM
					)
				:
					Q_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
					+ KV_PER_BLOCK * QK_DIM
			) + (V_EQUALS_K ? 0 : KV_PER_BLOCK * V_DIM)
		);

	X17_DEVICE void causal_mask_diagonal(Fragment_16x16<f32> &rScores) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize q = tid / 4;           // 0..7
		usize k = 2 * (tid % 4);    // 0,2,4,6
		constexpr f32 NEG_INF = -INFINITY;

		rScores.sub[0][1].val0 = NEG_INF;
		rScores.sub[0][1].val1 = NEG_INF;

		rScores.sub[0][0].val0 = k <= q ? rScores.sub[0][0].val0 : NEG_INF;
		rScores.sub[1][1].val0 = k <= q ? rScores.sub[1][1].val0 : NEG_INF;

		rScores.sub[0][0].val1 = k + 1 <= q ? rScores.sub[0][0].val1 : NEG_INF;
		rScores.sub[1][1].val1 = k + 1 <= q ? rScores.sub[1][1].val1 : NEG_INF;
	}

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
		// GMEM Matrices
		GMatrixDynSize<bf16, QK_DIM> gQ{gQ_ptr, seq_len, Q_STRIDE};
		GMatrixDynSize<bf16, NONROPE_DIM> gKc{gKc_ptr, seq_len, KC_STRIDE};
		GMatrixDynSize<bf16, ROPE_DIM> gKr{gKr_ptr, seq_len, KR_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gV{gV_ptr, seq_len, V_STRIDE};
		GMatrixDynSize<bf16, QK_DIM> gDK{gDK_ptr, seq_len};
		GMatrixDynSize<bf16, V_DIM> gDV{gDV_ptr, seq_len};
		GMatrixDynSize<bf16, V_DIM> gDO{gDO_ptr, seq_len};

		// SMEM layout: Q + dO preload region; last tile may overlap with K + V preload
		u32 smem = 0;
		usize q_warp_idx = (threadIdx.x / WARP_SIZE) / KV_WARPS;
		usize kv_warp_idx = (threadIdx.x / WARP_SIZE) % KV_WARPS;
		usize tid = threadIdx.x % WARP_SIZE;
		SMatrix<bf16, Q_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		// sK: may overlap with last Q+dO preload tile
		SMatrix<bf16, KV_PER_BLOCK, QK_DIM> sK{
			SMEM_OVERLAP_K_WITH_Q
				? tile_m<Q_PER_STEP>(sPreload, GMEM_PRELOAD - 1)._ptr
				: sPreload._ptr + sPreload.bytes()
		};
		// sV: always after preload (never overlapped), only when V is separate from K
		SMatrix<bf16, KV_PER_BLOCK, V_DIM> sV{
			SMEM_OVERLAP_K_WITH_Q
				? sPreload._ptr + sPreload.bytes()
				: sK._ptr + sK.bytes()
		};

		// Load KV from GMEM to SMEM (no commit — piggyback on first KV commit)
		usize kv_block_idx = blockIdx.x;
		usize kv_block_start = kv_block_idx * KV_PER_BLOCK;
		usize kv_start = kv_block_start + kv_warp_idx * KV_PER_WARP;
		GMatrix<bf16, KV_PER_BLOCK, NONROPE_DIM> gKc_block = tile_m<KV_PER_BLOCK>(gKc, kv_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gKc_block, sK, 0, 0);
		if constexpr (ROPE_DIM > 0) {
			GMatrix<bf16, KV_PER_BLOCK, ROPE_DIM> gKr_block = tile_m<KV_PER_BLOCK>(gKr, kv_block_idx);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gKr_block, sK, 0, NONROPE_DIM);
		}
		if constexpr (!V_EQUALS_K) {
			GMatrix<bf16, KV_PER_BLOCK, V_DIM> gV_block = tile_m<KV_PER_BLOCK>(gV, kv_block_idx);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gV_block, sV, 0, 0);
		}

		usize q_skip = kv_block_start / Q_PER_STEP;
		usize q_steps = (seq_len + Q_PER_STEP - 1) / Q_PER_STEP;

		// Start preloading first Q+dO blocks (first commit also commits KV)
		// When KV overlaps Q+dO SMEM, don't use the last preload tile yet (it holds KV)
		X17_UNROLL for (usize p = 0; p < EARLY_PRELOAD; ++p) {
			cp_async_q_do(gQ, gDO, sPreload, q_skip + p, q_steps);
			cp_async_commit();
		}

		// Sink and gate
		f32 sink_score = -INFINITY;
		f32 gate = 1.0f;
		if (sink != nullptr) {
			load_gmem_2x32b(sink, sink_score, gate);
		}

		// dK, dV accumulators (per-warp, will reduce across warps later)
		Fragment_16x16<f32> rDK[QK_TILES];
		zero_(rDK);
		Fragment_16x16<f32> rDV[V_TILES];
		zero_(rDV);

		cp_async_wait<EARLY_PRELOAD - 1>();
		sync_threads();

		//=== refactored up to here ===

		// Load K from sK into registers (each kv_warp loads its own 16-row tile)
		// rKV holds K; when V_EQUALS_K, rKV[0..V_TILES-1] is also V
		// When !V_EQUALS_K, load V from sV (persistent) into rKV when needed
		SMatrix<bf16, KV_PER_WARP, QK_DIM> sK_warp = tile_m<KV_PER_WARP>(sK, kv_warp_idx);
		Fragment_16x16<bf16> rKV[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sK_warp, 0, i * 16, rKV[i]);
		}
		// Now that K is in registers, reuse its SMEM for Q + dO preload
		if constexpr (SMEM_OVERLAP_K_WITH_Q) {
			cp_async_q_do(gQ, gDO, sPreload, q_skip + GMEM_PRELOAD - 1, q_steps);
			cp_async_commit();
		}
		// Load first Q + dO tile from SMEM to registers
		auto sSlot = tile_m<Q_PER_STEP>(sPreload, first_q_block % GMEM_PRELOAD);
		Fragment_16x16<bf16> rQ[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sSlot, q_warp_idx * Q_PER_WARP, i * 16, rQ[i]);
		}

		// Sequential loop over Q
		X17_NO_UNROLL for (usize q_step = q_skip; q_step < q_steps; ++q_step) {
			usize q_block = first_q_block + q_step;
			usize q_block_start = q_block * Q_PER_STEP;
			usize my_q_start = q_block_start + q_warp_idx * Q_PER_WARP;

			// S = Q * K^T (16 Q rows × 16 KV rows)
			Fragment_16x16<f32> rS;
			zero_(rS);
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				mma_a_bt(rQ[i], rKV[i], rS);
			}

			// Per-Q-row score_scale: score_scale = logb(n) / sqrt(QK_DIM)
			// where n = q_pos + 2 (1 for 0-index, 1 for sink)
			// In the MMA fragment layout, tid/4 gives the row within the 8-row sub-tile
			f32 top_n = my_q_start + tid / 4 + 1 + 1;
			f32 bot_n = my_q_start + tid / 4 + 9 + 1;
			f32 top_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(top_n);
			f32 bot_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(bot_n);

			scale_top_(rS, top_score_scale);
			scale_bottom_(rS, bot_score_scale);

			// Causal mask: only needed for early Q steps (q_step < partial_q_steps).
			// After that, all Q rows > all KV rows — no masking needed.
			if (q_step < partial_q_steps) {
				isize kv_pos = isize(kv_block_start + kv_warp_idx * KV_PER_WARP) - isize(kv_extra);
				isize q_pos = isize(my_q_start);
				if (kv_pos == q_pos) {
					causal_mask_diagonal(rS);
				} else if (kv_pos > q_pos) {
					fill_(rS, -INFINITY);
				}
			}

			{ // Get more data from GMEM
				// Wait for the next Q+dO preload to complete
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sSlot = tile_m<Q_PER_STEP>(sPreload, (q_block + 1) % GMEM_PRELOAD);

				// Preload next Q+dO tiles from GMEM
				cp_async_q_do(gQ, gDO, sPreload, first_q_block + q_step + GMEM_PRELOAD, first_q_block + q_steps);
				cp_async_commit();
			}

			// Load next Q from SMEM (for next iteration)
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sSlot, q_warp_idx * Q_PER_WARP, i * 16, rQ[i]);
			}

			// TODO: P, dP, dS computation and dK/dV accumulation (steps 3-7)
			(void)rS;
			(void)rDK;
			(void)rDV;
			(void)gate;
			(void)sink_score;
		}

		// TODO: reduce across warps and store dK, dV
	}
};

template<typename Attn_d_kv>
__global__ __launch_bounds__(Attn_d_kv::THREADS_PER_BLOCK) void
attn_d_kv(
	usize q_cnt, bf16 *gQ_ptr,
	usize kv_cnt, bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
	bf16 *gDO_ptr, bf16 *gDK_ptr, bf16 *gDV_ptr,
	f32 *gL_ptr, f32 *gD_ptr,
	f32 *sink
) {
	auto attn_d_kv = Attn_d_kv();
	attn_d_kv.run(q_cnt, gQ_ptr, kv_cnt, gKc_ptr, gKr_ptr, gV_ptr, gDO_ptr, gDK_ptr, gDV_ptr, gL_ptr, gD_ptr, sink);
}
