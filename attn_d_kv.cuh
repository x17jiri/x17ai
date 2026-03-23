#pragma once

#include "utils2.cuh"

#pragma nv_diag_suppress 186

template<
	const usize HEAD_CNT,
	const usize NONROPE_DIM,
	const usize ROPE_DIM,
	const usize V_DIM,
	const usize Q_WARPS,
	const usize KV_WARPS,
	const bool V_EQUALS_K = false,
	const usize GMEM_PRELOAD = 2
>
struct Attn_d_kv {
	static_assert(V_DIM <= NONROPE_DIM, "V_DIM must be <= NONROPE_DIM");

	static constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	static constexpr usize NONROPE_TILES = NONROPE_DIM / 16;
	static constexpr usize ROPE_TILES = ROPE_DIM / 16;
	static constexpr usize QK_TILES = NONROPE_TILES + ROPE_TILES;
	static constexpr usize V_TILES = V_DIM / 16;
	static constexpr usize PRELOAD_DIM = V_EQUALS_K ? QK_DIM : QK_DIM + V_DIM;
	static constexpr usize V_SMEM_COL = V_EQUALS_K ? 0 : QK_DIM;

	static constexpr usize Q_PER_WARP = 16;
	static constexpr usize Q_PER_BLOCK = Q_PER_WARP * Q_WARPS;
	static constexpr usize KV_PER_WARP = 16;
	static constexpr usize KV_PER_STEP = KV_PER_WARP * KV_WARPS;

	static constexpr usize WARPS_PER_BLOCK = Q_WARPS * KV_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
	static constexpr bool SMEM_OVERLAP_Q_WITH_KV = Q_PER_BLOCK <= KV_PER_STEP;
	static constexpr usize EARLY_PRELOAD = SMEM_OVERLAP_Q_WITH_KV ? GMEM_PRELOAD - 1 : GMEM_PRELOAD;

	static constexpr usize KC_STRIDE = NONROPE_DIM * HEAD_CNT;
	static constexpr usize KR_STRIDE = ROPE_DIM * HEAD_CNT;
	static constexpr usize V_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize Q_STRIDE = QK_DIM * HEAD_CNT;

	// =========================================================================
	// dKV backward kernel
	//
	// Each block owns a KV tile of KV_PER_STEP rows and loops over Q blocks.
	// Accumulates dK [KV_PER_STEP, QK_DIM] and dV [KV_PER_STEP, V_DIM].
	//
	// SMEM layout: Q+dO preload region (double-buffered Q_PER_BLOCK rows)
	// K is loaded into SMEM once at startup, moved to registers, then SMEM
	// is reused for the Q+dO streaming preload.
	//
	// Q_WARPS warps each process a Q_PER_WARP=16 row Q tile per step.
	// All warps share the same KV tile (K in registers).
	// After each warp computes its 16x16 dS/P tiles, they write partial
	// dK/dV contributions to SMEM for reduction across warps, or — since
	// KV_WARPS==1 and each of the 4 Q_WARPS produces a separate dK/dV
	// contribution — we accumulate in f32 registers and reduce at the end.
	//
	// Actually: each warp computes dK_w += dS_w^T @ Q_w, dV_w += P_w^T @ dO_w
	// where dS_w and P_w are 16x16 (16 Q rows × 16 KV rows).
	// The 16 KV rows are the SAME across all warps, but the 16 Q rows differ.
	// So each warp contributes to the SAME dK/dV accumulators — we need
	// cross-warp reduction. We'll reduce via SMEM after the loop.
	// =========================================================================

	// Preload dimension for Q+dO streaming: Q has QK_DIM cols, dO has V_DIM cols
	static constexpr usize QDO_DIM = QK_DIM + V_DIM;

	static constexpr usize SMEM_BYTES_DKV =
		sizeof(bf16) * Q_PER_BLOCK * QDO_DIM * GMEM_PRELOAD;

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

	X17_DEVICE void cp_async_q_do(
		GMatrixDynSize<bf16, QK_DIM> gQ,
		GMatrixDynSize<bf16, V_DIM> gDO,
		SMatrix<bf16, Q_PER_BLOCK * GMEM_PRELOAD, QDO_DIM> preload,
		usize step, usize q_steps
	) {
		if (step < q_steps) {
			auto slot = tile_m<Q_PER_BLOCK>(preload, step % GMEM_PRELOAD);
			// Load Q into columns [0, QK_DIM)
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<Q_PER_BLOCK>(gQ, step),
				slot, 0, 0
			);
			// Load dO into columns [QK_DIM, QK_DIM+V_DIM)
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<Q_PER_BLOCK>(gDO, step),
				slot, 0, QK_DIM
			);
		}
	}

	X17_DEVICE void run(
		usize q_cnt, bf16 *gQ_ptr,
		usize kv_cnt, bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
		bf16 *gDO_ptr, bf16 *gDK_ptr, bf16 *gDV_ptr,
		f32 *gL_ptr, f32 *gD_ptr,
		f32 *sink
	) {
		// GMEM Matrices
		GMatrixDynSize<bf16, QK_DIM> gQ{gQ_ptr, q_cnt, Q_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gDO{gDO_ptr, q_cnt};
		GMatrixDynSize<bf16, NONROPE_DIM> gKc{gKc_ptr, kv_cnt, KC_STRIDE};
		GMatrixDynSize<bf16, ROPE_DIM> gKr{gKr_ptr, kv_cnt, KR_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gV{gV_ptr, kv_cnt, V_STRIDE};
		GMatrixDynSize<bf16, QK_DIM> gDK{gDK_ptr, kv_cnt, Q_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gDV{gDV_ptr, kv_cnt, V_STRIDE};

		usize kv_block = blockIdx.x;  // which KV tile this block owns
		usize kv_start = kv_block * KV_PER_STEP;

		u32 smem = 0;
		usize q_warp_idx = (threadIdx.x / WARP_SIZE) % Q_WARPS;
		usize tid = threadIdx.x % WARP_SIZE;

		// SMEM: Q+dO preload region (double-buffered)
		SMatrix<bf16, Q_PER_BLOCK * GMEM_PRELOAD, QDO_DIM> sPreload{smem};

		// --- Load K from GMEM to SMEM, then to registers ---
		// Temporarily use the preload region to hold K (it fits: KV_PER_STEP*PRELOAD_DIM <= Q_PER_BLOCK*QDO_DIM)
		{
			SMatrix<bf16, KV_PER_STEP, PRELOAD_DIM> sK{smem};
			// Load Kc (non-rope part)
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<KV_PER_STEP>(gKc, kv_block),
				sK, 0, 0
			);
			// Load Kr (rope part)
			if constexpr (ROPE_DIM > 0) {
				cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
					threadIdx.x,
					tile_m<KV_PER_STEP>(gKr, kv_block),
					sK, 0, NONROPE_DIM
				);
			}
			cp_async_commit();
			cp_async_wait<0>();
			sync_threads();

			// Load K from SMEM to registers (all warps get the same K)
			// In dKV, K is the "fixed" operand. Each warp loads the same tile.
		}

		// K in registers: QK_TILES fragments of 16x16 bf16
		Fragment_16x16<bf16> rK[QK_TILES];
		{
			SMatrix<bf16, KV_PER_STEP, PRELOAD_DIM> sK{smem};
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sK, 0, i * 16, rK[i]);
			}
		}

		sync_threads(); // ensure all warps done reading K before SMEM reuse

		// Sink and gate
		f32 sink_score = -INFINITY;
		f32 gate = 1.0f;
		if (sink != nullptr) {
			load_gmem_2x32b(sink, sink_score, gate);
		}

		// Determine Q range to iterate over (reversed causal):
		// Only Q rows where q_pos >= kv_pos contribute. The first Q block
		// that can contribute has q_start >= kv_start (accounting for kv_extra).
		usize kv_extra = kv_cnt - q_cnt;
		// kv_pos = kv_start, q_pos = q_block * Q_PER_BLOCK + [0..Q_PER_BLOCK)
		// Causal: q_pos >= kv_pos - kv_extra, i.e. q_block*Q_PER_BLOCK >= kv_start - kv_extra
		// First Q block: ceil((kv_start - kv_extra) / Q_PER_BLOCK), clamped to 0
		isize first_q_block_signed = (isize(kv_start) - isize(kv_extra)) / isize(Q_PER_BLOCK);
		usize first_q_block = first_q_block_signed < 0 ? 0 : usize(first_q_block_signed);
		usize total_q_blocks = q_cnt / Q_PER_BLOCK;
		usize q_steps = total_q_blocks - first_q_block;

		// dK, dV accumulators (per-warp, will reduce across warps later)
		Fragment_16x16<f32> rDK[QK_TILES];
		zero_(rDK);
		Fragment_16x16<f32> rDV[V_TILES];
		zero_(rDV);

		// Start preloading first Q+dO blocks
		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			cp_async_q_do(gQ, gDO, sPreload, p, q_steps);
			cp_async_commit();
		}

		// Sequential loop over Q blocks
		X17_NO_UNROLL for (usize q_step = 0; q_step < q_steps; ++q_step) {
			usize q_block = first_q_block + q_step;
			usize q_block_start = q_block * Q_PER_BLOCK;
			usize my_q_start = q_block_start + q_warp_idx * Q_PER_WARP;

			// Wait for preloaded Q+dO to arrive
			cp_async_wait<GMEM_PRELOAD - 1>();
			sync_threads();

			auto sSlot = tile_m<Q_PER_BLOCK>(sPreload, q_step % GMEM_PRELOAD);

			// Load Q tile for this warp from SMEM
			Fragment_16x16<bf16> rQ[QK_TILES];
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sSlot, q_warp_idx * Q_PER_WARP, i * 16, rQ[i]);
			}

			// S = Q @ K^T (16 Q rows × 16 KV rows)
			Fragment_16x16<f32> rS;
			zero_(rS);
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				mma_a_bt(rQ[i], rK[i], rS);
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

			// Causal mask: mask where kv_pos > q_pos
			// kv_start is the start of the KV tile; q rows are [my_q_start, my_q_start+16)
			// For KV_WARPS==1, all 16 KV rows start at kv_start.
			{
				isize kv_pos = isize(kv_start) - isize(kv_extra);
				isize q_pos = isize(my_q_start);
				if (kv_pos == q_pos) {
					causal_mask_diagonal(rS);
				} else if (kv_pos > q_pos) {
					fill_(rS, -INFINITY);
				}
				// else: kv_pos < q_pos, no masking needed
			}

			// Preload next Q+dO block (overlapped with compute)
			cp_async_q_do(gQ, gDO, sPreload, q_step + GMEM_PRELOAD, q_steps);
			cp_async_commit();

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
