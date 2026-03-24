#pragma once

#include "utils2.cuh"
#pragma nv_diag_suppress 186

template<typename Attn_forward>
struct Attn_d_q {
	static constexpr usize HEAD_CNT = Attn_forward::HEAD_CNT;
	static constexpr usize NONROPE_DIM = Attn_forward::NONROPE_DIM;
	static constexpr usize ROPE_DIM = Attn_forward::ROPE_DIM;
	static constexpr usize V_DIM = Attn_forward::V_DIM;
	static constexpr usize Q_WARPS = 4;
	static constexpr usize KV_WARPS = 1;
	static constexpr bool V_EQUALS_K = Attn_forward::V_EQUALS_K;
	static constexpr usize GMEM_PRELOAD = Attn_forward::GMEM_PRELOAD;

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

	static constexpr usize KC_STRIDE = Attn_forward::KC_STRIDE;
	static constexpr usize KR_STRIDE = Attn_forward::KR_STRIDE;
	static constexpr usize V_STRIDE = Attn_forward::V_STRIDE;
	static constexpr usize Q_STRIDE = Attn_forward::Q_STRIDE;

	static constexpr usize SMEM_BYTES =
		SMEM_OVERLAP_Q_WITH_KV
			?
				sizeof(bf16) * KV_PER_STEP * PRELOAD_DIM * (GMEM_PRELOAD - 1)
				+ std::max(
					sizeof(bf16) * KV_PER_STEP * PRELOAD_DIM,
					sizeof(bf16) * Q_PER_BLOCK * (QK_DIM + 2 * V_DIM)
				)
			:
				sizeof(bf16) * KV_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
				+ sizeof(bf16) * Q_PER_BLOCK * (QK_DIM + 2 * V_DIM);

	X17_DEVICE void run(
		usize q_cnt, bf16 *gQ_ptr,
		usize kv_cnt, bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
		bf16 *gOut_ptr, bf16 *gDO_ptr, bf16 *gDQ_ptr,
		f32 *gL_ptr, f32 *gD_ptr,
		f32 *sink
	) {
		static_assert(KV_WARPS == 1, "d_q store requires KV_WARPS == 1");

		// GMEM Matrices
		GMatrixDynSize<bf16, NONROPE_DIM> gKc{gKc_ptr, kv_cnt, KC_STRIDE};
		GMatrixDynSize<bf16, ROPE_DIM> gKr{gKr_ptr, kv_cnt, KR_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gV{gV_ptr, kv_cnt, V_STRIDE};
		GMatrixDynSize<bf16, QK_DIM> gQ_full{gQ_ptr, q_cnt, Q_STRIDE};
		GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gQ_block = tile_m<Q_PER_BLOCK>(gQ_full, blockIdx.x);
		GMatrixDynSize<bf16, V_DIM> gOut_full{gOut_ptr, q_cnt};
		GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = tile_m<Q_PER_BLOCK>(gOut_full, blockIdx.x);
		GMatrixDynSize<bf16, V_DIM> gDO_full{gDO_ptr, q_cnt};
		GMatrix<bf16, Q_PER_BLOCK, V_DIM> gDO_block = tile_m<Q_PER_BLOCK>(gDO_full, blockIdx.x);
		GMatrixDynSize<bf16, QK_DIM> gDQ_full{gDQ_ptr, q_cnt, Q_STRIDE};
		GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gDQ_block = tile_m<Q_PER_BLOCK>(gDQ_full, blockIdx.x);

		// SMEM layout: KV preload region + Q + dO + O
		u32 smem = 0;
		usize q_warp_idx = (threadIdx.x / WARP_SIZE) % Q_WARPS;
		usize kv_warp_idx = (threadIdx.x / WARP_SIZE) / Q_WARPS;
		usize tid = threadIdx.x % WARP_SIZE;

		SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		SMatrix<bf16, Q_PER_BLOCK, QK_DIM> sQ{
			SMEM_OVERLAP_Q_WITH_KV
				? tile_m<KV_PER_STEP>(sPreload, GMEM_PRELOAD - 1)._ptr
				: sPreload._ptr + sPreload.bytes()
		};
		SMatrix<bf16, Q_PER_BLOCK, V_DIM> sdO{sQ._ptr + sQ.bytes()};
		SMatrix<bf16, Q_PER_BLOCK, V_DIM> sO{sdO._ptr + sdO.bytes()};

		// Load Q, dO, O from GMEM to SMEM (no commit — piggyback on first KV commit)
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gQ_block, sQ);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gDO_block, sdO);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gOut_block, sO);

		// TODO: this assumes `kv_cnt >= q_cnt`
		usize kv_extra = kv_cnt - q_cnt;
		usize block_q_start = blockIdx.x * Q_PER_BLOCK;
		usize kv_steps = (kv_extra + block_q_start + Q_PER_BLOCK + KV_PER_STEP - 1) / KV_PER_STEP;
		kv_steps = std::min(kv_steps, kv_cnt / KV_PER_STEP);
		usize full_kv_steps = (kv_extra + block_q_start) / KV_PER_STEP;
		full_kv_steps = std::min(full_kv_steps, kv_steps);

		// Start preloading K and V from GMEM to SMEM  (first commit also commits Q)
		// When Q overlaps KV SMEM, don't use the last preload tile yet (it holds Q)
		X17_UNROLL for (usize p = 0; p < EARLY_PRELOAD; ++p) {
			Attn_forward::cp_async_kv(gKc, gKr, gV, sPreload, p, kv_warp_idx, kv_steps);
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

		// Scalable-Softmax: score_scale = (1.0 / sqrt(QK_DIM)) * ln(n) * logb(e)
		//     1.0 / sqrt(QK_DIM) — standard attention scaling
		//     ln(n)              — SSMax factor (ln(n) = logb(n) / logb(e))
		//     logb(e)            — so we can use expb instead of exp
		// Since we are multiplying and dividing by logb(e), it cancels out, so:
		//     score_scale = (1.0 / sqrt(QK_DIM)) * logb(n)
		usize my_q_start = block_q_start + q_warp_idx * Q_PER_WARP;
		f32 top_n = my_q_start + tid / 4 + 1 + 1; // the final `+ 1` is for sink
		f32 bot_n = my_q_start + tid / 4 + 9 + 1;
		f32 top_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(top_n);
		f32 bot_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(bot_n);

		// Load logsumexp from forward pass, adjusted to fold score_scale AND gate into P:
		//   L' = L - logb(gate * score_scale / logb_e)
		//   P' = expb(S*score_scale - L') = P * gate * score_scale / logb_e
		//   D'  = D / gate
		// so dS = P' * (dP - D') and dQ = sum(dS) @ K — no scaling needed.
		f32 top_grad_scale = f32(1.0 / math::fast::logb_e) * gate * top_score_scale;
		f32 bot_grad_scale = f32(1.0 / math::fast::logb_e) * gate * bot_score_scale;
		f32 top_L = gL_ptr[my_q_start + tid / 4] - math::fast::logb(top_grad_scale);
		f32 bot_L = gL_ptr[my_q_start + tid / 4 + 8] - math::fast::logb(bot_grad_scale);

		// dQ accumulator
		Fragment_16x16<f32> rDQ[QK_TILES];
		zero_(rDQ);

		cp_async_wait<EARLY_PRELOAD - 1>();
		sync_threads();

		// Load Q from SMEM to registers
		Fragment_16x16<bf16> rQ[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sQ, q_warp_idx * Q_PER_WARP, i * 16, rQ[i]);
		}
		// Load dO from SMEM to registers
		Fragment_16x16<bf16> rDO[V_TILES];
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			smem_tile_to_fragment(sdO, q_warp_idx * Q_PER_WARP, i * 16, rDO[i]);
		}
		// Compute D = rowsum(dO ⊙ O) — load O tiles one at a time from SMEM
		f32 top_D = 0.0f, bot_D = 0.0f;
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			// TODO - Do we need PRELOAD scheduling?
			Fragment_16x16<bf16> rO;
			smem_tile_to_fragment(sO, q_warp_idx * Q_PER_WARP, i * 16, rO);
			top_D = math::fma(f32(rDO[i].sub[0][0].first()), f32(rO.sub[0][0].first()), top_D);
			top_D = math::fma(f32(rDO[i].sub[0][0].second()), f32(rO.sub[0][0].second()), top_D);
			top_D = math::fma(f32(rDO[i].sub[0][1].first()), f32(rO.sub[0][1].first()), top_D);
			top_D = math::fma(f32(rDO[i].sub[0][1].second()), f32(rO.sub[0][1].second()), top_D);

			bot_D = math::fma(f32(rDO[i].sub[1][0].first()), f32(rO.sub[1][0].first()), bot_D);
			bot_D = math::fma(f32(rDO[i].sub[1][0].second()), f32(rO.sub[1][0].second()), bot_D);
			bot_D = math::fma(f32(rDO[i].sub[1][1].first()), f32(rO.sub[1][1].first()), bot_D);
			bot_D = math::fma(f32(rDO[i].sub[1][1].second()), f32(rO.sub[1][1].second()), bot_D);
		}
		// Reduce across tid % 4 (4 threads per row hold different column groups)
		top_D += shuffle_xor_sync(top_D, 1);
		top_D += shuffle_xor_sync(top_D, 2);
		bot_D += shuffle_xor_sync(bot_D, 1);
		bot_D += shuffle_xor_sync(bot_D, 2);
		// D' = D / gate (since gate is folded into P'', dS = P'' * (dP - D/gate))
		f32 inv_gate = math::fast::recip(gate);
		top_D *= inv_gate;
		bot_D *= inv_gate;
		// Store D to GMEM (same pattern as L store)
		if ((tid & 1) == 0) {
			usize base = block_q_start + q_warp_idx * Q_PER_WARP;
			gD_ptr[base + (tid / 4) + ((tid & 2) * 4)] = ((tid & 2) == 0 ? top_D : bot_D);
		}
		// Load first KV tile from SMEM to registers
		SMatrix<bf16, KV_PER_WARP, PRELOAD_DIM> sKV;
		sKV = tile_m<KV_PER_WARP>(tile_m<KV_PER_STEP>(sPreload, 0), kv_warp_idx);
		Fragment_16x16<bf16> rKV[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
		}

		// Now that Q is in registers, reuse its SMEM for KV
		if constexpr (SMEM_OVERLAP_Q_WITH_KV) {
			Attn_forward::cp_async_kv(gKc, gKr, gV, sPreload, GMEM_PRELOAD - 1, kv_warp_idx, kv_steps);
			cp_async_commit();
		}

		// Sequential loop over KV
		X17_NO_UNROLL for (usize kv_step = 0; kv_step < kv_steps; ++kv_step) {
			// S = Q * K^T, interleaved with V load (rKV: K -> V)
			// NOTE: V loaded NON-transposed (unlike forward) because dP = dO @ V^T
			// needs B with inner-k = dim (matching dO), not kv.
			Fragment_16x16<f32> rS;
			zero_(rS);
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				mma_a_bt(rQ[i], rKV[i], rS);
				smem_tile_to_fragment(sKV, 0, V_SMEM_COL + i * 16, rKV[i]);
			}
			X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
				mma_a_bt(rQ[i], rKV[i], rS);
			}

			// WARNING: DON'T get tempted to FMA this into the expb below
			// Scale scores must happen before masking to avoid -inf * 0 == NaN when score_scale == 0
			scale_top_(rS, top_score_scale);
			scale_bottom_(rS, bot_score_scale);

			// Causal mask
			if (kv_step >= full_kv_steps) {
				usize kv_pos = kv_step * KV_PER_STEP + kv_warp_idx * KV_PER_WARP;
				if (kv_pos == kv_extra + my_q_start) {
					Attn_forward::causal_mask_diagonal(rS);
				} else if (kv_pos > kv_extra + my_q_start) {
					fill_(rS, -INFINITY);
				}
			}

			// P = expb(S - L)
			Fragment_16x16<f32> rP;
			rP.sub[0][0].val0 = math::fast::expb(rS.sub[0][0].val0 - top_L);
			rP.sub[0][0].val1 = math::fast::expb(rS.sub[0][0].val1 - top_L);
			rP.sub[0][1].val0 = math::fast::expb(rS.sub[0][1].val0 - top_L);
			rP.sub[0][1].val1 = math::fast::expb(rS.sub[0][1].val1 - top_L);

			rP.sub[1][0].val0 = math::fast::expb(rS.sub[1][0].val0 - bot_L);
			rP.sub[1][0].val1 = math::fast::expb(rS.sub[1][0].val1 - bot_L);
			rP.sub[1][1].val0 = math::fast::expb(rS.sub[1][1].val0 - bot_L);
			rP.sub[1][1].val1 = math::fast::expb(rS.sub[1][1].val1 - bot_L);

			// dP = dO * V^T, interleaved with K^T reload (rKV: V -> K^T)
			// K loaded TRANSPOSED because dQ = dS @ K needs B with inner-k = kv.
			Fragment_16x16<f32> rDP;
			zero_(rDP);
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				mma_a_bt(rDO[i], rKV[i], rDP);
				smem_tile_to_fragment_trans(sKV, 0, i * 16, rKV[i]);
			}
			// Load remaining K tiles transposed for dQ GEMM
			X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
				smem_tile_to_fragment_trans(sKV, 0, i * 16, rKV[i]);
			}

			// dS = P' * (dP - D')  (gate and score_scale/logb_e folded into P', D' = D/gate)
			Fragment_16x16<f32> rDS_f32;
			rDS_f32.sub[0][0].val0 = rP.sub[0][0].val0 * (rDP.sub[0][0].val0 - top_D);
			rDS_f32.sub[0][0].val1 = rP.sub[0][0].val1 * (rDP.sub[0][0].val1 - top_D);
			rDS_f32.sub[0][1].val0 = rP.sub[0][1].val0 * (rDP.sub[0][1].val0 - top_D);
			rDS_f32.sub[0][1].val1 = rP.sub[0][1].val1 * (rDP.sub[0][1].val1 - top_D);

			rDS_f32.sub[1][0].val0 = rP.sub[1][0].val0 * (rDP.sub[1][0].val0 - bot_D);
			rDS_f32.sub[1][0].val1 = rP.sub[1][0].val1 * (rDP.sub[1][0].val1 - bot_D);
			rDS_f32.sub[1][1].val0 = rP.sub[1][1].val0 * (rDP.sub[1][1].val0 - bot_D);
			rDS_f32.sub[1][1].val1 = rP.sub[1][1].val1 * (rDP.sub[1][1].val1 - bot_D);

			Fragment_16x16<bf16> rDS;
			cast(rDS_f32, rDS);

			{ // Get more data from GMEM
				// Wait for the next batch of GMEM -> SMEM preloads to complete
				cp_async_wait<GMEM_PRELOAD - 2>();
				if constexpr (KV_WARPS > 1) {
					bar_sync<Q_WARPS * WARP_SIZE>(kv_warp_idx + 1);
				} else {
					sync_threads();
				}
				sKV = tile_m<KV_PER_WARP>(tile_m<KV_PER_STEP>(sPreload, (kv_step + 1) % GMEM_PRELOAD), kv_warp_idx);

				// Preload next KV tiles from GMEM
				Attn_forward::cp_async_kv(gKc, gKr, gV, sPreload, kv_step + GMEM_PRELOAD, kv_warp_idx, kv_steps);
				cp_async_commit();
			}

			// dQ += dS * K, interleaved with next K load
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				mma_a_bt(rDS, rKV[i], rDQ[i]);
				smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
			}
		}

		store(rDQ, gDQ_block, q_warp_idx * Q_PER_WARP, 0);
	}
};

template<typename Attn_d_q>
__global__ __launch_bounds__(Attn_d_q::THREADS_PER_BLOCK) void
attn_d_q(
	usize q_cnt, bf16 *gQ_ptr,
	usize kv_cnt, bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
	bf16 *gOut_ptr, bf16 *gDO_ptr, bf16 *gDQ_ptr,
	f32 *gL_ptr, f32 *gD_ptr,
	f32 *sink
) {
	auto attn_d_q = Attn_d_q();
	attn_d_q.run(q_cnt, gQ_ptr, kv_cnt, gKc_ptr, gKr_ptr, gV_ptr, gOut_ptr, gDO_ptr, gDQ_ptr, gL_ptr, gD_ptr, sink);
}
