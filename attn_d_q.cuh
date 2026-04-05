#pragma once

#include "utils2.cuh"

#pragma nv_diag_suppress 186

template<typename Attn_forward>
struct Attn_d_q {
	static constexpr usize HEAD_CNT = Attn_forward::HEAD_CNT;
	static constexpr usize HEADS_PER_KERNEL = Attn_forward::HEADS_PER_KERNEL;
	static constexpr usize HEAD_GROUP_CNT = Attn_forward::HEAD_GROUP_CNT;
	static constexpr usize QK_DIM = Attn_forward::QK_DIM;
	static constexpr usize V_DIM = Attn_forward::V_DIM;
	static constexpr bool V_EQUALS_K = Attn_forward::V_EQUALS_K;
	static constexpr usize GMEM_PRELOAD = Attn_forward::GMEM_PRELOAD;

	static_assert(HEADS_PER_KERNEL > 0, "HEADS_PER_KERNEL must be > 0");
	static_assert(HEAD_CNT % HEADS_PER_KERNEL == 0, "HEAD_CNT must be divisible by HEADS_PER_KERNEL");
	static_assert(V_DIM <= QK_DIM, "V_DIM must be <= QK_DIM");

	static constexpr usize QK_TILES = QK_DIM / 16;
	static constexpr usize V_TILES = V_DIM / 16;
	static constexpr usize QK_GROUP_DIM = HEADS_PER_KERNEL * QK_DIM;
	static constexpr usize V_GROUP_DIM = HEADS_PER_KERNEL * V_DIM;
	static constexpr usize PRELOAD_DIM = QK_GROUP_DIM + (V_EQUALS_K ? 0 : V_GROUP_DIM);
	static constexpr usize V_SMEM_COL = V_EQUALS_K ? 0 : QK_GROUP_DIM;

	static constexpr usize Q_WARPS = 4;
	static constexpr usize KV_WARPS = 1;
	static constexpr usize Q_PER_WARP = 16;
	static constexpr usize Q_PER_BLOCK = Q_PER_WARP * Q_WARPS;
	static constexpr usize KV_PER_WARP = 16;
	static constexpr usize KV_PER_STEP = KV_PER_WARP * KV_WARPS;

	static constexpr usize WARPS_PER_BLOCK = Q_WARPS * KV_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
	static constexpr f32 SCORE_SCALE = Attn_forward::SCORE_SCALE;

	static constexpr usize Q_STRIDE = Attn_forward::Q_STRIDE;
	static constexpr usize K_STRIDE = Attn_forward::K_STRIDE;
	static constexpr usize V_STRIDE = Attn_forward::V_STRIDE;
	static constexpr usize O_STRIDE = Attn_forward::O_STRIDE;
	static constexpr usize DO_STRIDE = Attn_forward::DO_STRIDE;
	static constexpr usize DQ_STRIDE = Attn_forward::DQ_STRIDE;

	static_assert((QK_GROUP_DIM * sizeof(bf16)) % 128 == 0, "HEADS_PER_KERNEL must make grouped Q rows 128B aligned");
	static_assert((V_GROUP_DIM * sizeof(bf16)) % 128 == 0, "HEADS_PER_KERNEL must make grouped V rows 128B aligned");
	static_assert((PRELOAD_DIM * sizeof(bf16)) % 128 == 0, "HEADS_PER_KERNEL must make grouped KV preload rows 128B aligned");

	static constexpr usize SMEM_BYTES =
		sizeof(bf16) * (
			KV_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
			+ Q_PER_BLOCK * (QK_GROUP_DIM + 2 * V_GROUP_DIM)
		)
		+ sizeof(f32) * HEADS_PER_KERNEL * Q_PER_BLOCK;

	static constexpr size_t mma_count(size_t seq_len, size_t window_size) {
		seq_len /= 16;
		window_size = std::min(seq_len, window_size > 0 ? window_size / 16 : seq_len);
		usize masked = seq_len - window_size;
		return (
			seq_len * seq_len * (QK_TILES + V_TILES + QK_TILES)
			- masked * masked * (QK_TILES + V_TILES + QK_TILES)
		) / 2;
	}

	static constexpr double flops(size_t seq_len, size_t window_size) {
		return double(mma_count(seq_len, window_size)) * 2.0 * 16.0 * 16.0 * 16.0;
	}

	X17_DEVICE void run(
		usize seq_len, bf16 *gQ_ptr,
		bf16 *gK_ptr, bf16 *gV_ptr,
		bf16 *gOut_ptr, bf16 *gDO_ptr, bf16 *gDQ_ptr,
		f32 *gL_ptr, f32 *gD_ptr,
		f32 *sinks_and_gates,
		usize window_size
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");
		usize i_head_group = blockIdx.y;
		usize i_head_base = i_head_group * HEADS_PER_KERNEL;

		// GMEM Matrices
		GMatrixDynSize<bf16, QK_GROUP_DIM> gQ{gQ_ptr + QK_DIM * i_head_base, seq_len, Q_STRIDE};
		GMatrixDynSize<bf16, QK_GROUP_DIM> gK{gK_ptr + QK_DIM * i_head_base, seq_len, K_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gV{gV_ptr + V_DIM * i_head_base, seq_len, V_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gO{gOut_ptr + V_DIM * i_head_base, seq_len, O_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gDO{gDO_ptr + V_DIM * i_head_base, seq_len, DO_STRIDE};
		GMatrixDynSize<bf16, QK_GROUP_DIM> gDQ{gDQ_ptr + QK_DIM * i_head_base, seq_len, DQ_STRIDE};

		// SMEM layout: KV preload region + Q + dO + O
		u32 smem = 0;
		usize q_warp_idx = threadIdx.x / WARP_SIZE;
		usize tid = threadIdx.x % WARP_SIZE;
		SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		SMatrix<bf16, Q_PER_BLOCK, QK_GROUP_DIM> sQ{sPreload._ptr + sPreload.bytes()};
		SMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> sdO{sQ._ptr + sQ.bytes()};
		SMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> sO{sdO._ptr + sdO.bytes()};
		SMatrix_32b<f32, HEADS_PER_KERNEL, Q_PER_BLOCK> sL{sO._ptr + sO.bytes()};

		// Load Q, dO, O, and L from GMEM to SMEM (no commit — piggyback on first KV commit)
		usize q_block_idx = blockIdx.x;
		usize q_block_start = q_block_idx * Q_PER_BLOCK;
		usize q_block_end = q_block_start + Q_PER_BLOCK;
		usize q_start = q_block_start + q_warp_idx * Q_PER_WARP;
		GMatrix<bf16, Q_PER_BLOCK, QK_GROUP_DIM> gQ_block = tile_m<Q_PER_BLOCK>(gQ, q_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gQ_block, sQ);
		GMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> gDO_block = tile_m<Q_PER_BLOCK>(gDO, q_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gDO_block, sdO);
		GMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> gOut_block = tile_m<Q_PER_BLOCK>(gO, q_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gOut_block, sO);
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			GMatrix<f32, 1, Q_PER_BLOCK> gL_block{gL_ptr + seq_len * (i_head_base + h) + q_block_start};
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gL_block, sL, h, 0);
		}

		// round window_size up without overflow (window_size == 0 means disabled)
		usize max_window_size = std::numeric_limits<usize>::max();
		window_size = window_size > 0 ? window_size : max_window_size;
		usize window_steps = std::min((window_size - 1) / KV_PER_STEP + 1, max_window_size / KV_PER_STEP);
		window_size = window_steps * KV_PER_STEP;

		usize kv_begin = (q_block_start - std::min(q_block_start, window_size)) / KV_PER_STEP;
		usize kv_begin_full = (q_block_end - std::min(q_block_end, window_size)) / KV_PER_STEP;
		usize kv_end_full = q_block_start / KV_PER_STEP;
		usize kv_end = std::min(seq_len, q_block_end + KV_PER_STEP - 1) / KV_PER_STEP;

		// Start preloading K and V from GMEM to SMEM (first commit also commits Q)
		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			Attn_forward::cp_async_kv(gK, gV, sPreload, kv_begin + p, kv_end);
			cp_async_commit();
		}

		// Scalable-Softmax: score_scale = (1.0 / sqrt(QK_DIM)) * ln(n) * logb(e)
		//     1.0 / sqrt(QK_DIM) — standard attention scaling
		//     ln(n)              — SSMax factor (ln(n) = logb(n) / logb(e))
		//     logb(e)            — so we can use expb instead of exp
		// Since we are multiplying and dividing by logb(e), it cancels out, so:
		//     score_scale = (1.0 / sqrt(QK_DIM)) * logb(n)
		// When calculating `n`, we add:
		//     `e` to make sure the SSMax scale >= 1
		//     `1` to account for the sink token
		f32 top_n = std::min(window_size, q_start + tid / 4 + 1) + f32(std::numbers::e_v<f64> + 1.0);
		f32 bot_n = std::min(window_size, q_start + tid / 4 + 9) + f32(std::numbers::e_v<f64> + 1.0);
		f32 top_score_scale = SCORE_SCALE * math::fast::logb(top_n);
		f32 bot_score_scale = SCORE_SCALE * math::fast::logb(bot_n);

		// Sink: a virtual token with no V contribution - it only adds to the
		// softmax denominator, stealing probability from real tokens.
		// sinks_and_gates[2*i_head + 0] = raw score, [2*i_head + 1] = output gate
		f32 gate[HEADS_PER_KERNEL];
		if (sinks_and_gates != nullptr) {
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				f32 sink_score;
				load_gmem_2x32b(sinks_and_gates + 2 * (i_head_base + h), sink_score, gate[h]);
			}
		} else {
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				gate[h] = 1.0f;
			}
		}

		// dQ accumulator
		Fragment_16x16<f32> rDQ_f32[HEADS_PER_KERNEL][QK_TILES];
		zero_(rDQ_f32);

		cp_async_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		f32 top_L[HEADS_PER_KERNEL];
		f32 bot_L[HEADS_PER_KERNEL];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			u32 l_ptr = sL._ptr + h * sL.ROW_BYTES;
			top_L[h] = load_shared_1x32b<f32>(l_ptr + (q_warp_idx * Q_PER_WARP + tid / 4) * sizeof(f32));
			bot_L[h] = load_shared_1x32b<f32>(l_ptr + (q_warp_idx * Q_PER_WARP + tid / 4 + 8) * sizeof(f32));
		}

		// Load Q from SMEM to registers
		Fragment_16x16<bf16> rQ[HEADS_PER_KERNEL][QK_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sQ, q_warp_idx * Q_PER_WARP, h * QK_DIM + i * 16, rQ[h][i]);
			}
		}
		// Load first KV tile from SMEM to registers
		SMatrix<bf16, KV_PER_STEP, PRELOAD_DIM> sKV;
		sKV = tile_m<KV_PER_STEP>(sPreload, kv_begin % GMEM_PRELOAD);
		Fragment_16x16<bf16> rKV[HEADS_PER_KERNEL][QK_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sKV, 0, h * QK_DIM + i * 16, rKV[h][i]);
			}
		}
		// Load dO from SMEM to registers
		Fragment_16x16<bf16> rDO[HEADS_PER_KERNEL][V_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				smem_tile_to_fragment(sdO, q_warp_idx * Q_PER_WARP, h * V_DIM + i * 16, rDO[h][i]);
			}
		}
		// Compute D = rowsum(dO ⊙ O) — load O tiles one at a time from SMEM
		f32 top_D[HEADS_PER_KERNEL];
		f32 bot_D[HEADS_PER_KERNEL];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			top_D[h] = 0.0f;
			bot_D[h] = 0.0f;
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				// TODO - Do we need PRELOAD scheduling?
				Fragment_16x16<bf16> rO;
				smem_tile_to_fragment(sO, q_warp_idx * Q_PER_WARP, h * V_DIM + i * 16, rO);
				top_D[h] = math::fma(f32(rDO[h][i].sub[0][0].first()), f32(rO.sub[0][0].first()), top_D[h]);
				top_D[h] = math::fma(f32(rDO[h][i].sub[0][0].second()), f32(rO.sub[0][0].second()), top_D[h]);
				top_D[h] = math::fma(f32(rDO[h][i].sub[0][1].first()), f32(rO.sub[0][1].first()), top_D[h]);
				top_D[h] = math::fma(f32(rDO[h][i].sub[0][1].second()), f32(rO.sub[0][1].second()), top_D[h]);

				bot_D[h] = math::fma(f32(rDO[h][i].sub[1][0].first()), f32(rO.sub[1][0].first()), bot_D[h]);
				bot_D[h] = math::fma(f32(rDO[h][i].sub[1][0].second()), f32(rO.sub[1][0].second()), bot_D[h]);
				bot_D[h] = math::fma(f32(rDO[h][i].sub[1][1].first()), f32(rO.sub[1][1].first()), bot_D[h]);
				bot_D[h] = math::fma(f32(rDO[h][i].sub[1][1].second()), f32(rO.sub[1][1].second()), bot_D[h]);
			}
			// Reduce D across tid % 4 (4 threads per row hold different column groups)
			top_D[h] += shuffle_xor_sync(top_D[h], 1);
			top_D[h] += shuffle_xor_sync(top_D[h], 2);
			bot_D[h] += shuffle_xor_sync(bot_D[h], 1);
			bot_D[h] += shuffle_xor_sync(bot_D[h], 2);
			// D' = D / gate (since gate is folded into P', dS = P' * (dP - D/gate))
			f32 inv_gate = math::fast::recip(gate[h]);
			top_D[h] *= inv_gate;
			bot_D[h] *= inv_gate;
		}
		// Store D to GMEM (same pattern as L store)
		if ((tid & 1) == 0) {
			usize base = q_block_start + q_warp_idx * Q_PER_WARP;
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				f32 *gD_head_ptr = gD_ptr + seq_len * (i_head_base + h);
				gD_head_ptr[base + (tid / 4) + ((tid & 2) * 4)] = ((tid & 2) == 0 ? top_D[h] : bot_D[h]);
			}
		}

		// Adjust L to fold score_scale and gate into P:
		//   L' = L - logb(gate * score_scale / logb_e)
		//   P' = expb(S*score_scale - L') = P * gate * score_scale / logb_e
		//   D'  = D / gate
		// so dS = P' * (dP - D') and dQ = sum(dS) @ K — no scaling needed.
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			f32 top_grad_scale = f32(1.0 / math::fast::logb_e) * top_score_scale * gate[h];
			f32 bot_grad_scale = f32(1.0 / math::fast::logb_e) * bot_score_scale * gate[h];
			top_L[h] -= math::fast::logb(top_grad_scale);
			bot_L[h] -= math::fast::logb(bot_grad_scale);
		}

		// Sequential loop over KV
		X17_NO_UNROLL for (usize kv_step = kv_begin; kv_step < kv_end; ++kv_step) {
			// S = Q * K^T, interleaved with V load (rKV: K -> V)
			Fragment_16x16<f32> rS_f32[HEADS_PER_KERNEL];
			zero_(rS_f32);
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
					mma_a_bt(rQ[h][i], rKV[h][i], rS_f32[h]);
					smem_tile_to_fragment(
						sKV,
						0,
						(V_EQUALS_K ? h * QK_DIM : QK_GROUP_DIM + h * V_DIM) + i * 16,
						rKV[h][i]
					);
				}
				X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
					mma_a_bt(rQ[h][i], rKV[h][i], rS_f32[h]);
				}

				// WARNING: DON'T get tempted to FMA this into the expb below because
				// scaling must happen before masking to avoid -inf * 0 == NaN when scale == 0
				scale_top_(rS_f32[h], top_score_scale);
				scale_bottom_(rS_f32[h], bot_score_scale);
			}

			// Apply masks
			if (kv_step < kv_begin_full || kv_step >= kv_end_full) {
				// Window mask: mask keys outside the sliding window
				if (kv_step < kv_begin_full) {
					usize diag_warp = Q_WARPS + kv_step - kv_begin_full;
					if (q_warp_idx == diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							Attn_forward::window_mask_diagonal(rS_f32[h]);
						}
					} else if (q_warp_idx > diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							fill_(rS_f32[h], -INFINITY);
						}
					}
				}
				// Causal mask: mask future keys
				if (kv_step >= kv_end_full) {
					usize diag_warp = kv_step - kv_end_full;
					if (q_warp_idx == diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							Attn_forward::causal_mask_diagonal(rS_f32[h]);
						}
					} else if (q_warp_idx < diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							fill_(rS_f32[h], -INFINITY);
						}
					}
				}
			}

			Fragment_16x16<bf16> rDS[HEADS_PER_KERNEL];
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				// P = expb(S - L)
				Fragment_16x16<f32> rP_f32;
				rP_f32.sub[0][0].val0 = math::fast::expb(rS_f32[h].sub[0][0].val0 - top_L[h]);
				rP_f32.sub[0][0].val1 = math::fast::expb(rS_f32[h].sub[0][0].val1 - top_L[h]);
				rP_f32.sub[0][1].val0 = math::fast::expb(rS_f32[h].sub[0][1].val0 - top_L[h]);
				rP_f32.sub[0][1].val1 = math::fast::expb(rS_f32[h].sub[0][1].val1 - top_L[h]);

				rP_f32.sub[1][0].val0 = math::fast::expb(rS_f32[h].sub[1][0].val0 - bot_L[h]);
				rP_f32.sub[1][0].val1 = math::fast::expb(rS_f32[h].sub[1][0].val1 - bot_L[h]);
				rP_f32.sub[1][1].val0 = math::fast::expb(rS_f32[h].sub[1][1].val0 - bot_L[h]);
				rP_f32.sub[1][1].val1 = math::fast::expb(rS_f32[h].sub[1][1].val1 - bot_L[h]);

				// dP = dO * V^T, interleaved with K^T reload (rKV: V -> K^T)
				// K loaded TRANSPOSED because dQ = dS @ K needs B with inner-k = kv.
				Fragment_16x16<f32> rDP;
				zero_(rDP);
				X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
					mma_a_bt(rDO[h][i], rKV[h][i], rDP);
					smem_tile_to_fragment_trans(sKV, 0, h * QK_DIM + i * 16, rKV[h][i]);
				}
				// Load remaining K tiles transposed for dQ GEMM
				X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
					smem_tile_to_fragment_trans(sKV, 0, h * QK_DIM + i * 16, rKV[h][i]);
				}

				// dS = P' * (dP - D') # (gate and score_scale/logb_e folded into P', D' = D/gate)
				Fragment_16x16<f32> rDS_f32;
				rDS_f32.sub[0][0].val0 = rP_f32.sub[0][0].val0 * (rDP.sub[0][0].val0 - top_D[h]);
				rDS_f32.sub[0][0].val1 = rP_f32.sub[0][0].val1 * (rDP.sub[0][0].val1 - top_D[h]);
				rDS_f32.sub[0][1].val0 = rP_f32.sub[0][1].val0 * (rDP.sub[0][1].val0 - top_D[h]);
				rDS_f32.sub[0][1].val1 = rP_f32.sub[0][1].val1 * (rDP.sub[0][1].val1 - top_D[h]);

				rDS_f32.sub[1][0].val0 = rP_f32.sub[1][0].val0 * (rDP.sub[1][0].val0 - bot_D[h]);
				rDS_f32.sub[1][0].val1 = rP_f32.sub[1][0].val1 * (rDP.sub[1][0].val1 - bot_D[h]);
				rDS_f32.sub[1][1].val0 = rP_f32.sub[1][1].val0 * (rDP.sub[1][1].val0 - bot_D[h]);
				rDS_f32.sub[1][1].val1 = rP_f32.sub[1][1].val1 * (rDP.sub[1][1].val1 - bot_D[h]);
				cast(rDS_f32, rDS[h]);
			}

			{ // Get more data from GMEM
				// Wait for the next batch of GMEM -> SMEM preloads to complete
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sKV = tile_m<KV_PER_STEP>(sPreload, (kv_step + 1) % GMEM_PRELOAD);

				// Preload next KV tiles from GMEM
				Attn_forward::cp_async_kv(gK, gV, sPreload, kv_step + GMEM_PRELOAD, kv_end);
				cp_async_commit();
			}

			// dQ += dS * K, interleaved with next K load
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
					mma_a_bt(rDS[h], rKV[h][i], rDQ_f32[h][i]);
					smem_tile_to_fragment(sKV, 0, h * QK_DIM + i * 16, rKV[h][i]);
				}
			}
		}

		Fragment_16x16<bf16> rDQ[HEADS_PER_KERNEL * QK_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				cast(rDQ_f32[h][i], rDQ[h * QK_TILES + i]);
			}
		}
		GMatrix<bf16, Q_PER_BLOCK, QK_GROUP_DIM> gDQ_block = tile_m<Q_PER_BLOCK>(gDQ, blockIdx.x);
		store(rDQ, gDQ_block, q_warp_idx * Q_PER_WARP, 0);
	}
};

template<typename Attn_d_q>
__global__ __launch_bounds__(Attn_d_q::THREADS_PER_BLOCK) void
attn_d_q(
	usize seq_len, bf16 *gQ_ptr,
	bf16 *gK_ptr, bf16 *gV_ptr,
	bf16 *gOut_ptr, bf16 *gDO_ptr, bf16 *gDQ_ptr,
	f32 *gL_ptr, f32 *gD_ptr,
	f32 *sinks_and_gates,
	usize window_size
) {
	Attn_d_q attn_d_q = Attn_d_q();
	attn_d_q.run(seq_len, gQ_ptr, gK_ptr, gV_ptr, gOut_ptr, gDO_ptr, gDQ_ptr, gL_ptr, gD_ptr, sinks_and_gates, window_size);
}
