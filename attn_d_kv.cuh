#pragma once

#include "utils2.cuh"

#pragma nv_diag_suppress 186

template<typename Attn_forward>
struct Attn_d_kv {
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
	static constexpr usize PRELOAD_DIM = HEADS_PER_KERNEL * (QK_DIM + V_DIM); // preload region holds both Q and dO
	static constexpr usize KV_SMEM_DIM = QK_GROUP_DIM + (V_EQUALS_K ? 0 : V_GROUP_DIM);
	static constexpr usize V_SMEM_COL = V_EQUALS_K ? 0 : QK_GROUP_DIM;

	static constexpr usize Q_WARPS = 1;
	static constexpr usize KV_WARPS = 4;
	static constexpr usize Q_PER_WARP = 16;
	static constexpr usize Q_PER_STEP = Q_PER_WARP * Q_WARPS;
	static constexpr usize KV_PER_WARP = 16;
	static constexpr usize KV_PER_BLOCK = KV_PER_WARP * KV_WARPS;

	static constexpr usize WARPS_PER_BLOCK = Q_WARPS * KV_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
	static constexpr f32 SCORE_SCALE = Attn_forward::SCORE_SCALE;

	static constexpr usize Q_STRIDE = Attn_forward::Q_STRIDE;
	static constexpr usize K_STRIDE = Attn_forward::K_STRIDE;
	static constexpr usize V_STRIDE = Attn_forward::V_STRIDE;
	static constexpr usize DO_STRIDE = Attn_forward::DO_STRIDE;
	static constexpr usize DK_STRIDE = Attn_forward::DK_STRIDE;
	static constexpr usize DV_STRIDE = Attn_forward::DV_STRIDE;

	static_assert((QK_GROUP_DIM * sizeof(bf16)) % 128 == 0, "HEADS_PER_KERNEL must make grouped Q/K rows 128B aligned");
	static_assert((V_GROUP_DIM * sizeof(bf16)) % 128 == 0, "HEADS_PER_KERNEL must make grouped V rows 128B aligned");
	static_assert((PRELOAD_DIM * sizeof(bf16)) % 128 == 0, "HEADS_PER_KERNEL must make grouped preload rows 128B aligned");

	static constexpr usize SMEM_BYTES =
		sizeof(bf16) * (Q_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD)
		+ sizeof(bf16) * (KV_PER_BLOCK * KV_SMEM_DIM)
		+ sizeof(f32) * (HEADS_PER_KERNEL * GMEM_PRELOAD * Q_PER_STEP * 2); // sL + sD

	static constexpr size_t mma_count(size_t seq_len, size_t window_size) {
		seq_len /= 16;
		window_size = std::min(seq_len, window_size > 0 ? window_size / 16 : seq_len);
		usize masked = seq_len - window_size;
		return (
			seq_len * seq_len * (QK_TILES + V_TILES + V_TILES + QK_TILES)
			- masked * masked * (QK_TILES + V_TILES + V_TILES + QK_TILES)
		) / 2;
	}

	static constexpr double flops(size_t seq_len, size_t window_size) {
		return double(mma_count(seq_len, window_size)) * 2.0 * 16.0 * 16.0 * 16.0;
	}

	static X17_DEVICE void cp_async_q_do_ld(
		GMatrixDynSize<bf16, QK_GROUP_DIM> gQ,
		GMatrixDynSize<bf16, V_GROUP_DIM> gDO,
		f32 *gL_ptr,
		f32 *gD_ptr,
		SMatrix<bf16, Q_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
		SMatrix_32b<f32, HEADS_PER_KERNEL * GMEM_PRELOAD, Q_PER_STEP> l_preload,
		SMatrix_32b<f32, HEADS_PER_KERNEL * GMEM_PRELOAD, Q_PER_STEP> d_preload,
		usize p, usize q_end,
		usize seq_len,
		usize i_head_base
	) {
		if (p < q_end) {
			SMatrix<bf16, Q_PER_STEP, PRELOAD_DIM> slot = tile_m<Q_PER_STEP>(preload, p % GMEM_PRELOAD);
			// Load Q into columns [0, QK_GROUP_DIM)
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<Q_PER_STEP>(gQ, p),
				slot, 0, 0
			);
			// Load dO into columns [QK_GROUP_DIM, QK_GROUP_DIM+V_GROUP_DIM)
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<Q_PER_STEP>(gDO, p),
				slot, 0, QK_GROUP_DIM
			);
			usize slot_row = (p % GMEM_PRELOAD) * HEADS_PER_KERNEL;
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				GMatrix<f32, 1, Q_PER_STEP> gL{gL_ptr + seq_len * (i_head_base + h) + p * Q_PER_STEP};
				GMatrix<f32, 1, Q_PER_STEP> gD{gD_ptr + seq_len * (i_head_base + h) + p * Q_PER_STEP};
				cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gL, l_preload, slot_row + h, 0);
				cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gD, d_preload, slot_row + h, 0);
			}
		}
	}

	X17_DEVICE void run(
		usize seq_len, bf16 *gQ_ptr,
		bf16 *gK_ptr, bf16 *gV_ptr,
		bf16 *gDO_ptr, bf16 *gDK_ptr, bf16 *gDV_ptr,
		f32 *gL_ptr, f32 *gD_ptr,
		f32 *sinks_and_gates,
		usize window_size
	) {
		static_assert(Q_WARPS == 1, "current algorithm doesn't reduce over Q warps");
		usize i_head_group = blockIdx.y;
		usize i_head_base = i_head_group * HEADS_PER_KERNEL;

		// GMEM Matrices
		GMatrixDynSize<bf16, QK_GROUP_DIM> gQ{gQ_ptr + QK_DIM * i_head_base, seq_len, Q_STRIDE};
		GMatrixDynSize<bf16, QK_GROUP_DIM> gK{gK_ptr + QK_DIM * i_head_base, seq_len, K_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gV{gV_ptr + V_DIM * i_head_base, seq_len, V_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gDO{gDO_ptr + V_DIM * i_head_base, seq_len, DO_STRIDE};
		GMatrixDynSize<bf16, QK_GROUP_DIM> gDK{gDK_ptr + QK_DIM * i_head_base, seq_len, DK_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gDV{gDV_ptr + V_DIM * i_head_base, seq_len, DV_STRIDE};

		// SMEM layout: Q + dO preload region + KV + sL + sD
		u32 smem = 0;
		usize kv_warp_idx = threadIdx.x / WARP_SIZE;
		usize tid = threadIdx.x % WARP_SIZE;
		SMatrix<bf16, Q_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		SMatrix<bf16, KV_PER_BLOCK, KV_SMEM_DIM> sKV{sPreload._ptr + sPreload.bytes()};
		SMatrix_32b<f32, HEADS_PER_KERNEL * GMEM_PRELOAD, Q_PER_STEP> sL{sKV._ptr + sKV.bytes()};
		SMatrix_32b<f32, HEADS_PER_KERNEL * GMEM_PRELOAD, Q_PER_STEP> sD{sL._ptr + sL.bytes()};

		// Load K/V from GMEM to SMEM (no commit — piggyback on first KV commit)
		usize kv_block_idx = blockIdx.x;
		usize kv_block_start = kv_block_idx * KV_PER_BLOCK;
		GMatrix<bf16, KV_PER_BLOCK, QK_GROUP_DIM> gK_block = tile_m<KV_PER_BLOCK>(gK, kv_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gK_block, sKV, 0, 0);
		if constexpr (!V_EQUALS_K) {
			GMatrix<bf16, KV_PER_BLOCK, V_GROUP_DIM> gV_block = tile_m<KV_PER_BLOCK>(gV, kv_block_idx);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gV_block, sKV, 0, QK_GROUP_DIM);
		}

		// round window_size up without overflow (window_size == 0 means disabled)
		usize max_window_size = std::numeric_limits<usize>::max();
		window_size = window_size > 0 ? window_size : max_window_size;
		usize window_steps = std::min((window_size - 1) / Q_PER_STEP + 1, max_window_size / Q_PER_STEP);
		window_size = window_steps * Q_PER_STEP;

		// `q_begin` and `window_steps` are each ≤ USIZE_MAX / Q_PER_STEP, so their sum
		// plus KV_WARPS can't overflow as long as Q_PER_STEP >= 3.
		static_assert(Q_PER_STEP >= 3, "Q_PER_STEP >= 3 required to avoid overflow in q_end");
		usize q_begin = kv_block_start / Q_PER_STEP;
		usize q_begin_full = q_begin + KV_WARPS;
		usize q_end_full = q_begin + window_steps;
		usize q_end = std::min(q_end_full + KV_WARPS, seq_len / Q_PER_STEP);

		// Start preloading first Q+dO blocks (first commit also commits KV)
		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			cp_async_q_do_ld(gQ, gDO, gL_ptr, gD_ptr, sPreload, sL, sD, q_begin + p, q_end, seq_len, i_head_base);
			cp_async_commit();
		}

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

		// dK, dV accumulators
		Fragment_16x16<f32> rDK_f32[HEADS_PER_KERNEL][QK_TILES];
		zero_(rDK_f32);
		Fragment_16x16<f32> rDV_f32[HEADS_PER_KERNEL][V_TILES];
		zero_(rDV_f32);

		cp_async_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		// Load K from SMEM into registers
		SMatrix<bf16, KV_PER_WARP, KV_SMEM_DIM> sKV_warp = tile_m<KV_PER_WARP>(sKV, kv_warp_idx);
		Fragment_16x16<bf16> rK[HEADS_PER_KERNEL][QK_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sKV_warp, 0, h * QK_DIM + i * 16, rK[h][i]);
			}
		}
		// Load V from SMEM into registers
		// When V_EQUALS_K, V starts at col 0; otherwise at col QK_DIM
		Fragment_16x16<bf16> rV[HEADS_PER_KERNEL][V_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				smem_tile_to_fragment(sKV_warp, 0, V_SMEM_COL + h * V_DIM + i * 16, rV[h][i]);
			}
		}
		// Load first Q + dO tile from SMEM to registers
		SMatrix<bf16, Q_PER_STEP, PRELOAD_DIM> sSlot = tile_m<Q_PER_STEP>(sPreload, q_begin % GMEM_PRELOAD);
		SMatrix_32b<f32, HEADS_PER_KERNEL, Q_PER_STEP> sLSlot = tile_m<HEADS_PER_KERNEL>(sL, q_begin % GMEM_PRELOAD);
		SMatrix_32b<f32, HEADS_PER_KERNEL, Q_PER_STEP> sDSlot = tile_m<HEADS_PER_KERNEL>(sD, q_begin % GMEM_PRELOAD);
		Fragment_16x16<bf16> rQ[HEADS_PER_KERNEL][QK_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sSlot, 0, h * QK_DIM + i * 16, rQ[h][i]);
			}
		}
		Fragment_16x16<bf16> rDO[HEADS_PER_KERNEL][V_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				smem_tile_to_fragment(sSlot, 0, QK_GROUP_DIM + h * V_DIM + i * 16, rDO[h][i]);
			}
		}

		// Sequential loop over Q
		f32 logb_gate[HEADS_PER_KERNEL];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			logb_gate[h] = math::fast::logb(gate[h]);
		}
		X17_NO_UNROLL for (usize q_step = q_begin; q_step < q_end; ++q_step) {
			// Load L and D from SMEM
			f32 top_L[HEADS_PER_KERNEL];
			f32 top_D[HEADS_PER_KERNEL];
			f32 bot_L[HEADS_PER_KERNEL];
			f32 bot_D[HEADS_PER_KERNEL];
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				u32 l_ptr = sLSlot._ptr + h * sLSlot.ROW_BYTES;
				u32 d_ptr = sDSlot._ptr + h * sDSlot.ROW_BYTES;
				top_L[h] = load_shared_1x32b<f32>(l_ptr + (tid / 4) * sizeof(f32));
				top_D[h] = load_shared_1x32b<f32>(d_ptr + (tid / 4) * sizeof(f32));
				bot_L[h] = load_shared_1x32b<f32>(l_ptr + (tid / 4 + 8) * sizeof(f32));
				bot_D[h] = load_shared_1x32b<f32>(d_ptr + (tid / 4 + 8) * sizeof(f32));
			}

			// S = Q * K^T
			Fragment_16x16<f32> rS_f32[HEADS_PER_KERNEL];
			zero_(rS_f32);
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
					mma_a_bt(rQ[h][i], rK[h][i], rS_f32[h]);
					rQ[h][i].transpose_();
				}
			}

			// Per-Q-row score_scale: score_scale = logb(n) / sqrt(QK_DIM)
			// In the MMA fragment layout, tid/4 gives the row within the 8-row sub-tile
			usize q_start = q_step * Q_PER_STEP;
			f32 top_n = std::min(window_size, q_start + tid / 4 + 1) + f32(std::numbers::e_v<f64> + 1.0);
			f32 bot_n = std::min(window_size, q_start + tid / 4 + 9) + f32(std::numbers::e_v<f64> + 1.0);
			f32 top_score_scale = SCORE_SCALE * math::fast::logb(top_n);
			f32 bot_score_scale = SCORE_SCALE * math::fast::logb(bot_n);

			// WARNING: DON'T get tempted to FMA this into the expb below because
			// scaling must happen before masking to avoid -inf * 0 == NaN when scale == 0
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				scale_top_(rS_f32[h], top_score_scale);
				scale_bottom_(rS_f32[h], bot_score_scale);
			}

			// Apply masks
			if (q_step < q_begin_full || q_step >= q_end_full) {
				// Causal mask: first KV_WARPS Q steps straddle the causal diagonal
				if (q_step < q_begin_full) {
					usize diag_warp = q_step - q_begin;
					if (kv_warp_idx == diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							Attn_forward::causal_mask_diagonal(rS_f32[h]);
						}
					} else if (kv_warp_idx > diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							fill_(rS_f32[h], -INFINITY);
						}
					}
				}
				// Window mask: last KV_WARPS Q steps straddle the window boundary
				if (q_step >= q_end_full) {
					usize diag_warp = q_step - q_end_full;
					if (kv_warp_idx == diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							Attn_forward::window_mask_diagonal(rS_f32[h]);
						}
					} else if (kv_warp_idx < diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							fill_(rS_f32[h], -INFINITY);
						}
					}
				}
			}

			Fragment_16x16<bf16> rDS[HEADS_PER_KERNEL];
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				// Adjust L to fold gate into P
				//   L' = L - logb(gate)
				//   P = expb(S*score_scale - L') = gate * P_softmax
				top_L[h] -= logb_gate[h];
				bot_L[h] -= logb_gate[h];

				// P = expb(S*score_scale - L_g) = gate * P_softmax
				Fragment_16x16<f32> rP_f32;
				rP_f32.sub[0][0].val0 = math::fast::expb(rS_f32[h].sub[0][0].val0 - top_L[h]);
				rP_f32.sub[0][0].val1 = math::fast::expb(rS_f32[h].sub[0][0].val1 - top_L[h]);
				rP_f32.sub[0][1].val0 = math::fast::expb(rS_f32[h].sub[0][1].val0 - top_L[h]);
				rP_f32.sub[0][1].val1 = math::fast::expb(rS_f32[h].sub[0][1].val1 - top_L[h]);

				rP_f32.sub[1][0].val0 = math::fast::expb(rS_f32[h].sub[1][0].val0 - bot_L[h]);
				rP_f32.sub[1][0].val1 = math::fast::expb(rS_f32[h].sub[1][0].val1 - bot_L[h]);
				rP_f32.sub[1][1].val0 = math::fast::expb(rS_f32[h].sub[1][1].val0 - bot_L[h]);
				rP_f32.sub[1][1].val1 = math::fast::expb(rS_f32[h].sub[1][1].val1 - bot_L[h]);

				// dP = dO @ V^T
				Fragment_16x16<f32> rDP;
				zero_(rDP);
				X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
					mma_a_bt(rDO[h][i], rV[h][i], rDP);
					rDO[h][i].transpose_();
				}

				// dV += P^T @ dO  (P = gate * P_softmax, no rescale needed)
				Fragment_16x16<bf16> rP;
				cast(rP_f32, rP);
				rP.transpose_();
				X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
					mma_a_bt(rP, rDO[h][i], rDV_f32[h][i]);
				}

				// dS = (score_scale / logb_e) * P * (dP - D')
				Fragment_16x16<f32> rDS_f32;
				rDS_f32.sub[0][0].val0 = rP_f32.sub[0][0].val0 * (rDP.sub[0][0].val0 - top_D[h]);
				rDS_f32.sub[0][0].val1 = rP_f32.sub[0][0].val1 * (rDP.sub[0][0].val1 - top_D[h]);
				rDS_f32.sub[0][1].val0 = rP_f32.sub[0][1].val0 * (rDP.sub[0][1].val0 - top_D[h]);
				rDS_f32.sub[0][1].val1 = rP_f32.sub[0][1].val1 * (rDP.sub[0][1].val1 - top_D[h]);

				rDS_f32.sub[1][0].val0 = rP_f32.sub[1][0].val0 * (rDP.sub[1][0].val0 - bot_D[h]);
				rDS_f32.sub[1][0].val1 = rP_f32.sub[1][0].val1 * (rDP.sub[1][0].val1 - bot_D[h]);
				rDS_f32.sub[1][1].val0 = rP_f32.sub[1][1].val0 * (rDP.sub[1][1].val0 - bot_D[h]);
				rDS_f32.sub[1][1].val1 = rP_f32.sub[1][1].val1 * (rDP.sub[1][1].val1 - bot_D[h]);

				// P already has gate folded in.
				f32 top_dk_scale = top_score_scale * f32(1.0 / math::fast::logb_e);
				f32 bot_dk_scale = bot_score_scale * f32(1.0 / math::fast::logb_e);
				scale_top_(rDS_f32, top_dk_scale);
				scale_bottom_(rDS_f32, bot_dk_scale);

				cast(rDS_f32, rDS[h]);
				rDS[h].transpose_();
			}

			{ // Get more data from GMEM
				// Wait for the next batch of GMEM -> SMEM preloads to complete
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sSlot = tile_m<Q_PER_STEP>(sPreload, (q_step + 1) % GMEM_PRELOAD);
				sLSlot = tile_m<HEADS_PER_KERNEL>(sL, (q_step + 1) % GMEM_PRELOAD);
				sDSlot = tile_m<HEADS_PER_KERNEL>(sD, (q_step + 1) % GMEM_PRELOAD);

				// Preload next Q+dO tiles from GMEM
				cp_async_q_do_ld(gQ, gDO, gL_ptr, gD_ptr, sPreload, sL, sD, q_step + GMEM_PRELOAD, q_end, seq_len, i_head_base);
				cp_async_commit();
			}

			// dK += dS^T @ Q — transpose dS, then MMA with each Q tile
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
					mma_a_bt(rDS[h], rQ[h][i], rDK_f32[h][i]);
					smem_tile_to_fragment(sSlot, 0, h * QK_DIM + i * 16, rQ[h][i]);
				}

				// Load dO from SMEM for next iteration
				X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
					smem_tile_to_fragment(sSlot, 0, QK_GROUP_DIM + h * V_DIM + i * 16, rDO[h][i]);
				}
			}
		}

		// Store dK, dV to GMEM
		Fragment_16x16<bf16> rDK[HEADS_PER_KERNEL * QK_TILES];
		Fragment_16x16<bf16> rDV[HEADS_PER_KERNEL * V_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				cast(rDK_f32[h][i], rDK[h * QK_TILES + i]);
			}
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				cast(rDV_f32[h][i], rDV[h * V_TILES + i]);
			}
		}
		GMatrix<bf16, KV_PER_BLOCK, QK_GROUP_DIM> gDK_block = tile_m<KV_PER_BLOCK>(gDK, kv_block_idx);
		store(rDK, gDK_block, kv_warp_idx * KV_PER_WARP, 0);
		GMatrix<bf16, KV_PER_BLOCK, V_GROUP_DIM> gDV_block = tile_m<KV_PER_BLOCK>(gDV, kv_block_idx);
		store(rDV, gDV_block, kv_warp_idx * KV_PER_WARP, 0);
	}
};

template<typename Attn_d_kv>
__global__ __launch_bounds__(Attn_d_kv::THREADS_PER_BLOCK) void
attn_d_kv(
	usize seq_len, bf16 *gQ_ptr,
	bf16 *gK_ptr, bf16 *gV_ptr,
	bf16 *gDO_ptr, bf16 *gDK_ptr, bf16 *gDV_ptr,
	f32 *gL_ptr, f32 *gD_ptr,
	f32 *sinks_and_gates,
	usize window_size
) {
	Attn_d_kv attn_d_kv = Attn_d_kv();
	attn_d_kv.run(seq_len, gQ_ptr, gK_ptr, gV_ptr, gDO_ptr, gDK_ptr, gDV_ptr, gL_ptr, gD_ptr, sinks_and_gates, window_size);
}
