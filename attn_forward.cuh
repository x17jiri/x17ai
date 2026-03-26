#pragma once

#include "utils2.cuh"

#pragma nv_diag_suppress 186

template<
	const usize _HEAD_CNT,
	const usize _NONROPE_DIM,
	const usize _ROPE_DIM,
	const usize _V_DIM,
	const usize Q_WARPS,
	const usize KV_WARPS,
	const bool _V_EQUALS_K = false,
	const usize _GMEM_PRELOAD = 2
>
struct Attn_forward {
	// Re-expose template parameters as static constants for dependent types
	static constexpr usize HEAD_CNT = _HEAD_CNT;
	static constexpr usize NONROPE_DIM = _NONROPE_DIM;
	static constexpr usize ROPE_DIM = _ROPE_DIM;
	static constexpr usize V_DIM = _V_DIM;
	static constexpr bool V_EQUALS_K = _V_EQUALS_K;
	static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

	static_assert(V_DIM <= NONROPE_DIM, "V_DIM must be <= NONROPE_DIM");

	static constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	static constexpr usize NONROPE_TILES = NONROPE_DIM / 16;
	static constexpr usize ROPE_TILES = ROPE_DIM / 16;
	static constexpr usize QK_TILES = NONROPE_TILES + ROPE_TILES;
	static constexpr usize V_TILES = V_DIM / 16;
	static constexpr usize PRELOAD_DIM = QK_DIM + (V_EQUALS_K ? 0 : V_DIM);
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

	static constexpr usize SMEM_BYTES =
		SMEM_OVERLAP_Q_WITH_KV
			?
				sizeof(bf16) * KV_PER_STEP * PRELOAD_DIM * (GMEM_PRELOAD - 1)
				+ std::max(
					sizeof(bf16) * KV_PER_STEP * PRELOAD_DIM,
					sizeof(bf16) * Q_PER_BLOCK * QK_DIM
				)
			:
				sizeof(bf16) * KV_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
				+ sizeof(bf16) * Q_PER_BLOCK * QK_DIM;

	static X17_DEVICE void causal_mask_diagonal(Fragment_16x16<f32> &rScores) {
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

	X17_DEVICE void online_softmax(
		bool first_step,
		SoftmaxStats &top,
		SoftmaxStats &bot,
		Fragment_16x16<f32> &rScores,
		Fragment_16x16<f32> (&rOut)[V_TILES]
	) {
		// The `max` in `top` and `bot` is for the entire owned rows.
		// The `sum` is just the elements owned by the current thread.
		// Complete sum is calculated in combine_and_store().

		// Step 1: `max` of the owned values
		f32 new_top_max = math::max(
			math::max(rScores.sub[0][0].val0, rScores.sub[0][0].val1),
			math::max(rScores.sub[0][1].val0, rScores.sub[0][1].val1)
		);
		f32 new_bot_max = math::max(
			math::max(rScores.sub[1][0].val0, rScores.sub[1][0].val1),
			math::max(rScores.sub[1][1].val0, rScores.sub[1][1].val1)
		);

		// Step 2: Rescale outputs if needed
		f32 top_rescale = 1.0f;
		f32 bot_rescale = 1.0f;
		constexpr f32 THRESHOLD = 5.0 / math::fast::logb_2;
		bool needs_rescale = math::max(new_top_max - top.max, new_bot_max - bot.max) > THRESHOLD;
		if (any_sync(needs_rescale)) {
			new_top_max = math::max(new_top_max, shuffle_xor_sync(new_top_max, 1));
			new_top_max = math::max(new_top_max, shuffle_xor_sync(new_top_max, 2));

			new_bot_max = math::max(new_bot_max, shuffle_xor_sync(new_bot_max, 1));
			new_bot_max = math::max(new_bot_max, shuffle_xor_sync(new_bot_max, 2));

			new_top_max =
				new_top_max - top.max > THRESHOLD
					? new_top_max + THRESHOLD
					: top.max;

			new_bot_max =
				new_bot_max - bot.max > THRESHOLD
					? new_bot_max + THRESHOLD
					: bot.max;

			if (!first_step) {
				top_rescale = math::fast::expb(top.max - new_top_max);
				scale_top_(rOut, top_rescale);

				bot_rescale = math::fast::expb(bot.max - new_bot_max);
				scale_bottom_(rOut, bot_rescale);
			}

			top.max = new_top_max;
			bot.max = new_bot_max;
		}

		// Step 3: Replace scores with expb(score - max)
		rScores.sub[0][0].val0 = math::fast::expb(rScores.sub[0][0].val0 - top.max);
		rScores.sub[0][0].val1 = math::fast::expb(rScores.sub[0][0].val1 - top.max);
		rScores.sub[0][1].val0 = math::fast::expb(rScores.sub[0][1].val0 - top.max);
		rScores.sub[0][1].val1 = math::fast::expb(rScores.sub[0][1].val1 - top.max);

		rScores.sub[1][0].val0 = math::fast::expb(rScores.sub[1][0].val0 - bot.max);
		rScores.sub[1][0].val1 = math::fast::expb(rScores.sub[1][0].val1 - bot.max);
		rScores.sub[1][1].val0 = math::fast::expb(rScores.sub[1][1].val0 - bot.max);
		rScores.sub[1][1].val1 = math::fast::expb(rScores.sub[1][1].val1 - bot.max);

		// Step 4: `sum` of the owned values
		f32 top_add = (
			(rScores.sub[0][0].val0 + rScores.sub[0][0].val1)
			+ (rScores.sub[0][1].val0 + rScores.sub[0][1].val1)
		);
		top.sum = math::fma(top.sum, top_rescale, top_add);

		f32 bot_add = (
			(rScores.sub[1][0].val0 + rScores.sub[1][0].val1)
			+ (rScores.sub[1][1].val0 + rScores.sub[1][1].val1)
		);
		bot.sum = math::fma(bot.sum, bot_rescale, bot_add);
	}

	X17_DEVICE void combine_and_store(
		Fragment_16x16<f32> (&rOut)[V_TILES],
		SoftmaxStats top,
		SoftmaxStats bot,
		f32 sink_score,
		f32 top_score_scale,
		f32 bot_score_scale,
		f32 gate,
		u32 smem,
		usize q_start,
		usize q_warp_idx,
		usize kv_warp_idx,
		GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block,
		f32 *gL_ptr
	) {
		// Complete the row-wise sum reduction within each warp
		top.sum += shuffle_xor_sync(top.sum, 1);
		top.sum += shuffle_xor_sync(top.sum, 2);

		bot.sum += shuffle_xor_sync(bot.sum, 1);
		bot.sum += shuffle_xor_sync(bot.sum, 2);

		// Reduce across KV_WARPS warps that share the same Q rows via SMEM.
		// Each Q-warp group operates independently.
		// SMEM layout per Q-warp group:
		//   sReduce: (KV_WARPS-1) * Q_PER_WARP rows * V_DIM cols (f32)
		//   stats:   KV_WARPS * WARP_SIZE * 4 * f32
		constexpr usize REDUCE_BYTES = sizeof(f32) * (KV_WARPS - 1) * Q_PER_WARP * V_DIM;
		constexpr usize STATS_BYTES = KV_WARPS * WARP_SIZE * 4 * sizeof(f32);
		constexpr usize GROUP_BYTES = REDUCE_BYTES + STATS_BYTES;
		u32 group_smem = smem + q_warp_idx * GROUP_BYTES;
		SMatrix<f32, (KV_WARPS - 1) * Q_PER_WARP, V_DIM> sReduce{group_smem};
		u32 stats_smem = sReduce._ptr + sReduce.bytes();
		[[maybe_unused]] usize tid = threadIdx.x % WARP_SIZE;

		// Step 1: All warps store their stats to smem (each warp gets its own slot)
		if constexpr (KV_WARPS > 1) {
			cp_async_wait<0>(); // make sure no remaining cp_async lands in our scratch
			sync_threads(); // make sure no threads are reading or writing to SMEM

			store_shared_4x32b(
				stats_smem + (kv_warp_idx * WARP_SIZE + tid) * (4 * sizeof(f32)),
				top.sum, top.max, bot.sum, bot.max
			);

			bar_sync<KV_WARPS * WARP_SIZE>(q_warp_idx + 1);
		}

		f32 top_sink_scaled = sink_score * top_score_scale;
		f32 bot_sink_scaled = sink_score * bot_score_scale;

		f32 global_top_max, global_top_sum;
		f32 global_bot_max, global_bot_sum;

		// Step 2: Every thread reads all KV_WARPS stats, computes global max/sum
		if constexpr (KV_WARPS > 1) {
			global_top_max = top_sink_scaled;
			global_bot_max = bot_sink_scaled;

			f32 w_top_sum[KV_WARPS], w_top_max[KV_WARPS];
			f32 w_bot_sum[KV_WARPS], w_bot_max[KV_WARPS];
			X17_UNROLL for (usize w = 0; w < KV_WARPS; w++) {
				load_shared_4x32b(
					stats_smem + (w * WARP_SIZE + tid) * (4 * sizeof(f32)),
					w_top_sum[w], w_top_max[w], w_bot_sum[w], w_bot_max[w]
				);
				global_top_max = math::max(global_top_max, w_top_max[w]);
				global_bot_max = math::max(global_bot_max, w_bot_max[w]);
			}

			global_top_sum = math::fast::expb(top_sink_scaled - global_top_max);
			global_bot_sum = math::fast::expb(bot_sink_scaled - global_bot_max);

			X17_UNROLL for (usize w = 0; w < KV_WARPS; w++) {
				global_top_sum = math::fma(
					w_top_sum[w],
					math::fast::expb(w_top_max[w] - global_top_max),
					global_top_sum
				);
				global_bot_sum = math::fma(
					w_bot_sum[w],
					math::fast::expb(w_bot_max[w] - global_bot_max),
					global_bot_sum
				);
			}
		} else {
			global_top_max = math::max(top.max, top_sink_scaled);
			global_bot_max = math::max(bot.max, bot_sink_scaled);

			global_top_sum = math::fma(
				top.sum,
				math::fast::expb(top.max - global_top_max),
				math::fast::expb(top_sink_scaled - global_top_max)
			);
			global_bot_sum = math::fma(
				bot.sum,
				math::fast::expb(bot.max - global_bot_max),
				math::fast::expb(bot_sink_scaled - global_bot_max)
			);
		}

		// Step 3: Rescale, folding in normalization and gate
		f32 top_L = math::fast::logb(global_top_sum) + global_top_max;
		f32 bot_L = math::fast::logb(global_bot_sum) + global_bot_max;

		f32 top_rescale = math::fast::expb(top.max - top_L) * gate;
		f32 bot_rescale = math::fast::expb(bot.max - bot_L) * gate;

		scale_top_(rOut, top_rescale);
		scale_bottom_(rOut, bot_rescale);

		// Step 4: accumulate results
		if constexpr (KV_WARPS > 1) {
			if (kv_warp_idx != 0) {
				// KV warps 1..N store their rescaled+normalized values to smem
				SMatrix<f32, Q_PER_WARP, V_DIM> slot = tile_m<Q_PER_WARP>(sReduce, kv_warp_idx - 1);
				fragments_to_smem(rOut, slot);

				bar_sync<KV_WARPS * WARP_SIZE>(q_warp_idx + 1);

				return;
			} else {
				// KV warp 0 accumulates and stores to gmem
				bar_sync<KV_WARPS * WARP_SIZE>(q_warp_idx + 1);

				X17_UNROLL for (usize w = 0; w < KV_WARPS - 1; w++) {
					SMatrix<f32, Q_PER_WARP, V_DIM> slot = tile_m<Q_PER_WARP>(sReduce, w);
					Fragment_16x16<f32> temp[V_TILES];
					smem_to_fragments(temp, slot);
					acc_(rOut, temp);
				}
			}
		}

		store(rOut, gOut_block, q_warp_idx * Q_PER_WARP, 0);

		if (gL_ptr != nullptr && (tid & 1) == 0) {
			gL_ptr[q_start + (tid / 4) + ((tid & 2) * 4)] = ((tid & 2) == 0 ? top_L : bot_L);
		}
	}

	static X17_DEVICE void cp_async_kv(
		GMatrixDynSize<bf16, NONROPE_DIM> gKc,
		GMatrixDynSize<bf16, ROPE_DIM> gKr,
		GMatrixDynSize<bf16, V_DIM> gV,
		SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
		usize p, usize kv_warp_idx, usize kv_steps
	) {
		if (p < kv_steps) {
			auto preload_tile = tile_m<KV_PER_WARP>(tile_m<KV_PER_STEP>(preload, p % GMEM_PRELOAD), kv_warp_idx);
			cp_async_gmem_to_smem<Q_WARPS * WARP_SIZE>(
				threadIdx.x % (Q_WARPS * WARP_SIZE),
				tile_m<KV_PER_WARP>(tile_m<KV_PER_STEP>(gKc, p), kv_warp_idx),
				preload_tile, 0, 0
			);
			cp_async_gmem_to_smem<Q_WARPS * WARP_SIZE>(
				threadIdx.x % (Q_WARPS * WARP_SIZE),
				tile_m<KV_PER_WARP>(tile_m<KV_PER_STEP>(gKr, p), kv_warp_idx),
				preload_tile, 0, NONROPE_DIM
			);
			if constexpr (!V_EQUALS_K) {
				cp_async_gmem_to_smem<Q_WARPS * WARP_SIZE>(
					threadIdx.x % (Q_WARPS * WARP_SIZE),
					tile_m<KV_PER_WARP>(tile_m<KV_PER_STEP>(gV, p), kv_warp_idx),
					preload_tile, 0, QK_DIM
				);
			}
		}
	}

	X17_DEVICE void run(
		usize q_cnt, bf16 *gQ_ptr,
		usize kv_cnt, bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
		bf16 *gOut_ptr,
		f32 *gL_ptr,
		f32 *sink
	) {
		// GMEM Matrices
		GMatrixDynSize<bf16, QK_DIM> gQ{gQ_ptr, q_cnt, Q_STRIDE};
		GMatrixDynSize<bf16, NONROPE_DIM> gKc{gKc_ptr, kv_cnt, KC_STRIDE};
		GMatrixDynSize<bf16, ROPE_DIM> gKr{gKr_ptr, kv_cnt, KR_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gV{gV_ptr, kv_cnt, V_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gOut{gOut_ptr, q_cnt};

		// SMEM layout: K + V preload region; last tile may overlap with Q preload
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

		// Load Q from GMEM to SMEM (no commit — piggyback on first KV commit)
		usize q_block_idx = blockIdx.x;
		usize q_block_start = q_block_idx * Q_PER_BLOCK;
		usize q_start = q_block_start + q_warp_idx * Q_PER_WARP;
		GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gQ_block = tile_m<Q_PER_BLOCK>(gQ, q_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gQ_block, sQ);

		// TODO: this assumes `kv_cnt >= q_cnt`
		usize kv_extra = kv_cnt - q_cnt;
		usize kv_steps = (kv_extra + q_block_start + Q_PER_BLOCK + KV_PER_STEP - 1) / KV_PER_STEP;
		kv_steps = std::min(kv_steps, kv_cnt / KV_PER_STEP);
		usize full_kv_steps = (kv_extra + q_block_start) / KV_PER_STEP;
		full_kv_steps = std::min(full_kv_steps, kv_steps);

		// Start preloading K and V from GMEM to SMEM  (first commit also commits Q)
		// When Q overlaps KV SMEM, don't use the last preload tile yet (it holds Q)
		X17_UNROLL for (usize p = 0; p < EARLY_PRELOAD; ++p) {
			cp_async_kv(gKc, gKr, gV, sPreload, p, kv_warp_idx, kv_steps);
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
		f32 top_n = q_start + tid / 4 + 1 + 1; // the final `+ 1` is for sink
		f32 bot_n = q_start + tid / 4 + 9 + 1;
		f32 top_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(top_n);
		f32 bot_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(bot_n);

		SoftmaxStats r_top; r_top.max = std::numeric_limits<f32>::lowest(); r_top.sum = 0.0f;
		SoftmaxStats r_bot; r_bot.max = std::numeric_limits<f32>::lowest(); r_bot.sum = 0.0f;
		Fragment_16x16<f32> rOut[V_TILES];
		zero_(rOut);

		cp_async_wait<EARLY_PRELOAD - 1>();
		sync_threads();

		// Load Q from SMEM to registers
		Fragment_16x16<bf16> rQ[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sQ, q_warp_idx * Q_PER_WARP, i * 16, rQ[i]);
		}
		// Now that Q is in registers, reuse its SMEM for KV
		if constexpr (SMEM_OVERLAP_Q_WITH_KV) {
			cp_async_kv(gKc, gKr, gV, sPreload, GMEM_PRELOAD - 1, kv_warp_idx, kv_steps);
			cp_async_commit();
		}
		// Load first KV tile from SMEM to registers
		SMatrix<bf16, KV_PER_WARP, PRELOAD_DIM> sKV;
		sKV = tile_m<KV_PER_WARP>(tile_m<KV_PER_STEP>(sPreload, 0), kv_warp_idx);
		Fragment_16x16<bf16> rKV[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
		}

		// Sequential loop over KV
		X17_NO_UNROLL for (usize kv_step = 0; kv_step < kv_steps; ++kv_step) {
			// rScores = Q * K^T, interleaved with V load (rKV: K -> V)
			Fragment_16x16<f32> rScores_f32;
			zero_(rScores_f32);
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				mma_a_bt(rQ[i], rKV[i], rScores_f32);
				smem_tile_to_fragment_trans(sKV, 0, V_SMEM_COL + i * 16, rKV[i]);
			}
			X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
				mma_a_bt(rQ[i], rKV[i], rScores_f32);
			}

			// Scale scores must happen before masking to avoid -inf * 0 == NaN when score_scale == 0
			scale_top_(rScores_f32, top_score_scale);
			scale_bottom_(rScores_f32, bot_score_scale);

			// Causal mask: diagonal tile or full mask if past this warp's boundary
			if (kv_step >= full_kv_steps) {
				usize kv_pos = kv_step * KV_PER_STEP + kv_warp_idx * KV_PER_WARP;
				if (kv_pos == kv_extra + q_start) {
					causal_mask_diagonal(rScores_f32);
				} else if (kv_pos > kv_extra + q_start) {
					fill_(rScores_f32, -INFINITY);
				}
			}

			online_softmax(kv_step == 0, r_top, r_bot, rScores_f32, rOut);
			Fragment_16x16<bf16> rScores;
			cast(rScores_f32, rScores);

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
				cp_async_kv(gKc, gKr, gV, sPreload, kv_step + GMEM_PRELOAD, kv_warp_idx, kv_steps);
				cp_async_commit();
			}

			// rOut += rScores * V, interleaved with next K load
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				mma_a_bt(rScores, rKV[i], rOut[i]);
				smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
			}
			// Load remaining K tiles that weren't covered by V interleaving
			X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
				smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
			}
		}

		GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = tile_m<Q_PER_BLOCK>(gOut, q_block_idx);
		combine_and_store(rOut, r_top, r_bot, sink_score, top_score_scale, bot_score_scale, gate, smem, q_start, q_warp_idx, kv_warp_idx, gOut_block, gL_ptr);
	}
};

template<typename Attn_forward>
__global__ __launch_bounds__(Attn_forward::THREADS_PER_BLOCK) void
attn_forward(
	usize q_cnt, bf16 *gQ_ptr,
	usize kv_cnt, bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
	bf16 *gOut_ptr,
	f32 *gL_ptr,
	f32 *sink
) {
	auto attn_forward = Attn_forward();
	attn_forward.run(q_cnt, gQ_ptr, kv_cnt, gKc_ptr, gKr_ptr, gV_ptr, gOut_ptr, gL_ptr, sink);
}
