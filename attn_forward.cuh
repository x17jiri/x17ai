#pragma once

#include "utils2.cuh"

#pragma nv_diag_suppress 186

template<
	const usize _HEAD_CNT,
	const usize _NONROPE_DIM,
	const usize _ROPE_DIM,
	const usize _V_DIM,
	const bool _V_EQUALS_K = false,
	const usize _GMEM_PRELOAD = 2,
	const usize _WINDOW_SIZE = 0
>
struct Attn_forward {
	// Re-expose template parameters as static constants for dependent types
	static constexpr usize HEAD_CNT = _HEAD_CNT;
	static constexpr usize NONROPE_DIM = _NONROPE_DIM;
	static constexpr usize ROPE_DIM = _ROPE_DIM;
	static constexpr usize V_DIM = _V_DIM;
	static constexpr bool V_EQUALS_K = _V_EQUALS_K;
	static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;
	static constexpr usize WINDOW_SIZE = _WINDOW_SIZE;

	static_assert(WINDOW_SIZE == 0 || WINDOW_SIZE % 16 == 0,
		"WINDOW_SIZE must be 0 (disabled) or a multiple of 16");

	static_assert(V_DIM <= NONROPE_DIM, "V_DIM must be <= NONROPE_DIM");

	static constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	static constexpr usize NONROPE_TILES = NONROPE_DIM / 16;
	static constexpr usize ROPE_TILES = ROPE_DIM / 16;
	static constexpr usize QK_TILES = NONROPE_TILES + ROPE_TILES;
	static constexpr usize V_TILES = V_DIM / 16;
	static constexpr usize PRELOAD_DIM = QK_DIM + (V_EQUALS_K ? 0 : V_DIM);
	static constexpr usize V_SMEM_COL = V_EQUALS_K ? 0 : QK_DIM;

	static constexpr usize Q_WARPS = 4;
	static constexpr usize KV_WARPS = 1;
	static constexpr usize Q_PER_WARP = 16;
	static constexpr usize Q_PER_BLOCK = Q_PER_WARP * Q_WARPS;
	static constexpr usize KV_PER_WARP = 16;
	static constexpr usize KV_PER_STEP = KV_PER_WARP * KV_WARPS;

	static constexpr usize WARPS_PER_BLOCK = Q_WARPS * KV_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

	// TODO - other matrices (dO, O, ...) should have their stride
	static constexpr usize KC_STRIDE = NONROPE_DIM * HEAD_CNT;
	static constexpr usize KR_STRIDE = ROPE_DIM * HEAD_CNT;
	static constexpr usize V_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize Q_STRIDE = QK_DIM * HEAD_CNT;

	static constexpr usize SMEM_BYTES =
		sizeof(bf16) * (
			KV_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
			+ Q_PER_BLOCK * QK_DIM
		);

	static X17_DEVICE void causal_mask_diagonal(Fragment_16x16<f32> &rS_f32) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize q = tid / 4;           // 0..7
		usize k = 2 * (tid % 4);    // 0,2,4,6
		constexpr f32 NEG_INF = -INFINITY;

		rS_f32.sub[0][1].val0 = NEG_INF;
		rS_f32.sub[0][1].val1 = NEG_INF;

		rS_f32.sub[0][0].val0 = k <= q ? rS_f32.sub[0][0].val0 : NEG_INF;
		rS_f32.sub[1][1].val0 = k <= q ? rS_f32.sub[1][1].val0 : NEG_INF;

		rS_f32.sub[0][0].val1 = k + 1 <= q ? rS_f32.sub[0][0].val1 : NEG_INF;
		rS_f32.sub[1][1].val1 = k + 1 <= q ? rS_f32.sub[1][1].val1 : NEG_INF;
	}

	static X17_DEVICE void window_mask_diagonal(Fragment_16x16<f32> &rS_f32) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize q = tid / 4;           // 0..7
		usize k = 2 * (tid % 4);    // 0,2,4,6
		constexpr f32 NEG_INF = -INFINITY;

		rS_f32.sub[1][0].val0 = NEG_INF;
		rS_f32.sub[1][0].val1 = NEG_INF;

		rS_f32.sub[0][0].val0 = k > q ? rS_f32.sub[0][0].val0 : NEG_INF;
		rS_f32.sub[1][1].val0 = k > q ? rS_f32.sub[1][1].val0 : NEG_INF;

		rS_f32.sub[0][0].val1 = k + 1 > q ? rS_f32.sub[0][0].val1 : NEG_INF;
		rS_f32.sub[1][1].val1 = k + 1 > q ? rS_f32.sub[1][1].val1 : NEG_INF;
	}

	static constexpr f32 ONLINE_SOFTMAX_THRESHOLD = 5.0 / math::fast::logb_2;

	static constexpr size_t mma_count(size_t seq_len) {
		return (seq_len / 16) * (seq_len / 16) * (QK_TILES + V_TILES) / 2;
	}

	static constexpr double flops(size_t seq_len) {
		return double(mma_count(seq_len)) * 2.0 * 16.0 * 16.0 * 16.0;
	}

	X17_DEVICE void online_softmax(
		bool first_step,
		SoftmaxStats &top,
		SoftmaxStats &bot,
		Fragment_16x16<f32> &rS_f32,
		Fragment_16x16<f32> (&rO_f32)[V_TILES]
	) {
		// The `max` in `top` and `bot` is for the entire owned rows.
		// The `sum` is just the elements owned by the current thread.
		// Complete sum is calculated in combine_and_store().

		// Step 1: `max` of the owned values
		f32 new_top_max = math::max(
			math::max(rS_f32.sub[0][0].val0, rS_f32.sub[0][0].val1),
			math::max(rS_f32.sub[0][1].val0, rS_f32.sub[0][1].val1)
		);
		f32 new_bot_max = math::max(
			math::max(rS_f32.sub[1][0].val0, rS_f32.sub[1][0].val1),
			math::max(rS_f32.sub[1][1].val0, rS_f32.sub[1][1].val1)
		);

		// Step 2: Rescale outputs if needed
		f32 top_rescale = 1.0f;
		f32 bot_rescale = 1.0f;
		bool needs_rescale = math::max(new_top_max - top.max, new_bot_max - bot.max) > ONLINE_SOFTMAX_THRESHOLD;
		if (any_sync(needs_rescale)) {
			new_top_max = math::max(new_top_max, shuffle_xor_sync(new_top_max, 1));
			new_top_max = math::max(new_top_max, shuffle_xor_sync(new_top_max, 2));

			new_bot_max = math::max(new_bot_max, shuffle_xor_sync(new_bot_max, 1));
			new_bot_max = math::max(new_bot_max, shuffle_xor_sync(new_bot_max, 2));

			new_top_max =
				new_top_max - top.max > ONLINE_SOFTMAX_THRESHOLD
					? new_top_max + ONLINE_SOFTMAX_THRESHOLD
					: top.max;

			new_bot_max =
				new_bot_max - bot.max > ONLINE_SOFTMAX_THRESHOLD
					? new_bot_max + ONLINE_SOFTMAX_THRESHOLD
					: bot.max;

			top_rescale = math::fast::expb(top.max - new_top_max);
			bot_rescale = math::fast::expb(bot.max - new_bot_max);

			if (!first_step) {
				scale_top_(rO_f32, top_rescale);
				scale_bottom_(rO_f32, bot_rescale);
			}

			top.max = new_top_max;
			bot.max = new_bot_max;
		}

		// Step 3: Replace scores with expb(score - max)
		rS_f32.sub[0][0].val0 = math::fast::expb(rS_f32.sub[0][0].val0 - top.max);
		rS_f32.sub[0][0].val1 = math::fast::expb(rS_f32.sub[0][0].val1 - top.max);
		rS_f32.sub[0][1].val0 = math::fast::expb(rS_f32.sub[0][1].val0 - top.max);
		rS_f32.sub[0][1].val1 = math::fast::expb(rS_f32.sub[0][1].val1 - top.max);

		rS_f32.sub[1][0].val0 = math::fast::expb(rS_f32.sub[1][0].val0 - bot.max);
		rS_f32.sub[1][0].val1 = math::fast::expb(rS_f32.sub[1][0].val1 - bot.max);
		rS_f32.sub[1][1].val0 = math::fast::expb(rS_f32.sub[1][1].val0 - bot.max);
		rS_f32.sub[1][1].val1 = math::fast::expb(rS_f32.sub[1][1].val1 - bot.max);

		// Step 4: `sum` of the owned values
		f32 top_add = (
			(rS_f32.sub[0][0].val0 + rS_f32.sub[0][0].val1)
			+ (rS_f32.sub[0][1].val0 + rS_f32.sub[0][1].val1)
		);
		top.sum = math::fma(top.sum, top_rescale, top_add);

		f32 bot_add = (
			(rS_f32.sub[1][0].val0 + rS_f32.sub[1][0].val1)
			+ (rS_f32.sub[1][1].val0 + rS_f32.sub[1][1].val1)
		);
		bot.sum = math::fma(bot.sum, bot_rescale, bot_add);
	}

	X17_DEVICE void combine_and_store(
		Fragment_16x16<f32> (&rO_f32)[V_TILES],
		SoftmaxStats top,
		SoftmaxStats bot,
		f32 gate,
		usize q_start,
		usize q_warp_idx,
		GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block,
		f32 *gL_ptr
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");

		// Complete the row-wise sum reduction within each warp
		top.sum += shuffle_xor_sync(top.sum, 1);
		top.sum += shuffle_xor_sync(top.sum, 2);

		bot.sum += shuffle_xor_sync(bot.sum, 1);
		bot.sum += shuffle_xor_sync(bot.sum, 2);

		// Rescale, folding in normalization and gate
		f32 top_L = math::fast::logb(top.sum) + top.max;
		f32 bot_L = math::fast::logb(bot.sum) + bot.max;

		f32 top_rescale = math::fast::divide(gate, top.sum);
		f32 bot_rescale = math::fast::divide(gate, bot.sum);

		scale_top_(rO_f32, top_rescale);
		scale_bottom_(rO_f32, bot_rescale);

		store(rO_f32, gOut_block, q_warp_idx * Q_PER_WARP, 0);

		usize tid = threadIdx.x % WARP_SIZE;
		if (gL_ptr != nullptr && (tid & 1) == 0) {
			gL_ptr[q_start + (tid / 4) + ((tid & 2) * 4)] = ((tid & 2) == 0 ? top_L : bot_L);
		}
	}

	static X17_DEVICE void cp_async_kv(
		GMatrixDynSize<bf16, NONROPE_DIM> gKc,
		GMatrixDynSize<bf16, ROPE_DIM> gKr,
		GMatrixDynSize<bf16, V_DIM> gV,
		SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
		usize p, usize kv_steps
	) {
		if (p < kv_steps) {
			auto preload_tile = tile_m<KV_PER_STEP>(preload, p % GMEM_PRELOAD);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<KV_PER_STEP>(gKc, p),
				preload_tile, 0, 0
			);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<KV_PER_STEP>(gKr, p),
				preload_tile, 0, NONROPE_DIM
			);
			if constexpr (!V_EQUALS_K) {
				cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
					threadIdx.x,
					tile_m<KV_PER_STEP>(gV, p),
					preload_tile, 0, QK_DIM
				);
			}
		}
	}

	X17_DEVICE void run(
		usize seq_len, bf16 *gQ_ptr,
		bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
		bf16 *gOut_ptr,
		f32 *gL_ptr,
		f32 *sink
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");

		// GMEM Matrices
		GMatrixDynSize<bf16, QK_DIM> gQ{gQ_ptr, seq_len, Q_STRIDE};
		GMatrixDynSize<bf16, NONROPE_DIM> gKc{gKc_ptr, seq_len, KC_STRIDE};
		GMatrixDynSize<bf16, ROPE_DIM> gKr{gKr_ptr, seq_len, KR_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gV{gV_ptr, seq_len, V_STRIDE};
		GMatrixDynSize<bf16, V_DIM> gO{gOut_ptr, seq_len};

		// SMEM layout: KV preload region + Q
		u32 smem = 0;
		usize q_warp_idx = threadIdx.x / WARP_SIZE;
		usize tid = threadIdx.x % WARP_SIZE;
		SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		SMatrix<bf16, Q_PER_BLOCK, QK_DIM> sQ{sPreload._ptr + sPreload.bytes()};

		// Load Q from GMEM to SMEM (no commit — piggyback on first KV commit)
		usize q_block_idx = blockIdx.x;
		usize q_block_start = q_block_idx * Q_PER_BLOCK;
		usize q_start = q_block_start + q_warp_idx * Q_PER_WARP;
		GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gQ_block = tile_m<Q_PER_BLOCK>(gQ, q_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gQ_block, sQ);

		usize kv_steps = (q_block_start + Q_PER_BLOCK + KV_PER_STEP - 1) / KV_PER_STEP;
		kv_steps = std::min(kv_steps, seq_len / KV_PER_STEP);
		usize full_kv_steps = q_block_start / KV_PER_STEP;
		full_kv_steps = std::min(full_kv_steps, kv_steps);

		// Window: skip KV tiles entirely outside every warp's window.
		// Uses q_block_start (warp 0) because it has the earliest query position
		// and therefore needs the furthest-back keys. Later warps need fewer old
		// keys and their excess tiles are masked per-warp in the loop body.
		usize kv_first_step = 0;
		if constexpr (WINDOW_SIZE > 0) {
			if (q_block_start >= WINDOW_SIZE) {
				kv_first_step = (q_block_start - WINDOW_SIZE) / KV_PER_STEP;
			}
		}

		// Start preloading K and V from GMEM to SMEM (first commit also commits Q)
		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			cp_async_kv(gKc, gKr, gV, sPreload, kv_first_step + p, kv_steps);
			cp_async_commit();
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

		// Sink: a virtual token with no V contribution - it only adds to the
		// softmax denominator, stealing probability from real tokens.
		// sink[0] = raw score, sink[1] = output gate
		f32 sink_score = -INFINITY;
		f32 gate = 1.0f;
		if (sink != nullptr) {
			load_gmem_2x32b(sink, sink_score, gate);
		}
		f32 top_sink_scaled = math::max(sink_score * top_score_scale, std::numeric_limits<f32>::lowest());
		f32 bot_sink_scaled = math::max(sink_score * bot_score_scale, std::numeric_limits<f32>::lowest());

		SoftmaxStats r_top;
		r_top.max = top_sink_scaled + ONLINE_SOFTMAX_THRESHOLD;
		r_top.sum = math::fast::expb(-ONLINE_SOFTMAX_THRESHOLD); // TODO - constexpr_expb?
		SoftmaxStats r_bot;
		r_bot.max = bot_sink_scaled + ONLINE_SOFTMAX_THRESHOLD;
		r_bot.sum = r_top.sum;

		// O accumulator
		Fragment_16x16<f32> rO_f32[V_TILES];
		zero_(rO_f32);

		cp_async_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		// Load Q from SMEM to registers
		Fragment_16x16<bf16> rQ[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sQ, q_warp_idx * Q_PER_WARP, i * 16, rQ[i]);
		}
		// Load first KV tile from SMEM to registers
		SMatrix<bf16, KV_PER_STEP, PRELOAD_DIM> sKV;
		sKV = tile_m<KV_PER_STEP>(sPreload, kv_first_step % GMEM_PRELOAD);
		Fragment_16x16<bf16> rKV[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
		}

		// Sequential loop over KV
		X17_NO_UNROLL for (usize kv_step = kv_first_step; kv_step < kv_steps; ++kv_step) {
			// S = Q * K^T, interleaved with V load (rKV: K -> V)
			Fragment_16x16<f32> rS_f32;
			zero_(rS_f32);
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				mma_a_bt(rQ[i], rKV[i], rS_f32);
				smem_tile_to_fragment_trans(sKV, 0, V_SMEM_COL + i * 16, rKV[i]);
			}
			X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
				mma_a_bt(rQ[i], rKV[i], rS_f32);
			}

			// Scale scores must happen before masking to avoid -inf * 0 == NaN when score_scale == 0
			scale_top_(rS_f32, top_score_scale);
			scale_bottom_(rS_f32, bot_score_scale);

/*
			usize kv_pos = kv_step * KV_PER_STEP;

			// Window mask: mask keys outside the sliding window
			if constexpr (WINDOW_SIZE > 0) {
				if (kv_pos + WINDOW_SIZE <= q_start) {
					if (kv_pos + WINDOW_SIZE == q_start) {
						window_mask_diagonal(rS_f32);
					} else {
						fill_(rS_f32, -INFINITY);
					}
				}
			}

			// Causal mask: diagonal tile or full mask if past boundary
			if (kv_pos >= q_block_start) {
				if (kv_pos == q_start) {
					causal_mask_diagonal(rS_f32);
				} else {
					fill_(rS_f32, -INFINITY);
				}
			}
*/
			// Window mask: mask keys outside the sliding window
			if constexpr (WINDOW_SIZE > 0) {
				usize kv_pos = kv_step * KV_PER_STEP;
				if (kv_pos + WINDOW_SIZE == q_start) {
					window_mask_diagonal(rS_f32);
				} else if (kv_pos + WINDOW_SIZE < q_start) {
					fill_(rS_f32, -INFINITY);
				}
			}

			// Causal mask: diagonal tile or full mask if past boundary
			if (kv_step >= full_kv_steps) {
				usize kv_pos = kv_step * KV_PER_STEP;
				if (kv_pos == q_start) {
					causal_mask_diagonal(rS_f32);
				} else if (kv_pos > q_start) {
					fill_(rS_f32, -INFINITY);
				}
			}

			online_softmax(kv_step == kv_first_step, r_top, r_bot, rS_f32, rO_f32);
			Fragment_16x16<bf16> rP;
			cast(rS_f32, rP);

			{ // Get more data from GMEM
				// Wait for the next batch of GMEM -> SMEM preloads to complete
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sKV = tile_m<KV_PER_STEP>(sPreload, (kv_step + 1) % GMEM_PRELOAD);

				// Preload next KV tiles from GMEM
				cp_async_kv(gKc, gKr, gV, sPreload, kv_step + GMEM_PRELOAD, kv_steps);
				cp_async_commit();
			}

			// rO += S * V, interleaved with next K load
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				mma_a_bt(rP, rKV[i], rO_f32[i]);
				smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
			}
			// Load remaining K tiles that weren't covered by V interleaving
			X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
				smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
			}
		}

		GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = tile_m<Q_PER_BLOCK>(gO, q_block_idx);
		combine_and_store(rO_f32, r_top, r_bot, gate, q_start, q_warp_idx, gOut_block, gL_ptr);
	}
};

template<typename Attn_forward>
__global__ __launch_bounds__(Attn_forward::THREADS_PER_BLOCK) void
attn_forward(
	usize seq_len, bf16 *gQ_ptr,
	bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
	bf16 *gOut_ptr,
	f32 *gL_ptr,
	f32 *sink
) {
	auto attn_forward = Attn_forward();
	attn_forward.run(seq_len, gQ_ptr, gKc_ptr, gKr_ptr, gV_ptr, gOut_ptr, gL_ptr, sink);
}
