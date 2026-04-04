#pragma once

#include "utils2.cuh"

#pragma nv_diag_suppress 186

template<
	const usize _HEAD_CNT,
	const usize _NONROPE_DIM,
	const usize _ROPE_DIM,
	const usize _V_DIM,
	const bool _V_EQUALS_K = false,
	const usize _GMEM_PRELOAD = 2
>
struct Attn_forward {
	// Expose template parameters needed by dependent kernels.
	static constexpr usize HEAD_CNT = _HEAD_CNT;
	static constexpr usize QK_DIM = _NONROPE_DIM + _ROPE_DIM;
	static constexpr usize V_DIM = _V_DIM;
	static constexpr bool V_EQUALS_K = _V_EQUALS_K;
	static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

	static_assert(V_DIM <= QK_DIM, "V_DIM must be <= QK_DIM");

	static constexpr usize QK_TILES = QK_DIM / 16;
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
	static constexpr f32 SCORE_SCALE = 1.0 / constexpr_sqrt(f64(QK_DIM));

	// TODO - other matrices (dO, O, ...) should have their stride
	static constexpr usize K_STRIDE = QK_DIM * HEAD_CNT;
	static constexpr usize V_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize Q_STRIDE = QK_DIM * HEAD_CNT;

	static constexpr usize SMEM_BYTES =
		sizeof(bf16) * (
			KV_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
			+ Q_PER_BLOCK * QK_DIM
		);

	static X17_DEVICE void causal_mask_diagonal(Fragment_16x16<f32> &rS_f32) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize q = tid / 4;          // 0..7
		usize k = 2 * (tid % 4);    // 0,2,4,6
		constexpr f32 NEG_INF = -INFINITY;

		rS_f32.sub[0][1].val0 = NEG_INF;
		rS_f32.sub[0][1].val1 = NEG_INF;

		rS_f32.sub[0][0].val0 = k <= q ? rS_f32.sub[0][0].val0 : NEG_INF;
		rS_f32.sub[1][1].val0 = k <= q ? rS_f32.sub[1][1].val0 : NEG_INF;

		rS_f32.sub[0][0].val1 = k + 1 <= q ? rS_f32.sub[0][0].val1 : NEG_INF;
		rS_f32.sub[1][1].val1 = k + 1 <= q ? rS_f32.sub[1][1].val1 : NEG_INF;
	}

	/// This is the exact opposite of the causal mask
	static X17_DEVICE void window_mask_diagonal(Fragment_16x16<f32> &rS_f32) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize q = tid / 4;          // 0..7
		usize k = 2 * (tid % 4);    // 0,2,4,6
		constexpr f32 NEG_INF = -INFINITY;

		rS_f32.sub[1][0].val0 = NEG_INF;
		rS_f32.sub[1][0].val1 = NEG_INF;

		rS_f32.sub[0][0].val0 = k > q ? rS_f32.sub[0][0].val0 : NEG_INF;
		rS_f32.sub[1][1].val0 = k > q ? rS_f32.sub[1][1].val0 : NEG_INF;

		rS_f32.sub[0][0].val1 = k + 1 > q ? rS_f32.sub[0][0].val1 : NEG_INF;
		rS_f32.sub[1][1].val1 = k + 1 > q ? rS_f32.sub[1][1].val1 : NEG_INF;
	}

	static constexpr size_t mma_count(size_t seq_len, size_t window_size) {
		seq_len /= 16;
		window_size = std::min(seq_len, window_size > 0 ? window_size / 16 : seq_len);
		usize masked = seq_len - window_size;
		return (
			seq_len * seq_len * (QK_TILES + V_TILES)
			- masked * masked * (QK_TILES + V_TILES)
		) / 2;
	}

	static constexpr double flops(size_t seq_len, size_t window_size) {
		return double(mma_count(seq_len, window_size)) * 2.0 * 16.0 * 16.0 * 16.0;
	}

	static constexpr f32 ONLINE_SOFTMAX_THRESHOLD = 5.0 / math::fast::logb_2;

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


			if (!first_step) {
				top_rescale = math::fast::expb(top.max - new_top_max);
				scale_top_(rO_f32, top_rescale);

				bot_rescale = math::fast::expb(bot.max - new_bot_max);
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
		f32 top_sink_scaled,
		f32 bot_sink_scaled,
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

		// !! SINK SUM MUST BE ADDED HERE, NOT BEFORE THE LOOP !!
		// The loop accumulates partial sums across 4 threads per row.
		// shuffle_xor above reduces them into one total. The sink is a
		// single scalar — adding it before the loop would count it 4×.
		// We must add it exactly once, after the reduction.
		top.sum += math::fast::expb(top_sink_scaled - top.max);
		bot.sum += math::fast::expb(bot_sink_scaled - bot.max);

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
		GMatrixDynSize<bf16, QK_DIM> gK,
		GMatrixDynSize<bf16, V_DIM> gV,
		SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
		usize p, usize kv_end
	) {
		if (p < kv_end) {
			auto preload_tile = tile_m<KV_PER_STEP>(preload, p % GMEM_PRELOAD);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				threadIdx.x,
				tile_m<KV_PER_STEP>(gK, p),
				preload_tile, 0, 0
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
		bf16 *gK_ptr, bf16 *gV_ptr,
		bf16 *gOut_ptr,
		f32 *gL_ptr,
		f32 *sink,
		usize window_size
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");

		// GMEM Matrices
		GMatrixDynSize<bf16, QK_DIM> gQ{gQ_ptr, seq_len, Q_STRIDE};
		GMatrixDynSize<bf16, QK_DIM> gK{gK_ptr, seq_len, K_STRIDE};
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
		usize q_block_end = q_block_start + Q_PER_BLOCK;
		usize q_start = q_block_start + q_warp_idx * Q_PER_WARP;
		GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gQ_block = tile_m<Q_PER_BLOCK>(gQ, q_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gQ_block, sQ);

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
			cp_async_kv(gK, gV, sPreload, kv_begin + p, kv_end);
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
		r_top.sum = 0.0f; // sink sum added in combine_and_store (not here — would be counted 4×)
		SoftmaxStats r_bot;
		r_bot.max = bot_sink_scaled + ONLINE_SOFTMAX_THRESHOLD;
		r_bot.sum = 0.0f;

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
		sKV = tile_m<KV_PER_STEP>(sPreload, kv_begin % GMEM_PRELOAD);
		Fragment_16x16<bf16> rKV[QK_TILES];
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
		}

		// Sequential loop over KV
		X17_NO_UNROLL for (usize kv_step = kv_begin; kv_step < kv_end; ++kv_step) {
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

			// Scaling must happen before masking to avoid -inf * 0 == NaN when scale == 0
			scale_top_(rS_f32, top_score_scale);
			scale_bottom_(rS_f32, bot_score_scale);

			// Apply masks
			if (kv_step < kv_begin_full || kv_step >= kv_end_full) {
				// Window mask: mask keys outside the sliding window
				if (kv_step < kv_begin_full) {
					usize diag_warp = Q_WARPS + kv_step - kv_begin_full;
					if (q_warp_idx == diag_warp) {
						window_mask_diagonal(rS_f32);
					} else if (q_warp_idx > diag_warp) {
						fill_(rS_f32, -INFINITY);
					}
				}
				// Causal mask: mask future keys
				if (kv_step >= kv_end_full) {
					usize diag_warp = kv_step - kv_end_full;
					if (q_warp_idx == diag_warp) {
						causal_mask_diagonal(rS_f32);
					} else if (q_warp_idx < diag_warp) {
						fill_(rS_f32, -INFINITY);
					}
				}
			}

			online_softmax(kv_step == kv_begin, r_top, r_bot, rS_f32, rO_f32);
			Fragment_16x16<bf16> rP;
			cast(rS_f32, rP);

			{ // Get more data from GMEM
				// Wait for the next batch of GMEM -> SMEM preloads to complete
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sKV = tile_m<KV_PER_STEP>(sPreload, (kv_step + 1) % GMEM_PRELOAD);

				// Preload next KV tiles from GMEM
				cp_async_kv(gK, gV, sPreload, kv_step + GMEM_PRELOAD, kv_end);
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
		combine_and_store(rO_f32, r_top, r_bot, top_sink_scaled, bot_sink_scaled, gate, q_start, q_warp_idx, gOut_block, gL_ptr);
	}
};

template<typename Attn_forward>
__global__ __launch_bounds__(Attn_forward::THREADS_PER_BLOCK) void
attn_forward(
	usize seq_len, bf16 *gQ_ptr,
	bf16 *gK_ptr, bf16 *gV_ptr,
	bf16 *gOut_ptr,
	f32 *gL_ptr,
	f32 *sink,
	usize window_size
) {
	auto attn_forward = Attn_forward();
	attn_forward.run(seq_len, gQ_ptr, gK_ptr, gV_ptr, gOut_ptr, gL_ptr, sink, window_size);
}
