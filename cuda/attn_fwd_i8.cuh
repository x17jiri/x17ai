#pragma once

#include "utils.cuh"
#include "utils_b8.cuh"

using b8::FixedI8;

#pragma nv_diag_suppress 186

// =============================================================================
// Fused FlashAttention-style forward kernel (SM80, bf16, tensor-core MMA).
//
// Computes causal sliding-window attention with:
//   - Attention sinks: The sink token is handled separately since it lives outside the sliding window.
//   - Scalable-Softmax (SSMax): per-query temperature = ln(e + n_tokens).
//   - Online softmax with lazy rescaling for numerical stability.
//   - A token does NOT attend to itself
//
// Grid: (seq_len / Q_PER_BLOCK, HEAD_GROUP_CNT)
// Block: WARPS_PER_BLOCK * 32 threads
//
// Memory pipeline is double-buffered by default (controlled by GMEM_PRELOAD).
// =============================================================================

struct SoftmaxStats {
	f32 sum;
	f32 max;
};

template<
	const usize _N_HEADS,
	const usize _HEADS_PER_KERNEL,
	const usize _HEAD_DIM,
	const usize _MODEL_DIM,
	const f64 V_SCALE_FIX,
	const usize Q_STRIDE,
	const usize KV_STRIDE,
	const usize O_STRIDE
>
struct AttnForward {
	// Expose template parameters needed by dependent kernels.
	static constexpr usize N_HEADS = _N_HEADS;
	static constexpr usize HEADS_PER_KERNEL = _HEADS_PER_KERNEL;
	static constexpr usize HEAD_DIM = _HEAD_DIM;
	static constexpr usize MODEL_DIM = _MODEL_DIM;

	static constexpr usize HEAD_GROUP_CNT = N_HEADS / HEADS_PER_KERNEL;
	static constexpr usize GMEM_PRELOAD = 2;

	static constexpr f64 BASE_TEMPERATURE = math::constexpr_inv_sqrt(HEAD_DIM);

	static_assert(HEADS_PER_KERNEL > 0, "HEADS_PER_KERNEL must be > 0");
	static_assert(N_HEADS % HEADS_PER_KERNEL == 0, "N_HEADS must be divisible by HEADS_PER_KERNEL");
	static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be divisible by 32");
	static_assert(
		(HEADS_PER_KERNEL * HEAD_DIM * sizeof(FixedI8)) % 128 == 0,
		"HEADS_PER_KERNEL must make grouped i8 rows 128B aligned"
	);

	static constexpr usize HEAD_TILES = HEAD_DIM / 32;
	static constexpr usize HEAD_GROUP_DIM = HEADS_PER_KERNEL * HEAD_DIM;
	static constexpr usize PRELOAD_DIM = 2 * HEAD_GROUP_DIM;

	static constexpr usize Q_WARPS = 4;
	static constexpr usize KV_WARPS = 1;
	static constexpr usize Q_PER_WARP = 16;
	static constexpr usize Q_PER_BLOCK = Q_PER_WARP * Q_WARPS;
	static constexpr usize KV_PER_WARP = 16;
	static constexpr usize KV_PER_STEP = KV_PER_WARP * KV_WARPS;

	static constexpr usize WARPS_PER_BLOCK = Q_WARPS * KV_WARPS;
	static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

	static constexpr usize OWNED_ROWS = Q_PER_WARP / 8;

	static constexpr usize SMEM_BYTES =
		KV_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
		+ Q_PER_BLOCK * HEAD_GROUP_DIM;

	// Mask the upper-triangular part of a 16x16 score tile (current key and future keys).
	// A token does NOT attend to itself so the diagonal is masked as well.
	static X17_DEVICE void causal_mask_diagonal(b32::Fragment_16x16<f32> &rS_f32) {
		usize tid = threadIdx.x % WARP_SIZE;

		usize q1 = tid / 4; // 0..7
		usize q2 = q1 + 8;  // 8..15

		usize k1 = 4 * (tid % 4); // 0,4,8,12
		usize k2 = k1 + 1;        // 1,5,9,13
		usize k3 = k1 + 2;        // 2,6,10,14
		usize k4 = k1 + 3;        // 3,7,11,15

		constexpr f32 NEG_INF = -INFINITY;

		rS_f32.v8x16[0].h8x8[0].val0 = k1 < q1 ? rS_f32.v8x16[0].h8x8[0].val0 : NEG_INF;
		rS_f32.v8x16[0].h8x8[0].val1 = k2 < q1 ? rS_f32.v8x16[0].h8x8[0].val1 : NEG_INF;
		rS_f32.v8x16[0].h8x8[1].val0 = k3 < q1 ? rS_f32.v8x16[0].h8x8[1].val0 : NEG_INF;
		rS_f32.v8x16[0].h8x8[1].val1 = k4 < q1 ? rS_f32.v8x16[0].h8x8[1].val1 : NEG_INF;

		rS_f32.v8x16[1].h8x8[0].val0 = k1 < q2 ? rS_f32.v8x16[1].h8x8[0].val0 : NEG_INF;
		rS_f32.v8x16[1].h8x8[0].val1 = k2 < q2 ? rS_f32.v8x16[1].h8x8[0].val1 : NEG_INF;
		rS_f32.v8x16[1].h8x8[1].val0 = k3 < q2 ? rS_f32.v8x16[1].h8x8[1].val0 : NEG_INF;
		rS_f32.v8x16[1].h8x8[1].val1 = k4 < q2 ? rS_f32.v8x16[1].h8x8[1].val1 : NEG_INF;
	}

	/// This is the exact opposite of the causal mask
	static X17_DEVICE void window_mask_diagonal(b32::Fragment_16x16<f32> &rS_f32) {
		usize tid = threadIdx.x % WARP_SIZE;

		usize q1 = tid / 4; // 0..7
		usize q2 = q1 + 8;  // 8..15

		usize k1 = 4 * (tid % 4); // 0,4,8,12
		usize k2 = k1 + 1;        // 1,5,9,13
		usize k3 = k1 + 2;        // 2,6,10,14
		usize k4 = k1 + 3;        // 3,7,11,15

		constexpr f32 NEG_INF = -INFINITY;

		rS_f32.v8x16[0].h8x8[0].val0 = k1 >= q1 ? rS_f32.v8x16[0].h8x8[0].val0 : NEG_INF;
		rS_f32.v8x16[0].h8x8[0].val1 = k2 >= q1 ? rS_f32.v8x16[0].h8x8[0].val1 : NEG_INF;
		rS_f32.v8x16[0].h8x8[1].val0 = k3 >= q1 ? rS_f32.v8x16[0].h8x8[1].val0 : NEG_INF;
		rS_f32.v8x16[0].h8x8[1].val1 = k4 >= q1 ? rS_f32.v8x16[0].h8x8[1].val1 : NEG_INF;

		rS_f32.v8x16[1].h8x8[0].val0 = k1 >= q2 ? rS_f32.v8x16[1].h8x8[0].val0 : NEG_INF;
		rS_f32.v8x16[1].h8x8[0].val1 = k2 >= q2 ? rS_f32.v8x16[1].h8x8[0].val1 : NEG_INF;
		rS_f32.v8x16[1].h8x8[1].val0 = k3 >= q2 ? rS_f32.v8x16[1].h8x8[1].val0 : NEG_INF;
		rS_f32.v8x16[1].h8x8[1].val1 = k4 >= q2 ? rS_f32.v8x16[1].h8x8[1].val1 : NEG_INF;
	}

	static constexpr size_t mma_count(size_t seq_len, size_t window_size) {
		seq_len /= 16;
		window_size = std::min(seq_len, window_size > 0 ? window_size / 16 : seq_len);
		usize masked = seq_len - window_size;
		// Count equivalent 16x16x16 MMA ops so flops() can keep using 2 * 16^3.
		// Per visible 16x16 score tile:
		//   - Q * K^T uses HEAD_TILES i8 MMAs with k = 32 => 2 * HEAD_TILES equivalents
		//   - P * V uses 2 bf16 MMAs per 16x32 V tile => 2 * HEAD_TILES equivalents
		constexpr size_t MMA_EQUIVS_PER_SCORE_TILE = 4 * HEAD_TILES;
		return (
			seq_len * seq_len * MMA_EQUIVS_PER_SCORE_TILE
			- masked * masked * MMA_EQUIVS_PER_SCORE_TILE
		) / 2;
	}

	static constexpr double flops(size_t seq_len, size_t window_size) {
		return double(mma_count(seq_len, window_size)) * 2.0 * 16.0 * 16.0 * 16.0;
	}

	X17_DEVICE void calculate_sink_scores(
		b8::Fragment_16x32<FixedI8> const (&rQ)[HEADS_PER_KERNEL][HEAD_TILES],
		u32 const (&rSinkK)[HEAD_GROUP_DIM / 16],
		i32 (&sink_score)[HEADS_PER_KERNEL][OWNED_ROWS]
	) {
		static_assert(OWNED_ROWS == 2);
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; ++h) {
			X17_UNROLL for (usize j = 0; j < OWNED_ROWS; ++j) {
				i32 acc = 0;
				X17_UNROLL for (usize i = 0; i < HEAD_TILES; ++i) {
					b8::Fragment_16x32<FixedI8> const &q = rQ[h][i];
					usize sink_idx = h * (HEAD_DIM / 16) + i * 2;

					u32 sink_left = rSinkK[sink_idx + 0];
					acc = __dp4a(
						static_cast<i32>(q.h16x16[0].v8x16[j].val),
						static_cast<i32>(sink_left),
						acc
					);

					u32 sink_right = rSinkK[sink_idx + 1];
					acc = __dp4a(
						static_cast<i32>(q.h16x16[1].v8x16[j].val),
						static_cast<i32>(sink_right),
						acc
					);
				}

				acc += shuffle_xor_sync(acc, 1);
				acc += shuffle_xor_sync(acc, 2);

				sink_score[h][j] = acc;
			}
		}
	}

	template<typename T>
	X17_DEVICE void logicalize_score_columns(b32::Fragment_16x16<T> &frag) {
		X17_UNROLL for (usize row = 0; row < 2; ++row) {
			T tmp = frag.v8x16[row].h8x8[0].val1;
			frag.v8x16[row].h8x8[0].val1 = frag.v8x16[row].h8x8[1].val0;
			frag.v8x16[row].h8x8[1].val0 = tmp;
		}
	}

	X17_DEVICE void load_sink_kv(
		FixedI8 const *gSinkKV_ptr,
		usize i_head_base,
		u32 (&rSinkKV)[HEAD_GROUP_DIM / 4 / 4]
	) {
		usize group_lane = threadIdx.x % 4;
		FixedI8 const *sink_ptr = gSinkKV_ptr + i_head_base * HEAD_DIM;
		constexpr usize LOAD_CNT = HEAD_GROUP_DIM / 16;
		X17_UNROLL for (usize i = 0; i < LOAD_CNT; ++i) {
			usize src_col = i * 16 + group_lane * 4;
			rSinkKV[i] = *reinterpret_cast<u32 const *>(sink_ptr + src_col);
		}
	}

	X17_DEVICE void online_softmax(
		SoftmaxStats (&stats)[OWNED_ROWS],
		b32::Fragment_16x16<f32> &rS_f32,
		b32::Fragment_16x32<f32> (&rO_f32)[HEAD_TILES]
	) {
		static_assert(OWNED_ROWS == 2);
		SoftmaxStats &top = stats[0];
		SoftmaxStats &bot = stats[1];

		// The `max` is for the entire owned row.
		// The `sum` is just the elements owned by the current thread.
		// Complete sum is calculated in combine_and_store().

		// Step 1: `max` of the owned values
		f32 new_top_max = math::max(
			math::max(rS_f32.v8x16[0].h8x8[0].val0, rS_f32.v8x16[0].h8x8[0].val1),
			math::max(rS_f32.v8x16[0].h8x8[1].val0, rS_f32.v8x16[0].h8x8[1].val1)
		);

		f32 new_bot_max = math::max(
			math::max(rS_f32.v8x16[1].h8x8[0].val0, rS_f32.v8x16[1].h8x8[0].val1),
			math::max(rS_f32.v8x16[1].h8x8[1].val0, rS_f32.v8x16[1].h8x8[1].val1)
		);

		// Step 2: Rescale outputs if needed
		f32 top_rescale = 1.0f;
		f32 bot_rescale = 1.0f;
		bool top_needs_rescale = new_top_max > top.max || new_bot_max > bot.max;
		if (any_sync(top_needs_rescale)) {
			new_top_max = math::max(new_top_max, top.max);
			new_top_max = math::max(new_top_max, shuffle_xor_sync(new_top_max, 1));
			new_top_max = math::max(new_top_max, shuffle_xor_sync(new_top_max, 2));

			new_bot_max = math::max(new_bot_max, bot.max);
			new_bot_max = math::max(new_bot_max, shuffle_xor_sync(new_bot_max, 1));
			new_bot_max = math::max(new_bot_max, shuffle_xor_sync(new_bot_max, 2));

			top_rescale = math::fast::expb(top.max - new_top_max);
			bot_rescale = math::fast::expb(bot.max - new_bot_max);
			X17_UNROLL for (usize i = 0; i < HEAD_TILES; ++i) {
				X17_UNROLL for (usize j = 0; j < 2; ++j) {
					scale_(rO_f32[i].h16x16[j].v8x16[0], top_rescale);
					scale_(rO_f32[i].h16x16[j].v8x16[1], bot_rescale);
				}
			}

			top.max = new_top_max;
			bot.max = new_bot_max;
		}

		// Step 3: Replace scores with expb(score - max)
		rS_f32.v8x16[0].h8x8[0].val0 = math::fast::expb(rS_f32.v8x16[0].h8x8[0].val0 - top.max);
		rS_f32.v8x16[0].h8x8[0].val1 = math::fast::expb(rS_f32.v8x16[0].h8x8[0].val1 - top.max);
		rS_f32.v8x16[0].h8x8[1].val0 = math::fast::expb(rS_f32.v8x16[0].h8x8[1].val0 - top.max);
		rS_f32.v8x16[0].h8x8[1].val1 = math::fast::expb(rS_f32.v8x16[0].h8x8[1].val1 - top.max);

		rS_f32.v8x16[1].h8x8[0].val0 = math::fast::expb(rS_f32.v8x16[1].h8x8[0].val0 - bot.max);
		rS_f32.v8x16[1].h8x8[0].val1 = math::fast::expb(rS_f32.v8x16[1].h8x8[0].val1 - bot.max);
		rS_f32.v8x16[1].h8x8[1].val0 = math::fast::expb(rS_f32.v8x16[1].h8x8[1].val0 - bot.max);
		rS_f32.v8x16[1].h8x8[1].val1 = math::fast::expb(rS_f32.v8x16[1].h8x8[1].val1 - bot.max);

		// Step 4: `sum` of the owned values
		f32 top_add = (
			(rS_f32.v8x16[0].h8x8[0].val0 + rS_f32.v8x16[0].h8x8[0].val1)
			+ (rS_f32.v8x16[0].h8x8[1].val0 + rS_f32.v8x16[0].h8x8[1].val1)
		);
		top.sum = math::fma(top.sum, top_rescale, top_add);

		f32 bot_add = (
			(rS_f32.v8x16[1].h8x8[0].val0 + rS_f32.v8x16[1].h8x8[0].val1)
			+ (rS_f32.v8x16[1].h8x8[1].val0 + rS_f32.v8x16[1].h8x8[1].val1)
		);
		bot.sum = math::fma(bot.sum, bot_rescale, bot_add);
	}

	X17_DEVICE void combine_and_store(
		b32::Fragment_16x32<f32> (&rO_f32)[HEADS_PER_KERNEL][HEAD_TILES],
		SoftmaxStats (&stats)[HEADS_PER_KERNEL][OWNED_ROWS],
		usize q_start,
		usize q_warp_idx,
		GMatrix<FixedI8, Q_PER_BLOCK, HEAD_GROUP_DIM> gOut_block,
		f32 *gL_ptr,
		usize seq_len,
		usize i_head_base
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");
		static_assert(OWNED_ROWS == 2);
		b8::Fragment_16x32<FixedI8> rO[HEADS_PER_KERNEL * HEAD_TILES];
		f32 top_L[HEADS_PER_KERNEL];
		f32 bot_L[HEADS_PER_KERNEL];
		usize tid = threadIdx.x % WARP_SIZE;

		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			SoftmaxStats &top_stats = stats[h][0];
			SoftmaxStats &bot_stats = stats[h][1];

			// Complete the row-wise sum reduction within each warp
			top_stats.sum += shuffle_xor_sync(top_stats.sum, 1);
			top_stats.sum += shuffle_xor_sync(top_stats.sum, 2);

			bot_stats.sum += shuffle_xor_sync(bot_stats.sum, 1);
			bot_stats.sum += shuffle_xor_sync(bot_stats.sum, 2);

			// Rescale, folding in normalization
			top_L[h] = math::fast::logb(top_stats.sum) + top_stats.max;
			bot_L[h] = math::fast::logb(bot_stats.sum) + bot_stats.max;

			f32 top_rescale = math::fast::recip(top_stats.sum) * (1.0f / 255.0f);
			f32 bot_rescale = math::fast::recip(bot_stats.sum) * (1.0f / 255.0f);

			X17_UNROLL for (usize i = 0; i < HEAD_TILES; ++i) {
				X17_UNROLL for (usize j = 0; j < 2; ++j) {
					b32::scale_(rO_f32[h][i].h16x16[j].v8x16[0], top_rescale);
					b32::scale_(rO_f32[h][i].h16x16[j].v8x16[1], bot_rescale);
				}
				cast<false>(rO_f32[h][i], rO[h * HEAD_TILES + i]);
			}
		}

		// Store log-sum-exp (L) values to GMEM for the backward pass.
		// Each Q row is owned by 4 threads. We split the work as follows:
		//    - tid % 4 == 0: write L for "top"
		//    - tid % 4 == 2: write L for "bot"
		//    - otherwise: don't write anything
		if (gL_ptr != nullptr && (tid & 1) == 0) {
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				f32 *gL_head_ptr = gL_ptr + seq_len * (i_head_base + h);
				gL_head_ptr[q_start + (tid / 4) + ((tid & 2) * 4)] =
					((tid & 2) == 0 ? top_L[h] : bot_L[h]);
			}
		}

		b8::store(rO, gOut_block, q_warp_idx * Q_PER_WARP, 0);
	}

	static X17_DEVICE void async_load_kv(
		GMatrixDynSize<FixedI8, 2 * HEAD_GROUP_DIM> gKV,
		b8::SMatrixEvenOdd<FixedI8, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
		usize p, usize kv_end
	) {
		if (p < kv_end) {
			auto preload_tile = tile_m<KV_PER_STEP>(preload, p % GMEM_PRELOAD);
			async_load<THREADS_PER_BLOCK, KV_PER_STEP, PRELOAD_DIM>(
				threadIdx.x,
				gKV, p * KV_PER_STEP, 0,
				preload_tile, 0, 0
			);
		}
	}

	X17_DEVICE void run(
		usize seq_len, FixedI8 *gQ_ptr, FixedI8 *gKV_ptr,
		FixedI8 const *gSinkK_ptr,
		FixedI8 const *gSinkV_ptr,
		i32 const *gMax_ptr,
		FixedI8 *gOut_ptr,
		f32 *gL_ptr,
		usize window_size
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");
		usize i_head_group = blockIdx.y;
		usize i_head_base = i_head_group * HEADS_PER_KERNEL;

		// GMEM Matrices
		GMatrixDynSize<FixedI8, HEAD_GROUP_DIM> gQ{gQ_ptr + HEAD_DIM * i_head_base, Q_STRIDE};
		GMatrixDynSize<FixedI8, 2*HEAD_GROUP_DIM> gKV{gKV_ptr + 2*HEAD_DIM * i_head_base, KV_STRIDE};
		GMatrixDynSize<FixedI8, HEAD_GROUP_DIM> gO{gOut_ptr + HEAD_DIM * i_head_base, O_STRIDE};

		// SMEM layout: KV preload region + Q
		u32 smem = 0;
		usize q_warp_idx = threadIdx.x / WARP_SIZE;
		usize tid = threadIdx.x % WARP_SIZE;
		b8::SMatrixEvenOdd<FixedI8, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		b8::SMatrix<FixedI8, Q_PER_BLOCK, HEAD_GROUP_DIM> sQ{sPreload._ptr + sPreload.bytes()};

		// Load Q from GMEM to SMEM (committed with the first KV preload).
		usize q_block_idx = blockIdx.x;
		usize q_block_start = q_block_idx * Q_PER_BLOCK;
		usize q_block_end = q_block_start + Q_PER_BLOCK;
		usize q_start = q_block_start + q_warp_idx * Q_PER_WARP;
		async_load<THREADS_PER_BLOCK, Q_PER_BLOCK, HEAD_GROUP_DIM>(
			threadIdx.x,
			gQ, q_block_start, 0,
			sQ, 0, 0
		);
		u32 rSinkK[HEAD_GROUP_DIM / 4 / 4];
		load_sink_kv(gSinkK_ptr, i_head_base, rSinkK);
		u32 rSinkV[HEAD_GROUP_DIM / 4 / 4];
		load_sink_kv(gSinkV_ptr, i_head_base, rSinkV);

		// round window_size up without overflow (window_size == 0 means disabled)
		usize max_window_size = std::numeric_limits<usize>::max();
		window_size = window_size > 0 ? window_size : max_window_size;
		usize window_steps = std::min((window_size - 1) / KV_PER_STEP + 1, max_window_size / KV_PER_STEP);
		window_size = window_steps * KV_PER_STEP;

		// Sliding-window + causal iteration boundaries (in KV_PER_STEP units).
		// The Q block attends to a trapezoidal region of KV:
		//
		//      |--- window edge ---|--- fully unmasked ---|--- causal edge ---|
		//   kv_begin         kv_begin_full           kv_end_full            kv_end
		usize kv_begin = (q_block_start - std::min(q_block_start, window_size)) / KV_PER_STEP;
		usize kv_begin_full = (q_block_end - std::min(q_block_end, window_size)) / KV_PER_STEP;
		usize kv_end_full = q_block_start / KV_PER_STEP;
		usize kv_end = std::min(seq_len, q_block_end + KV_PER_STEP - 1) / KV_PER_STEP;

		// Start preloading K and V from GMEM to SMEM (first commit also commits Q)
		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			async_load_kv(gKV, sPreload, kv_begin + p, kv_end);
			async_load_commit();
		}

		// Temperature is composed of these factors:
		//     - BASE_TEMPERATURE: used to fix the variance of the dot products
		//     - logb(e): used so we can replace `exp` with `expb` in softmax
		//     - ssmax = ln(n) = logb(n) / logb(e): Scalable Softmax
		//         - Where `n = number of attended tokens + e_approx`
		//         - Each Q attends to:
		//             - at most `window_size` previous tokens
		//             - sink token
		//             - NOT self
		//         - `e_approx` is integer approximation of `e` and is used to ensure `ssmax >= 1`
		//     - 1.0 / FIXED_I8_SCALE^2: because both inputs to the dot product
		//       are scaled up by FIXED_I8_SCALE
		// Since we're multiplying and dividing by logb(e), it cancels out, so:
		//     temperature = BASE_TEMPERATURE * logb(n) / FIXED_I8_SCALE^2
		f32 row_temperature[OWNED_ROWS];
		X17_UNROLL for (usize row = 0; row < OWNED_ROWS; ++row) {
			constexpr u32 e_approx = 3;
			constexpr f64 FIXED_I8_SCALE_2 = f64(b8::FIXED_I8_SCALE) * f64(b8::FIXED_I8_SCALE);
			u32 n = std::min(window_size + 1 + e_approx, q_start + tid / 4 + (8*row + 1 + e_approx));
			row_temperature[row] = f32(BASE_TEMPERATURE / FIXED_I8_SCALE_2) * math::fast::logb(f32(n));
		}

		async_load_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		// Load Q from SMEM to registers in the native i8 MMA layout.
		b8::Fragment_16x32<FixedI8> rQ[HEADS_PER_KERNEL][HEAD_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < HEAD_TILES; i++) {
				load_tile(sQ, q_warp_idx * Q_PER_WARP, h * HEAD_DIM + i * 32, rQ[h][i]);
			}
		}
		// Load first KV tile from SMEM to registers
		// `rKV` holds K tiles during S = Q * K^T, then gets overwritten
		// with V tiles for O += P * V within the same loop iteration. The interleaved
		// MMA + SMEM load pattern hides the load latency.
		b8::SMatrixEvenOdd<FixedI8, KV_PER_STEP, PRELOAD_DIM> sKV =
			tile_m<KV_PER_STEP>(sPreload, kv_begin % GMEM_PRELOAD);
		b8::Fragment_16x32<FixedI8> rKV[HEADS_PER_KERNEL][HEAD_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < HEAD_TILES; i++) {
				load_tile(sKV, 0, 2 * h * HEAD_DIM + i * 32, rKV[h][i]);
			}
		}

		// Initialize online softmax stats with the sink token's contribution.
		//
		// The sink token is not part of the KV loop, so we seed the stats:
		//   max = sink_score
		//   sum = expb(sink_score - max) = expb(0) = 1.0
		//
		// Why `* 0.25` in sum? In the MMA fragment layout, 4 threads share each
		// Q row. Each thread independently accumulates a partial sum and combine_and_store()
		// sums all 4 partials. The sink contributes only once to the real sum,
		// so each thread's copy must be 1/4 of the value.
		f32 sink_score[HEADS_PER_KERNEL][OWNED_ROWS];
		i32 sink_score_i32[HEADS_PER_KERNEL][OWNED_ROWS];
		calculate_sink_scores(rQ, rSinkK, sink_score_i32);

		SoftmaxStats stats[HEADS_PER_KERNEL][OWNED_ROWS];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize row = 0; row < OWNED_ROWS; ++row) {
				sink_score[h][row] = f32(sink_score_i32[h][row]) * row_temperature[row];
				stats[h][row].max = sink_score[h][row];
				stats[h][row].sum = 0.25f;
			}
		}

		// O accumulator
		b32::Fragment_16x32<f32> rO_f32[HEADS_PER_KERNEL][HEAD_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; ++h) {
			X17_UNROLL for (usize i = 0; i < HEAD_TILES; ++i) {
				u32 packed0 = rSinkV[(h * HEAD_TILES + i) * 2 + 0];
				u32 packed1 = rSinkV[(h * HEAD_TILES + i) * 2 + 1];
				b8::Fragment_16x32<FixedI8> sink_v_i8;
				X17_UNROLL for (usize j = 0; j < OWNED_ROWS; ++j) {
					sink_v_i8.h16x16[0].v8x16[j].val = packed0;
					sink_v_i8.h16x16[1].v8x16[j].val = packed1;
				}
				cast<false>(sink_v_i8, rO_f32[h][i]);
				X17_UNROLL for (usize j = 0; j < 2; ++j) {
					scale_(rO_f32[h][i].h16x16[j].v8x16[0], 255.0f);
					scale_(rO_f32[h][i].h16x16[j].v8x16[1], 255.0f);
				}
			}
		}

		// Sequential loop over KV
		X17_NO_UNROLL for (usize kv_step = kv_begin; kv_step < kv_end; ++kv_step) {
			// S = Q * K^T, interleaved with V load (rKV: K -> V)
			b32::Fragment_16x16<f32> rS_f32[HEADS_PER_KERNEL];
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				b32::Fragment_16x16<i32> rS_i32;
				zero_(rS_i32);
				X17_UNROLL for (usize i = 0; i < HEAD_TILES; i++) {
					mma_a_bt(rQ[h][i], rKV[h][i], rS_i32);
					load_tile_pretrans(sKV, 0, ((2 * h + 1) * HEAD_TILES + i) * 32, rKV[h][i]);
				}
				logicalize_score_columns(rS_i32);
				cast(rS_i32, rS_f32[h]);

				// Scaling must happen before masking to avoid -inf * 0 == NaN when scale == 0.
				X17_UNROLL for (usize row = 0; row < OWNED_ROWS; ++row) {
					scale_(rS_f32[h].v8x16[row], row_temperature[row]);
				}
			}

			// Apply masks
			if (kv_step < kv_begin_full || kv_step >= kv_end_full) {
				// Window mask: mask keys outside the sliding window
				if (kv_step < kv_begin_full) {
					usize diag_warp = Q_WARPS + kv_step - kv_begin_full;
					if (q_warp_idx == diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							window_mask_diagonal(rS_f32[h]);
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
							causal_mask_diagonal(rS_f32[h]);
						}
					} else if (q_warp_idx < diag_warp) {
						X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
							fill_(rS_f32[h], -INFINITY);
						}
					}
				}
			}

			b8::Fragment_16x16<u8> rP[HEADS_PER_KERNEL];
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				online_softmax(stats[h], rS_f32[h], rO_f32[h]);

				union {
					u8 split[4];
					u32 packed;
				} t;

				// TODO - round
				t.split[0] = u8(__float2int_rn(rS_f32[h].v8x16[0].h8x8[0].val0 * 255.0));
				t.split[1] = u8(__float2int_rn(rS_f32[h].v8x16[0].h8x8[0].val1 * 255.0));
				t.split[2] = u8(__float2int_rn(rS_f32[h].v8x16[0].h8x8[1].val0 * 255.0));
				t.split[3] = u8(__float2int_rn(rS_f32[h].v8x16[0].h8x8[1].val1 * 255.0));

				rP[h].v8x16[0].val = t.packed;

				t.split[0] = u8(__float2int_rn(rS_f32[h].v8x16[1].h8x8[0].val0 * 255.0));
				t.split[1] = u8(__float2int_rn(rS_f32[h].v8x16[1].h8x8[0].val1 * 255.0));
				t.split[2] = u8(__float2int_rn(rS_f32[h].v8x16[1].h8x8[1].val0 * 255.0));
				t.split[3] = u8(__float2int_rn(rS_f32[h].v8x16[1].h8x8[1].val1 * 255.0));

				rP[h].v8x16[1].val = t.packed;
			}

			{ // Get more data from GMEM
				// Wait for the next batch of GMEM -> SMEM preloads to complete
				async_load_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sKV = tile_m<KV_PER_STEP>(sPreload, (kv_step + 1) % GMEM_PRELOAD);

				// Preload next KV tiles from GMEM
				usize next_kv = kv_step + GMEM_PRELOAD;
				async_load_kv(gKV, sPreload, next_kv, kv_end);
				async_load_commit();
			}

			// rO += P * V, interleaved with next K load
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				X17_UNROLL for (usize i = 0; i < HEAD_TILES; i++) {
					X17_UNROLL for (usize j = 0; j < 2; ++j) {
						rKV[h][i].h16x16[j].finish_trans_load_();

						b32::Fragment_16x16<i32> t;
						zero_(t);
						mma_a_bt(rP[h], rKV[h][i].h16x16[j], t);

						b32::Fragment_16x16<f32> &o = rO_f32[h][i].h16x16[j];
						X17_UNROLL for (usize jj = 0; jj < 2; ++jj) {
							o.v8x16[jj].h8x8[0].val0 += t.v8x16[jj].h8x8[0].val0;
							o.v8x16[jj].h8x8[0].val1 += t.v8x16[jj].h8x8[1].val0;
							o.v8x16[jj].h8x8[1].val0 += t.v8x16[jj].h8x8[0].val1;
							o.v8x16[jj].h8x8[1].val1 += t.v8x16[jj].h8x8[1].val1;
						}
					}
					load_tile(sKV, 0, ((2 * h) * HEAD_TILES + i) * 32, rKV[h][i]);
				}
			}
		}

		GMatrix<FixedI8, Q_PER_BLOCK, HEAD_GROUP_DIM> gOut_block = gO.template tile_m<Q_PER_BLOCK>(q_block_idx);
		combine_and_store(
			rO_f32,
			stats,
			q_start,
			q_warp_idx,
			gOut_block,
			gL_ptr,
			seq_len,
			i_head_base
		);
	}
};

template<typename AttnForward>
__global__ __launch_bounds__(AttnForward::THREADS_PER_BLOCK) void
attn_forward(
	usize seq_len, FixedI8 *gQ_ptr,
	FixedI8 *gKV_ptr,
	FixedI8 const *gSinkK_ptr,
	FixedI8 const *gSinkV_ptr,
	i32 const *gMax_ptr,
	FixedI8 *gOut_ptr,
	f32 *gL_ptr,
	usize window_size
) {
	AttnForward attn_forward = AttnForward();
	attn_forward.run(seq_len, gQ_ptr, gKV_ptr, gSinkK_ptr, gSinkV_ptr, gMax_ptr, gOut_ptr, gL_ptr, window_size);
}
