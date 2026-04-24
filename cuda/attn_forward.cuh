#pragma once

#include "utils.cuh"

#pragma nv_diag_suppress 186

// =============================================================================
// Fused FlashAttention-style forward kernel (SM80, bf16, tensor-core MMA).
//
// Computes causal sliding-window attention with:
//   - Attention sinks (StreamingLLM): token 0's key is always attended to;
//     the token is handled separately since it lives outside the sliding window.
//   - Scalable-Softmax (SSMax): per-query temperature = ln(e + n_tokens).
//   - Online softmax with lazy rescaling for numerical stability.
//   - A token does NOT attend to itself
//
// Grid: (ceil(seq_len / Q_PER_BLOCK), HEAD_GROUP_CNT)
// Block: WARPS_PER_BLOCK * 32 threads
//
// Memory pipeline is double-buffered by default (controlled by GMEM_PRELOAD).
// =============================================================================

// Template parameters:
//   _HEAD_CNT        – total number of attention heads
//   _HEADS_PER_KERNEL– heads processed together in one threadblock. The SMatrix class needs
//                      the number of columns to be multiple of 64. This multiplier is useful
//                      for tiny heads with QK_DIM < 64
//   _QK_DIM          – dimension of Q and K vectors per head
//   _V_DIM           – dimension of V vectors per head
//   _V_EQUALS_K      – when true, V is a prefix of K (V_DIM must be <= QK_DIM)
template<
	const usize _HEAD_CNT,
	const usize _HEADS_PER_KERNEL,
	const usize _QK_DIM,
	const usize _V_DIM,
	const usize _D_MODEL,
	const usize _QKV_FAN_IN,
	const bool _V_EQUALS_K = false
>
struct Attn_forward {
	// Expose template parameters needed by dependent kernels.
	static constexpr usize HEAD_CNT = _HEAD_CNT;
	static constexpr usize HEADS_PER_KERNEL = _HEADS_PER_KERNEL;
	static constexpr usize HEAD_GROUP_CNT = HEAD_CNT / HEADS_PER_KERNEL;
	static constexpr usize QK_DIM = _QK_DIM;
	static constexpr usize V_DIM = _V_DIM;
	static constexpr usize D_MODEL = _D_MODEL;
	static constexpr usize QKV_FAN_IN = _QKV_FAN_IN;
	static constexpr bool V_EQUALS_K = _V_EQUALS_K;
	static constexpr usize GMEM_PRELOAD = 2;

	// This is not the standard attention scaling which is `1.0 / sqrt(QK_DIM)`, because
	// our inputs are L2 normalized. To get unit variance of scores, we need to multiply
	// by the sqrt, not divide
	static constexpr f64 BASE_TEMPERATURE = math::constexpr_sqrt(QK_DIM);

	static constexpr f64 V_SCALE = math::constexpr_sqrt(f64(D_MODEL) / f64(QKV_FAN_IN));
	static constexpr f64 V_SCALE_FIX = 1.5;

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

	static constexpr usize Q_STRIDE = QK_DIM * HEAD_CNT;
	static constexpr usize K_STRIDE = QK_DIM * HEAD_CNT;
	static constexpr usize V_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize G_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize O_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize DO_STRIDE = V_DIM * HEAD_CNT;
	static constexpr usize DQ_STRIDE = QK_DIM * HEAD_CNT;
	static constexpr usize DK_STRIDE = QK_DIM * HEAD_CNT;
	static constexpr usize DV_STRIDE = V_DIM * HEAD_CNT;

	static_assert((QK_GROUP_DIM * sizeof(bf16)) % 128 == 0, "HEADS_PER_KERNEL must make grouped Q rows 128B aligned");
	static_assert((PRELOAD_DIM * sizeof(bf16)) % 128 == 0, "HEADS_PER_KERNEL must make grouped KV preload rows 128B aligned");

	static constexpr usize SMEM_BYTES =
		sizeof(bf16) * (
			KV_PER_STEP * PRELOAD_DIM * GMEM_PRELOAD
			+ Q_PER_BLOCK * QK_GROUP_DIM
		);

	// Mask the upper-triangular part of a 16x16 score tile (current key and future keys).
	// A token does NOT attend to itself so the diagonal is masked as well.
	//
	// MMA 16x16 fragment layout — each thread owns 4 elements:
	//   q rows: {tid/4, tid/4 + 8}          ("top" and "bot" halves)
	//   k cols: {2*(tid%4), 2*(tid%4)+1, 2*(tid%4)+8, 2*(tid%4)+9}
	//
	// Mapped to sub[qi][ki].val{0,1}:
	//   sub[0][0] = (q,   k  ), (q,   k+1)  — top-left 8x8
	//   sub[0][1] = (q,   k+8), (q,   k+9)  — top-right 8x8
	//   sub[1][0] = (q+8, k  ), (q+8, k+1)  — bot-left 8x8
	//   sub[1][1] = (q+8, k+8), (q+8, k+9)  — bot-right 8x8
	//
	// For causal masking (q_global <= k_global → mask):
	//   - top-right 8x8 is entirely masked (q < 8, k >= 8)
	//   - top-left and bot-right diagonals: element-wise comparison
	//   - bot-left 8x8 is entirely unmasked (q >= 8, k < 8)
	static X17_DEVICE void causal_mask_diagonal(Fragment_16x16<f32> &rS_f32) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize q = tid / 4;          // 0..7
		usize k = 2 * (tid % 4);    // 0,2,4,6
		constexpr f32 NEG_INF = -INFINITY;

		rS_f32.sub[0][1].val0 = NEG_INF;
		rS_f32.sub[0][1].val1 = NEG_INF;

		rS_f32.sub[0][0].val0 = k < q ? rS_f32.sub[0][0].val0 : NEG_INF;
		rS_f32.sub[1][1].val0 = k < q ? rS_f32.sub[1][1].val0 : NEG_INF;

		rS_f32.sub[0][0].val1 = k + 1 < q ? rS_f32.sub[0][0].val1 : NEG_INF;
		rS_f32.sub[1][1].val1 = k + 1 < q ? rS_f32.sub[1][1].val1 : NEG_INF;
	}

	/// This is the exact opposite of the causal mask
	static X17_DEVICE void window_mask_diagonal(Fragment_16x16<f32> &rS_f32) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize q = tid / 4;          // 0..7
		usize k = 2 * (tid % 4);    // 0,2,4,6
		constexpr f32 NEG_INF = -INFINITY;

		rS_f32.sub[1][0].val0 = NEG_INF;
		rS_f32.sub[1][0].val1 = NEG_INF;

		rS_f32.sub[0][0].val0 = k >= q ? rS_f32.sub[0][0].val0 : NEG_INF;
		rS_f32.sub[1][1].val0 = k >= q ? rS_f32.sub[1][1].val0 : NEG_INF;

		rS_f32.sub[0][0].val1 = k + 1 >= q ? rS_f32.sub[0][0].val1 : NEG_INF;
		rS_f32.sub[1][1].val1 = k + 1 >= q ? rS_f32.sub[1][1].val1 : NEG_INF;
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

	X17_DEVICE void load_sink_scores(
		f32 const *gSinkScore_ptr,
		usize seq_len,
		usize q_start,
		usize i_head_base,
		f32 (&top_sink_score)[HEADS_PER_KERNEL],
		f32 (&bot_sink_score)[HEADS_PER_KERNEL]
	) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize row_in_half = tid / 4;
		usize top_row = q_start + row_in_half;
		usize bot_row = top_row + 8;

		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; ++h) {
			// TODO - load with just one instruction, but wait until we get correctness right
			top_sink_score[h] = load_gmem_1x32b(gSinkScore_ptr + (i_head_base + h) * seq_len + top_row);
			bot_sink_score[h] = load_gmem_1x32b(gSinkScore_ptr + (i_head_base + h) * seq_len + bot_row);
		}
	}

	X17_DEVICE void load_sink_v(
		bf16 const *gSinkV_ptr,
		usize i_head_base,
		bf16 (&rSinkV)[QK_GROUP_DIM / 4]
	) {
		usize pair_in_quad = threadIdx.x % 4;
		bf16 const *sink_ptr = gSinkV_ptr + i_head_base * QK_DIM;
		constexpr usize LOAD_CNT = QK_GROUP_DIM / 8;
		X17_UNROLL for (usize i = 0; i < LOAD_CNT; ++i) {
			usize src_col = i * 8 + pair_in_quad * 2;
			f32 packed_f = load_gmem_1x32b(reinterpret_cast<f32 const *>(sink_ptr + src_col));
			u32 packed = __float_as_uint(packed_f);
			union {
				u32 packed;
				bf16 values[2];
			} sink_pair;
			sink_pair.packed = packed;
			rSinkV[2 * i + 0] = sink_pair.values[0];
			rSinkV[2 * i + 1] = sink_pair.values[1];
		}
	}

	// Lazy-rescale threshold for online softmax
	//
	// Standard online softmax rescales O and sum every time a new max appears.
	// That's expensive (touches all V_TILES of rO). Instead, we only rescale
	// when the new max exceeds the current max by more than this threshold.
	//
	// When rescaling happens, we also add the threshold to the new max to create some headroom.
	static constexpr f32 ONLINE_SOFTMAX_THRESHOLD = 5.0 / math::fast::logb_2;

	X17_DEVICE void online_softmax(
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

			scale_top_(rO_f32, top_rescale);
			scale_bottom_(rO_f32, bot_rescale);

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

	static X17_DEVICE void zig_zag_geglu(
		Fragment_8x8<f32> &out,
		Fragment_8x8<bf16> const &g
	) {
		f32 out0 = out.first();
		f32 out1 = out.second();
		f32 g0 = f32(V_SCALE) * f32(g.first());
		f32 g1 = f32(V_SCALE) * f32(g.second());
		out.set(
			math::fast::geglu(g0, out0),
			math::fast::geglu(out1, g1)
		);
	}

	static X17_DEVICE void zig_zag_geglu(
		Fragment_16x16<f32> &out,
		Fragment_16x16<bf16> const &g
	) {
		zig_zag_geglu(out.sub[0][0], g.sub[0][0]);
		zig_zag_geglu(out.sub[0][1], g.sub[0][1]);
		zig_zag_geglu(out.sub[1][0], g.sub[1][0]);
		zig_zag_geglu(out.sub[1][1], g.sub[1][1]);
	}

	X17_DEVICE void combine_and_store(
		Fragment_16x16<f32> (&rO_f32)[HEADS_PER_KERNEL][V_TILES],
		SoftmaxStats (&top_stats)[HEADS_PER_KERNEL],
		SoftmaxStats (&bot_stats)[HEADS_PER_KERNEL],
		usize q_start,
		usize q_warp_idx,
		SMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> sG,
		GMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> gOut_block,
		f32 *gL_ptr,
		usize seq_len,
		usize i_head_base
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");
		Fragment_16x16<bf16> rO[HEADS_PER_KERNEL * V_TILES];
		f32 top_L[HEADS_PER_KERNEL];
		f32 bot_L[HEADS_PER_KERNEL];
		usize tid = threadIdx.x % WARP_SIZE;

		cp_async_wait<0>();
		sync_threads();

		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			// Complete the row-wise sum reduction within each warp
			top_stats[h].sum += shuffle_xor_sync(top_stats[h].sum, 1);
			top_stats[h].sum += shuffle_xor_sync(top_stats[h].sum, 2);

			bot_stats[h].sum += shuffle_xor_sync(bot_stats[h].sum, 1);
			bot_stats[h].sum += shuffle_xor_sync(bot_stats[h].sum, 2);

			// Rescale, folding in normalization
			top_L[h] = math::fast::logb(top_stats[h].sum) + top_stats[h].max;
			bot_L[h] = math::fast::logb(bot_stats[h].sum) + bot_stats[h].max;

			f32 top_rescale = math::fast::divide(f32(V_SCALE * V_SCALE_FIX), top_stats[h].sum);
			f32 bot_rescale = math::fast::divide(f32(V_SCALE * V_SCALE_FIX), bot_stats[h].sum);

			scale_top_(rO_f32[h], top_rescale);
			scale_bottom_(rO_f32[h], bot_rescale);
			X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
				Fragment_16x16<bf16> rG;
				smem_tile_to_fragment(sG, q_warp_idx * Q_PER_WARP, h * V_DIM + i * 16, rG);
				zig_zag_geglu(rO_f32[h][i], rG);
				cast(rO_f32[h][i], rO[h * V_TILES + i]);
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

		store(rO, gOut_block, q_warp_idx * Q_PER_WARP, 0);
	}

	static X17_DEVICE void cp_async_kv(
		GMatrixDynSize<bf16, QK_GROUP_DIM> gK,
		GMatrixDynSize<bf16, V_GROUP_DIM> gV,
		SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
		usize p, usize kv_end
	) {
		if (p < kv_end) {
			SMatrix<bf16, KV_PER_STEP, PRELOAD_DIM> preload_tile = tile_m<KV_PER_STEP>(preload, p % GMEM_PRELOAD);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK, KV_PER_STEP, QK_GROUP_DIM>(
				threadIdx.x, tile_m<KV_PER_STEP>(gK, p), preload_tile, 0, 0, 0, 0
			);
			if constexpr (!V_EQUALS_K) {
				cp_async_gmem_to_smem<THREADS_PER_BLOCK, KV_PER_STEP, V_GROUP_DIM>(
					threadIdx.x, tile_m<KV_PER_STEP>(gV, p), preload_tile, 0, 0, 0, V_SMEM_COL
				);
			}
		}
	}

	static X17_DEVICE void cp_async_g(
		GMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> gG_block,
		SMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> sG
	) {
		cp_async_gmem_to_smem<THREADS_PER_BLOCK, Q_PER_BLOCK, V_GROUP_DIM>(
			threadIdx.x, gG_block, sG, 0, 0, 0, 0
		);
	}

	X17_DEVICE void run(
		usize seq_len, bf16 *gQ_ptr,
		bf16 *gK_ptr, bf16 *gV_ptr, bf16 *gG_ptr,
		bf16 const *gSinkV_ptr,
		f32 const *gSinkScore_ptr,
		bf16 *gOut_ptr,
		f32 *gL_ptr,
		usize window_size
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");
		usize i_head_group = blockIdx.y;
		usize i_head_base = i_head_group * HEADS_PER_KERNEL;

		// GMEM Matrices
		GMatrixDynSize<bf16, QK_GROUP_DIM> gQ{gQ_ptr + QK_DIM * i_head_base, seq_len, Q_STRIDE};
		GMatrixDynSize<bf16, QK_GROUP_DIM> gK{gK_ptr + QK_DIM * i_head_base, seq_len, K_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gV{gV_ptr + V_DIM * i_head_base, seq_len, V_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gG{gG_ptr + V_DIM * i_head_base, seq_len, G_STRIDE};
		GMatrixDynSize<bf16, V_GROUP_DIM> gO{gOut_ptr + V_DIM * i_head_base, seq_len, O_STRIDE};

		// SMEM layout: KV preload region + Q
		u32 smem = 0;
		usize q_warp_idx = threadIdx.x / WARP_SIZE;
		usize tid = threadIdx.x % WARP_SIZE;
		SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		SMatrix<bf16, Q_PER_BLOCK, QK_GROUP_DIM> sQ{sPreload._ptr + sPreload.bytes()};
		SMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> sG{sQ._ptr};

		// Load Q from GMEM to SMEM (committed with the first KV preload).
		usize q_block_idx = blockIdx.x;
		usize q_block_start = q_block_idx * Q_PER_BLOCK;
		usize q_block_end = q_block_start + Q_PER_BLOCK;
		usize q_start = q_block_start + q_warp_idx * Q_PER_WARP;
		GMatrix<bf16, Q_PER_BLOCK, QK_GROUP_DIM> gQ_block = tile_m<Q_PER_BLOCK>(gQ, q_block_idx);
		GMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> gG_block = tile_m<Q_PER_BLOCK>(gG, q_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK, Q_PER_BLOCK, QK_GROUP_DIM>(
			threadIdx.x, gQ_block, sQ, 0, 0, 0, 0
		);
		bf16 rSinkV[QK_GROUP_DIM / 4];
		load_sink_v(gSinkV_ptr, i_head_base, rSinkV);

		// Sink scores were precomputed during qkv_proj from the unrotated Q path.
		f32 top_sink_score[HEADS_PER_KERNEL];
		f32 bot_sink_score[HEADS_PER_KERNEL];
		load_sink_scores(gSinkScore_ptr, seq_len, q_start, i_head_base, top_sink_score, bot_sink_score);

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
			cp_async_kv(gK, gV, sPreload, kv_begin + p, kv_end);
			cp_async_commit();
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
		// Since we're multiplying and dividing by logb(e), it cancels out, so:
		//     temperature = BASE_TEMPERATURE * logb(n)
		u32 e_approx = 3;
		u32 top_n = std::min(window_size + 1 + e_approx, q_start + tid / 4 + (0 + 1 + e_approx));
		u32 bot_n = std::min(window_size + 1 + e_approx, q_start + tid / 4 + (8 + 1 + e_approx));
		f32 top_temperature = f32(BASE_TEMPERATURE) * math::fast::logb(f32(top_n));
		f32 bot_temperature = f32(BASE_TEMPERATURE) * math::fast::logb(f32(bot_n));

		cp_async_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		// Load Q from SMEM to registers
		Fragment_16x16<bf16> rQ[HEADS_PER_KERNEL][QK_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sQ, q_warp_idx * Q_PER_WARP, h * QK_DIM + i * 16, rQ[h][i]);
			}
		}
		// Load first KV tile from SMEM to registers
		// `rKV` holds K tiles during S = Q * K^T, then gets overwritten
		// with V tiles for O += P * V within the same loop iteration. The interleaved
		// MMA + SMEM load pattern hides the load latency.
		SMatrix<bf16, KV_PER_STEP, PRELOAD_DIM> sKV;
		sKV = tile_m<KV_PER_STEP>(sPreload, kv_begin % GMEM_PRELOAD);
		Fragment_16x16<bf16> rKV[HEADS_PER_KERNEL][QK_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
				smem_tile_to_fragment(sKV, 0, h * QK_DIM + i * 16, rKV[h][i]);
			}
		}

		// Initialize online softmax stats with the sink token's contribution.
		//
		// The sink token is not part of the KV loop, so we seed the stats:
		//   max = sink_score + THRESHOLD
		//   sum = expb(sink_score - max) = expb(-THRESHOLD)
		//
		// Why `+ THRESHOLD` in max? This "headroom" is used to reduce the number of rescales.
		//
		// Why `* 0.25` in sum? In the MMA fragment layout, 4 threads share each
		// Q row. Each thread independently accumulates a partial sum and combine_and_store()
		// sums all 4 partials. The sink contributes only once to the real sum,
		// so each thread's copy must be 1/4 of the value.
		f32 initial_scale = math::fast::constexpr_expb(-ONLINE_SOFTMAX_THRESHOLD);
		SoftmaxStats top_stats[HEADS_PER_KERNEL];
		SoftmaxStats bot_stats[HEADS_PER_KERNEL];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			top_sink_score[h] *= top_temperature;
			bot_sink_score[h] *= bot_temperature;

			f32 sum = initial_scale * 0.25;
			top_stats[h].max = top_sink_score[h] + ONLINE_SOFTMAX_THRESHOLD;
			top_stats[h].sum = sum;
			bot_stats[h].max = bot_sink_score[h] + ONLINE_SOFTMAX_THRESHOLD;
			bot_stats[h].sum = sum;
		}

		// O accumulator
		Fragment_16x16<f32> rO_f32[HEADS_PER_KERNEL][V_TILES];
		initial_scale = math::fast::constexpr_expb(-ONLINE_SOFTMAX_THRESHOLD) / V_SCALE;
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; ++h) {
			X17_UNROLL for (usize i = 0; i < V_TILES; ++i) {
				rO_f32[h][i].sub[0][0].val0 = initial_scale * f32(rSinkV[h * QK_DIM / 4 + i * 4 + 0]);
				rO_f32[h][i].sub[0][0].val1 = initial_scale * f32(rSinkV[h * QK_DIM / 4 + i * 4 + 1]);
				rO_f32[h][i].sub[0][1].val0 = initial_scale * f32(rSinkV[h * QK_DIM / 4 + i * 4 + 2]);
				rO_f32[h][i].sub[0][1].val1 = initial_scale * f32(rSinkV[h * QK_DIM / 4 + i * 4 + 3]);

				rO_f32[h][i].sub[1][0].val0 = rO_f32[h][i].sub[0][0].val0;
				rO_f32[h][i].sub[1][0].val1 = rO_f32[h][i].sub[0][0].val1;
				rO_f32[h][i].sub[1][1].val0 = rO_f32[h][i].sub[0][1].val0;
				rO_f32[h][i].sub[1][1].val1 = rO_f32[h][i].sub[0][1].val1;
			}
		}
		bool g_prefetched = false;

		// Sequential loop over KV
		X17_NO_UNROLL for (usize kv_step = kv_begin; kv_step < kv_end; ++kv_step) {
			// S = Q * K^T, interleaved with V load (rKV: K -> V)
			Fragment_16x16<f32> rS_f32[HEADS_PER_KERNEL];
			zero_(rS_f32);
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
					mma_a_bt(rQ[h][i], rKV[h][i], rS_f32[h]);
					smem_tile_to_fragment_trans(sKV, 0, V_SMEM_COL + h * V_DIM + i * 16, rKV[h][i]);
				}
				X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
					mma_a_bt(rQ[h][i], rKV[h][i], rS_f32[h]);
				}

				// Scaling must happen before masking to avoid -inf * 0 == NaN when scale == 0
				scale_top_(rS_f32[h], top_temperature);
				scale_bottom_(rS_f32[h], bot_temperature);
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

			Fragment_16x16<bf16> rP[HEADS_PER_KERNEL];
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				online_softmax(top_stats[h], bot_stats[h], rS_f32[h], rO_f32[h]);
				cast(rS_f32[h], rP[h]);
			}

			{ // Get more data from GMEM
				// Wait for the next batch of GMEM -> SMEM preloads to complete
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();
				sKV = tile_m<KV_PER_STEP>(sPreload, (kv_step + 1) % GMEM_PRELOAD);

				// Preload next KV tiles from GMEM. Once the KV pipeline reaches its end,
				// reuse the now-dead Q staging area to start pulling in the per-row G tile
				// for the post-attention fused op.
				usize next_kv = kv_step + GMEM_PRELOAD;
				if (next_kv < kv_end) {
					cp_async_kv(gK, gV, sPreload, next_kv, kv_end);
				} else if (!g_prefetched && next_kv == kv_end) {
					cp_async_g(gG_block, sG);
					g_prefetched = true;
				}
				cp_async_commit();
			}

			// rO += S * V, interleaved with next K load
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
					mma_a_bt(rP[h], rKV[h][i], rO_f32[h][i]);
					smem_tile_to_fragment(sKV, 0, h * QK_DIM + i * 16, rKV[h][i]);
				}
				X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
					smem_tile_to_fragment(sKV, 0, h * QK_DIM + i * 16, rKV[h][i]);
				}
			}
		}

		if (!g_prefetched) {
			cp_async_g(gG_block, sG);
			cp_async_commit();
		}

		GMatrix<bf16, Q_PER_BLOCK, V_GROUP_DIM> gOut_block = tile_m<Q_PER_BLOCK>(gO, q_block_idx);
		combine_and_store(
			rO_f32,
			top_stats,
			bot_stats,
			q_start,
			q_warp_idx,
			sG,
			gOut_block,
			gL_ptr,
			seq_len,
			i_head_base
		);
	}
};

template<typename Attn_forward>
__global__ __launch_bounds__(Attn_forward::THREADS_PER_BLOCK) void
attn_forward(
	usize seq_len, bf16 *gQ_ptr,
	bf16 *gK_ptr, bf16 *gV_ptr, bf16 *gG_ptr,
	bf16 const *gSinkV_ptr,
	f32 const *gSinkScore_ptr,
	bf16 *gOut_ptr,
	f32 *gL_ptr,
	usize window_size
) {
	Attn_forward attn_forward = Attn_forward();
	attn_forward.run(seq_len, gQ_ptr, gK_ptr, gV_ptr, gG_ptr, gSinkV_ptr, gSinkScore_ptr, gOut_ptr, gL_ptr, window_size);
}
