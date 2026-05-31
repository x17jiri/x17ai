#pragma once

#include "utils.cuh"
#include "utils_b8.cuh"

using b8::FixedI8;

#pragma nv_diag_suppress 186

template<bool SHUFFLE = true>
X17_DEVICE void cast(b8::Fragment_16x16<FixedI8> const &src, b32::Fragment_16x16<f32> &dst) {
	union Packed4 {
		u32 tuple4;
		u16 tuple2[2];
		FixedI8 val[4];
	};
	Packed4 top, bot, left, right;

	top.tuple4 = src.v8x16[0].val;
	bot.tuple4 = src.v8x16[1].val;

	left.tuple2[0] = top.tuple2[0];
	left.tuple2[1] = bot.tuple2[0];
	right.tuple2[0] = top.tuple2[1];
	right.tuple2[1] = bot.tuple2[1];

	if constexpr (SHUFFLE) {
		Packed4 l, r;
		usize tid = threadIdx.x;
		left.tuple4 = shuffle_xor_sync(left.tuple4, 1);

		l.tuple4 = (tid & 1) == 0 ? right.tuple4 : left.tuple4;
		r.tuple4 = (tid & 1) == 0 ? left.tuple4 : right.tuple4;

		l.tuple4 = shuffle_xor_sync(l.tuple4, 3);

		left.tuple4 = (tid & 2) == 0 ? r.tuple4 : l.tuple4;
		right.tuple4 = (tid & 2) == 0 ? l.tuple4 : r.tuple4;

		left.tuple4 = shuffle_xor_sync(left.tuple4, 2);
	}

	f32 top0 = left.val[0];
	f32 top1 = left.val[1];

	f32 top8 = right.val[0];
	f32 top9 = right.val[1];

	f32 bot0 = left.val[2];
	f32 bot1 = left.val[3];

	f32 bot8 = right.val[2];
	f32 bot9 = right.val[3];

	dst.v8x16[0].h8x8[0].val0 = top0;
	dst.v8x16[0].h8x8[0].val1 = top1;

	dst.v8x16[0].h8x8[1].val0 = top8;
	dst.v8x16[0].h8x8[1].val1 = top9;

	dst.v8x16[1].h8x8[0].val0 = bot0;
	dst.v8x16[1].h8x8[0].val1 = bot1;

	dst.v8x16[1].h8x8[1].val0 = bot8;
	dst.v8x16[1].h8x8[1].val1 = bot9;
}

X17_DEVICE void cast(b32::Fragment_16x16<f32> const &src, Fragment_16x16<bf16> &dst) {
	X17_UNROLL for (usize row = 0; row < 2; ++row) {
		X17_UNROLL for (usize col = 0; col < 2; ++col) {
			dst.sub[row][col].set(
				round_cast<bf16>(src.v8x16[row].h8x8[col].val0),
				round_cast<bf16>(src.v8x16[row].h8x8[col].val1)
			);
		}
	}
}

template<bool SHUFFLE = true>
X17_DEVICE void cast(b8::Fragment_16x16<FixedI8> const &src, Fragment_16x16<bf16> &dst) {
	b32::Fragment_16x16<f32> tmp;
	cast<SHUFFLE>(src, tmp);
	cast(tmp, dst);
}

template<bool SHUFFLE = true>
X17_DEVICE void cast(b32::Fragment_16x16<f32> const &src, b8::Fragment_16x16<FixedI8> &dst) {
	union {
		u32 tuple4;
		struct {
			FixedI8 top0, top1, bot0, bot1;
		} packed;
	} left, right;

	left.packed.top0 = b8::to_fixedi8(src.v8x16[0].h8x8[0].val0);
	left.packed.top1 = b8::to_fixedi8(src.v8x16[0].h8x8[0].val1);
	left.packed.bot0 = b8::to_fixedi8(src.v8x16[1].h8x8[0].val0);
	left.packed.bot1 = b8::to_fixedi8(src.v8x16[1].h8x8[0].val1);

	right.packed.top0 = b8::to_fixedi8(src.v8x16[0].h8x8[1].val0);
	right.packed.top1 = b8::to_fixedi8(src.v8x16[0].h8x8[1].val1);
	right.packed.bot0 = b8::to_fixedi8(src.v8x16[1].h8x8[1].val0);
	right.packed.bot1 = b8::to_fixedi8(src.v8x16[1].h8x8[1].val1);

	if constexpr (SHUFFLE) {
		usize tid = threadIdx.x;
		u32 l = left.tuple4;
		u32 r = right.tuple4;

		l = shuffle_xor_sync(l, 2);

		u32 left_or_right = (tid & 2) == 0 ? r : l;
		u32 right_or_left = (tid & 2) == 0 ? l : r;

		left_or_right = shuffle_xor_sync(left_or_right, 3);

		u32 top = (tid & 1) == 0 ? right_or_left : left_or_right;
		u32 bot = (tid & 1) == 0 ? left_or_right : right_or_left;

		top = shuffle_xor_sync(top, 1);

		dst.v8x16[0].val = __byte_perm(top, bot, 0x5410);
		dst.v8x16[1].val = __byte_perm(top, bot, 0x7632);
	} else {
		union {
			u32 tuple4;
			u16 tuple2[2];
		} top, bot;

		top.tuple2[0] = left.tuple4 & 0xFFFFu;
		top.tuple2[1] = right.tuple4 & 0xFFFFu;
		bot.tuple2[0] = (left.tuple4 >> 16) & 0xFFFFu;
		bot.tuple2[1] = (right.tuple4 >> 16) & 0xFFFFu;

		dst.v8x16[0].val = top.tuple4;
		dst.v8x16[1].val = bot.tuple4;
	}
}

template<bool SHUFFLE = true>
X17_DEVICE void cast(b8::Fragment_16x32<FixedI8> const &src, b32::Fragment_16x32<f32> &dst) {
	X17_UNROLL for (usize i = 0; i < 2; ++i) {
		cast<SHUFFLE>(src.h16x16[i], dst.h16x16[i]);
	}
}

template<bool SHUFFLE = true>
X17_DEVICE void cast(b32::Fragment_16x32<f32> const &src, b8::Fragment_16x32<FixedI8> &dst) {
	X17_UNROLL for (usize i = 0; i < 2; ++i) {
		cast<SHUFFLE>(src.h16x16[i], dst.h16x16[i]);
	}
}

template<bool SHUFFLE = true>
X17_DEVICE void cast(b8::Fragment_32x32<FixedI8> const &src, b32::Fragment_32x32<f32> &dst) {
	X17_UNROLL for (usize j = 0; j < 2; ++j) {
		cast<SHUFFLE>(src.v16x32[j], dst.v16x32[j]);
	}
}

template<bool SHUFFLE = true>
X17_DEVICE void cast(b32::Fragment_32x32<f32> const &src, b8::Fragment_32x32<FixedI8> &dst) {
	X17_UNROLL for (usize j = 0; j < 2; ++j) {
		cast<SHUFFLE>(src.v16x32[j], dst.v16x32[j]);
	}
}

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
// Grid: (seq_len / Q_PER_BLOCK, HEAD_GROUP_CNT)
// Block: WARPS_PER_BLOCK * 32 threads
//
// Memory pipeline is double-buffered by default (controlled by GMEM_PRELOAD).
// =============================================================================

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
	static constexpr usize HEAD_GROUP_CNT = HEAD_CNT / HEADS_PER_KERNEL;
	static constexpr usize HEAD_DIM = _HEAD_DIM;
	static constexpr usize MODEL_DIM = _MODEL_DIM;
	static constexpr usize GMEM_PRELOAD = 2;

	static constexpr f64 BASE_TEMPERATURE = math::constexpr_inv_sqrt(HEAD_DIM);

	static_assert(HEADS_PER_KERNEL > 0, "HEADS_PER_KERNEL must be > 0");
	static_assert(HEAD_CNT % HEADS_PER_KERNEL == 0, "HEAD_CNT must be divisible by HEADS_PER_KERNEL");
	static_assert(VG_DIM <= QK_DIM, "VG_DIM must be <= QK_DIM");

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
			seq_len * seq_len * (QK_TILES + VG_TILES)
			- masked * masked * (QK_TILES + VG_TILES)
		) / 2;
	}

	static constexpr double flops(size_t seq_len, size_t window_size) {
		return double(mma_count(seq_len, window_size)) * 2.0 * 16.0 * 16.0 * 16.0;
	}

	X17_DEVICE void calculate_sink_scores(
		b8::Fragment_16x32<FixedI8> const (&rQ)[HEADS_PER_KERNEL][HEAD_TILES],
		u32 const (&rSinkK)[HEAD_GROUP_DIM / 16],
		i32 (&sink_score)[OWNED_ROWS][HEADS_PER_KERNEL]
	) {
		static_assert(OWNED_ROWS == 2);
		X17_UNROLL for (usize j = 0; j < OWNED_ROWS; ++j) {
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; ++h) {
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

				sink_score[j][h] = acc;
			}
		}
	}

/*	X17_DEVICE void load_max_scores(
		f32 const *gMax_ptr,
		usize seq_len,
		usize q_start,
		usize i_head_base,
		f32 (&top_max)[HEADS_PER_KERNEL],
		f32 (&bot_max)[HEADS_PER_KERNEL]
	) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize row_in_half = tid / 4;
		usize top_row = q_start + row_in_half;
		usize bot_row = top_row + 8;

		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; ++h) {
			top_max[h] = load_gmem_1x32b(gMax_ptr + (i_head_base + h) * seq_len + top_row);
			bot_max[h] = load_gmem_1x32b(gMax_ptr + (i_head_base + h) * seq_len + bot_row);
		}
	}*/

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
			rSinkKV[i] = load_gmem_1x32b(reinterpret_cast<u32 const *>(sink_ptr + src_col));
		}
	}

	// Lazy-rescale threshold for online softmax
	//
	// Standard online softmax rescales O and sum every time a new max appears.
	// That's expensive (touches all VG_TILES of rO). Instead, we only rescale
	// when the new max exceeds the current max by more than this threshold.
	//
	// When rescaling happens, we also add the threshold to the new max to create some headroom.
	static constexpr f32 ONLINE_SOFTMAX_THRESHOLD = 5.0 / math::fast::logb_2;

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

			X17_UNROLL for (usize i = 0; i < HEAD_TILES; ++i) {
				X17_UNROLL for (usize j = 0; j < 2; ++j) {
					b32::scale_(rO_f32[i].h16x16[j].v8x16[0], top_rescale);
					b32::scale_(rO_f32[i].h16x16[j].v8x16[1], bot_rescale);
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

			f32 top_rescale = math::fast::recip(top_stats.sum);
			f32 bot_rescale = math::fast::recip(bot_stats.sum);

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

	static X17_DEVICE void cp_async_kv(
		GMatrixDynSize<FixedI8, 2 * HEAD_GROUP_DIM> gKV,
		b8::SMatrix<FixedI8, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
		usize p, usize kv_end
	) {
		if (p < kv_end) {
			auto preload_tile = tile_m<KV_PER_STEP>(preload, p % GMEM_PRELOAD);
			preload_tile.template cp_async_from<THREADS_PER_BLOCK, KV_PER_STEP, PRELOAD_DIM>(
				threadIdx.x, gKV, p * KV_PER_STEP, 0, 0, 0
			);
		}
	}

	X17_DEVICE void run(
		usize seq_len, FixedI8 *gQ_ptr, FixedI8 *gKV_ptr,
		FixedI8 const *gSinkK_ptr,
		FixedI8 const *gSinkV_ptr,
		f32 const *gMax_ptr,
		FixedI8 *gOut_ptr,
		f32 *gL_ptr,
		usize window_size
	) {
		static_assert(KV_WARPS == 1, "current algorithm doesn't reduce over KV warps");
		usize i_head_group = blockIdx.y;
		usize i_head_base = i_head_group * HEADS_PER_KERNEL;

		// GMEM Matrices
		GMatrixDynSize<FixedI8, HEAD_GROUP_DIM> gQ{gQ_ptr + HEAD_DIM * i_head_base, seq_len, Q_STRIDE};
		GMatrixDynSize<FixedI8, 2*HEAD_GROUP_DIM> gKV{gKV_ptr + 2*HEAD_DIM * i_head_base, seq_len, KV_STRIDE};
		GMatrixDynSize<FixedI8, HEAD_GROUP_DIM> gO{gOut_ptr + HEAD_DIM * i_head_base, seq_len, O_STRIDE};

		// SMEM layout: KV preload region + Q
		u32 smem = 0;
		usize q_warp_idx = threadIdx.x / WARP_SIZE;
		usize tid = threadIdx.x % WARP_SIZE;
		b8::SMatrix<FixedI8, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> sPreload{smem};
		b8::SMatrix<FixedI8, Q_PER_BLOCK, HEAD_GROUP_DIM> sQ{sPreload._ptr + sPreload.bytes()};

		// Load Q from GMEM to SMEM (committed with the first KV preload).
		usize q_block_idx = blockIdx.x;
		usize q_block_start = q_block_idx * Q_PER_BLOCK;
		usize q_block_end = q_block_start + Q_PER_BLOCK;
		usize q_start = q_block_start + q_warp_idx * Q_PER_WARP;
		GMatrix<FixedI8, Q_PER_BLOCK, HEAD_GROUP_DIM> gQ_block = tile_m<Q_PER_BLOCK>(gQ, q_block_idx);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK, Q_PER_BLOCK, QK_GROUP_DIM>(
			threadIdx.x, gQ_block, sQ, 0, 0, 0, 0
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
			cp_async_kv(gKV, sPreload, kv_begin + p, kv_end);
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
		//     - 1.0 / FIXED_I8_SCALE^2: because both inputs to the dot product
		//       are scaled up by FIXED_I8_SCALE
		// Since we're multiplying and dividing by logb(e), it cancels out, so:
		//     temperature = BASE_TEMPERATURE * logb(n) / FIXED_I8_SCALE^2
		u32 e_approx = 3;
		constexpr f64 FIXED_I8_SCALE_2 = f64(b8::FIXED_I8_SCALE) * f64(b8::FIXED_I8_SCALE);
		f32 row_temperature[OWNED_ROWS];
		X17_UNROLL for (usize row = 0; row < OWNED_ROWS; ++row) {
			u32 n = std::min(window_size + 1 + e_approx, q_start + tid / 4 + (8*row + 1 + e_approx));
			row_temperature[row] = f32(BASE_TEMPERATURE / FIXED_I8_SCALE_2) * math::fast::logb(f32(n));
		}

		cp_async_wait<GMEM_PRELOAD - 1>();
		sync_threads();

		// Load Q from SMEM to registers in the native i8 MMA layout.
		b8::Fragment_16x32<FixedI8> rQ[HEADS_PER_KERNEL][HEAD_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < HEAD_TILES; i++) {
				sQ.tile_to_fragment(q_warp_idx * Q_PER_WARP, h * HEAD_DIM + i * 32, rQ[h][i]);
			}
		}
		// Load first KV tile from SMEM to registers
		// `rKV` holds K tiles during S = Q * K^T, then gets overwritten
		// with V tiles for O += P * V within the same loop iteration. The interleaved
		// MMA + SMEM load pattern hides the load latency.
		b8::SMatrix<FixedI8, KV_PER_STEP, PRELOAD_DIM> sKV;
		sKV = tile_m<KV_PER_STEP>(sPreload, kv_begin % GMEM_PRELOAD);
		b8::Fragment_16x32<FixedI8> rKV[HEADS_PER_KERNEL][HEAD_TILES];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			X17_UNROLL for (usize i = 0; i < HEAD_TILES; i++) {
				sKV.tile_to_fragment(0, 2 * h * HEAD_DIM + i * 32, rKV[h][i]);
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
		f32 sink_score[OWNED_ROWS][HEADS_PER_KERNEL];
		i32 sink_score_i32[OWNED_ROWS][HEADS_PER_KERNEL];
		calculate_sink_scores(rQ, rSinkK, sink_score_i32);

		f32 initial_scale = math::fast::constexpr_expb(-ONLINE_SOFTMAX_THRESHOLD);
		SoftmaxStats stats[HEADS_PER_KERNEL][OWNED_ROWS];
		X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
			f32 sum = initial_scale * 0.25;
			X17_UNROLL for (usize row = 0; row < OWNED_ROWS; ++row) {
				sink_score[row][h] = f32(sink_score_i32[row][h]) * row_temperature[row];
				stats[h][row].max = sink_score[row][h] + ONLINE_SOFTMAX_THRESHOLD;
				stats[h][row].sum = sum;
			}
		}

		// O accumulator
		b32::Fragment_16x32<f32> rO_f32[HEADS_PER_KERNEL][HEAD_TILES];
		initial_scale = math::fast::constexpr_expb(-ONLINE_SOFTMAX_THRESHOLD);
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
				b32::scale_(rO_f32[h][i], initial_scale);
			}
		}

		/*if (gMax_ptr != nullptr) {
			f32 top_max[HEADS_PER_KERNEL];
			f32 bot_max[HEADS_PER_KERNEL];
			load_max_scores(gMax_ptr, seq_len, q_start, i_head_base, top_max, bot_max);

			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; ++h) {
				f32 top_seed = math::fast::expb(top_sink_score[h] - top_max[h]);
				f32 bot_seed = math::fast::expb(bot_sink_score[h] - bot_max[h]);

				top_stats[h].max = top_max[h];
				top_stats[h].sum = top_seed * 0.25f;
				bot_stats[h].max = bot_max[h];
				bot_stats[h].sum = bot_seed * 0.25f;

				X17_UNROLL for (usize i = 0; i < VG_TILES; ++i) {
					f32 sink0 = f32(rSinkV[h * QK_DIM / 4 + i * 4 + 0]) / SPARSE_SCALE;
					f32 sink1 = f32(rSinkV[h * QK_DIM / 4 + i * 4 + 1]) / SPARSE_SCALE;
					f32 sink2 = f32(rSinkV[h * QK_DIM / 4 + i * 4 + 2]) / SPARSE_SCALE;
					f32 sink3 = f32(rSinkV[h * QK_DIM / 4 + i * 4 + 3]) / SPARSE_SCALE;

					rO_f32[h][i].sub[0][0].val0 = top_seed * sink0;
					rO_f32[h][i].sub[0][0].val1 = top_seed * sink1;
					rO_f32[h][i].sub[0][1].val0 = top_seed * sink2;
					rO_f32[h][i].sub[0][1].val1 = top_seed * sink3;

					rO_f32[h][i].sub[1][0].val0 = bot_seed * sink0;
					rO_f32[h][i].sub[1][0].val1 = bot_seed * sink1;
					rO_f32[h][i].sub[1][1].val0 = bot_seed * sink2;
					rO_f32[h][i].sub[1][1].val1 = bot_seed * sink3;
				}
			}
		}*/

		// Sequential loop over KV
		X17_NO_UNROLL for (usize kv_step = kv_begin; kv_step < kv_end; ++kv_step) {
			// S = Q * K^T, interleaved with V load (rKV: K -> V)
			b32::Fragment_16x16<f32> rS_f32[HEADS_PER_KERNEL];
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				b32::Fragment_16x16<i32> rS_i32;
				zero_(rS_i32);
				X17_UNROLL for (usize i = 0; i < HEAD_TILES; i++) {
					mma_a_bt(rQ[h][i], rKV[h][i], rS_i32);
					sKV.tile_to_fragment(0, ((2 * h + 1) * HEAD_TILES + i) * 32, rKV[h][i]);
				}
				b32::cast(rS_i32, rS_f32[h]);

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

			Fragment_16x16<bf16> rP[HEADS_PER_KERNEL];
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				online_softmax(stats[h], rS_f32[h], rO_f32[h]);
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
				cp_async_kv(gKV, sPreload, next_kv, kv_end);
				cp_async_commit();
			}


			// rO += P * V, interleaved with next K load
			X17_UNROLL for (usize h = 0; h < HEADS_PER_KERNEL; h++) {
				X17_UNROLL for (usize i = 0; i < HEAD_TILES; i++) {
					X17_UNROLL for (usize j = 0; j < 2; ++j) {
						Fragment_16x16<bf16> rV;
						cast<false>(rKV[h][i].h16x16[j], rV);
						rV.transpose_();
						mma_a_bt(rP[h], rV, rO_f32[h][i].h16x16[j]);
					}
					sKV.tile_to_fragment(0, ((2 * h) * HEAD_TILES + i) * 32, rKV[h][i]);
				}
			}
		}

		GMatrix<FixedI8, Q_PER_BLOCK, HEAD_GROUP_DIM> gOut_block = tile_m<Q_PER_BLOCK>(gO, q_block_idx);
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
	f32 const *gMax_ptr,
	FixedI8 *gOut_ptr,
	f32 *gL_ptr,
	usize window_size
) {
	AttnForward attn_forward = AttnForward();
	attn_forward.run(seq_len, gQ_ptr, gKV_ptr, gSinkK_ptr, gSinkV_ptr, gMax_ptr, gOut_ptr, gL_ptr, window_size);
}
