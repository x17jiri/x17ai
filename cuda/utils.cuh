#pragma once

#include "utils_core.cuh"
#include "utils_b8.cuh"
#include "utils_b16.cuh"
#include "utils_b32.cuh"

using FixedI8 = b8::FixedI8;

//--------------------------------------------------------------------------------------------------

X17_DEVICE void cast(b32::Fragment_16x16<f32> const &src, b16::Fragment_16x16<bf16> &dst) {
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

template<bool SHUFFLE = true>
X17_DEVICE void cast(b8::Fragment_16x16<FixedI8> const &src, b16::Fragment_16x16<bf16> &dst) {
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

//--------------------------------------------------------------------------------------------------

X17_DEVICE void mma_a_bt(
	b16::Fragment_16x16<bf16> const &a,
	b16::Fragment_16x16<bf16> const &b,
	b32::Fragment_16x16<f32> &c
) {
	sm80::mma_bf16_f32(
		c.v8x16[0].h8x8[0].val0, c.v8x16[0].h8x8[0].val1,
		c.v8x16[1].h8x8[0].val0, c.v8x16[1].h8x8[0].val1,
		a.sub[0][0].val, a.sub[1][0].val, a.sub[0][1].val, a.sub[1][1].val,
		b.sub[0][0].val, b.sub[0][1].val,
		c.v8x16[0].h8x8[0].val0, c.v8x16[0].h8x8[0].val1,
		c.v8x16[1].h8x8[0].val0, c.v8x16[1].h8x8[0].val1
	);
	sm80::mma_bf16_f32(
		c.v8x16[0].h8x8[1].val0, c.v8x16[0].h8x8[1].val1,
		c.v8x16[1].h8x8[1].val0, c.v8x16[1].h8x8[1].val1,
		a.sub[0][0].val, a.sub[1][0].val, a.sub[0][1].val, a.sub[1][1].val,
		b.sub[1][0].val, b.sub[1][1].val,
		c.v8x16[0].h8x8[1].val0, c.v8x16[0].h8x8[1].val1,
		c.v8x16[1].h8x8[1].val0, c.v8x16[1].h8x8[1].val1
	);
}

//--------------------------------------------------------------------------------------------------

/*
template<
	const f64 INP_SCALE_2 = 1.0,
	const f64 OUT_SCALE_2 = 1.0,
	const f64 VAR_FIX_2 = math::fast::GELU_VAR_FIX_2
>
X17_DEVICE void geglu_and_backvec_(
	Fragment_8x8<f32> &i1,
	Fragment_8x8<f32> &i2,
	Fragment_8x8<bf16> &o
) {
	f32 gate1 = i1.first();
	f32 lin1 = i1.second();

	f32 gate2 = i2.first();
	f32 lin2 = i2.second();

	auto g1 = math::fast::gelu<INP_SCALE_2, INP_SCALE_2 * OUT_SCALE_2, VAR_FIX_2>(gate1);
	auto g2 = math::fast::gelu<INP_SCALE_2, INP_SCALE_2 * OUT_SCALE_2, VAR_FIX_2>(gate2);

	i1.set(
		lin1 * g1.dVal,
		g1.val
	);
	i2.set(
		lin2 * g2.dVal,
		g2.val
	);

	o.set(
		g1.val * lin1,
		g2.val * lin2
	);

	usize tid = threadIdx.x;
	o.transpose_();
	o.val = shuffle_sync(o.val, (tid & 12) * 2 + (tid & 16) / 4 + (tid & 3));
	o.transpose_();
}

X17_DEVICE void geglu_backward_(
	Fragment_8x8<bf16> &d_o,
	Fragment_8x8<bf16> &backvec1,
	Fragment_8x8<bf16> &backvec2
) {
	usize tid = threadIdx.x;
	d_o.transpose_();
	d_o.val = shuffle_sync(d_o.val, (tid & 24) / 2 + (tid & 4) * 4 + (tid & 3));
	d_o.transpose_();

	backvec1.set(
		round_cast<bf16>(f32(backvec1.first()) * f32(d_o.first())),
		round_cast<bf16>(f32(backvec1.second()) * f32(d_o.first()))
	);
	backvec2.set(
		round_cast<bf16>(f32(backvec2.first()) * f32(d_o.second())),
		round_cast<bf16>(f32(backvec2.second()) * f32(d_o.second()))
	);
}

/// Calculates GeGLU of consecutive values and stores the result to `o`.
/// The content of `i1`, `i2` is replace with backward multipliers.
template<
	const f64 INP_SCALE_2 = 1.0,
	const f64 OUT_SCALE_2 = 1.0,
	const f64 VAR_FIX_2 = math::fast::GELU_VAR_FIX_2
>
X17_DEVICE void geglu_and_backvec_(
	Fragment_16x16<f32> &i1,
	Fragment_16x16<f32> &i2,
	Fragment_16x16<bf16> &o
) {
	geglu_and_backvec_<INP_SCALE_2, OUT_SCALE_2, VAR_FIX_2>(i1.sub[0][0], i1.sub[0][1], o.sub[0][0]);
	geglu_and_backvec_<INP_SCALE_2, OUT_SCALE_2, VAR_FIX_2>(i2.sub[0][0], i2.sub[0][1], o.sub[0][1]);
	geglu_and_backvec_<INP_SCALE_2, OUT_SCALE_2, VAR_FIX_2>(i1.sub[1][0], i1.sub[1][1], o.sub[1][0]);
	geglu_and_backvec_<INP_SCALE_2, OUT_SCALE_2, VAR_FIX_2>(i2.sub[1][0], i2.sub[1][1], o.sub[1][1]);
}
*/
