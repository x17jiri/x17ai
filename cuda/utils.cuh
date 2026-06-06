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
				round_cast<bf16>(src.v8x16[row].h8x8[col].get0()),
				round_cast<bf16>(src.v8x16[row].h8x8[col].get1())
			);
		}
	}
}

X17_DEVICE void cast(b32::Fragment_16x16<i32> const &src, b32::Fragment_16x16<f32> &dst) {
	X17_UNROLL for (usize row = 0; row < 2; ++row) {
		X17_UNROLL for (usize col = 0; col < 2; ++col) {
			dst.v8x16[row].h8x8[col].set(
				round_cast<f32>(src.v8x16[row].h8x8[col].get0()),
				round_cast<f32>(src.v8x16[row].h8x8[col].get1())
			);
		}
	}
}

template<const f64 SCALE = 1.0>
X17_DEVICE void trunc_cast(b32::Fragment_16x16<f32> const &src, b8::Fragment_16x16<u8> &dst) {
	X17_UNROLL for (usize row = 0; row < 2; ++row) {
		if constexpr (SCALE == 1.0) {
			dst.v8x16[row].set(
				__float2int_rz(src.v8x16[row].h8x8[0].get0()),
				__float2int_rz(src.v8x16[row].h8x8[0].get1()),
				__float2int_rz(src.v8x16[row].h8x8[1].get0()),
				__float2int_rz(src.v8x16[row].h8x8[1].get1())
			);
		} else {
			dst.v8x16[row].set(
				__float2int_rz(src.v8x16[row].h8x8[0].get0() * f32(SCALE)),
				__float2int_rz(src.v8x16[row].h8x8[0].get1() * f32(SCALE)),
				__float2int_rz(src.v8x16[row].h8x8[1].get0() * f32(SCALE)),
				__float2int_rz(src.v8x16[row].h8x8[1].get1() * f32(SCALE))
			);
		}
	}
}

X17_DEVICE void round_cast(b32::Fragment_16x16<f32> const &src, b8::Fragment_16x16<u8> &dst) {
	X17_UNROLL for (usize row = 0; row < 2; ++row) {
		dst.v8x16[row].set(
			__float2int_rn(src.v8x16[row].h8x8[0].get0()),
			__float2int_rn(src.v8x16[row].h8x8[0].get1()),
			__float2int_rn(src.v8x16[row].h8x8[1].get0()),
			__float2int_rn(src.v8x16[row].h8x8[1].get1())
		);
	}
}

template<bool SHUFFLE, f32 SCALE = 1.0f>
X17_DEVICE void cast(b8::Fragment_16x16<FixedI8> const &src, b32::Fragment_16x16<f32> &dst) {
	static_assert(SHUFFLE == false);
	union Packed4 {
		u32 tuple4;
		u16 tuple2[2];
		FixedI8 val[4];
	};
	Packed4 top, bot;
	top.tuple4 = src.v8x16[0].data;
	bot.tuple4 = src.v8x16[1].data;

	if constexpr (SCALE == 1.0) {
		dst.v8x16[0].h8x8[0].set(top.val[0], top.val[1]);
		dst.v8x16[0].h8x8[1].set(top.val[2], top.val[3]);
		dst.v8x16[1].h8x8[0].set(bot.val[0], bot.val[1]);
		dst.v8x16[1].h8x8[1].set(bot.val[2], bot.val[3]);
	} else {
		constexpr i32 INT_SCALE = u32(SCALE);
		constexpr i32 MAX_INT_SCALE = std::numeric_limits<i32>::max() / 128;
		constexpr i32 MIN_INT_SCALE = -MAX_INT_SCALE;
		if constexpr (
			f32(INT_SCALE) == SCALE
			&& INT_SCALE > MIN_INT_SCALE
			&& INT_SCALE < MAX_INT_SCALE
		) {
			dst.v8x16[0].h8x8[0].set(i32(top.val[0]) * INT_SCALE, i32(top.val[1]) * INT_SCALE);
			dst.v8x16[0].h8x8[1].set(i32(top.val[2]) * INT_SCALE, i32(top.val[3]) * INT_SCALE);
			dst.v8x16[1].h8x8[0].set(i32(bot.val[0]) * INT_SCALE, i32(bot.val[1]) * INT_SCALE);
			dst.v8x16[1].h8x8[1].set(i32(bot.val[2]) * INT_SCALE, i32(bot.val[3]) * INT_SCALE);
		} else {
			dst.v8x16[0].h8x8[0].set(top.val[0] * SCALE, top.val[1] * SCALE);
			dst.v8x16[0].h8x8[1].set(top.val[2] * SCALE, top.val[3] * SCALE);
			dst.v8x16[1].h8x8[0].set(bot.val[0] * SCALE, bot.val[1] * SCALE);
			dst.v8x16[1].h8x8[1].set(bot.val[2] * SCALE, bot.val[3] * SCALE);
		}
	}
}

template<bool SHUFFLE>
X17_DEVICE void cast(b8::Fragment_16x16<FixedI8> const &src, b16::Fragment_16x16<bf16> &dst) {
	b32::Fragment_16x16<f32> tmp;
	cast<SHUFFLE>(src, tmp);
	cast(tmp, dst);
}

template<bool SHUFFLE>
X17_DEVICE void round_clamp_cast(b32::Fragment_16x16<f32> const &src, b8::Fragment_16x16<FixedI8> &dst) {
	static_assert(SHUFFLE == false);
	union Packed4 {
		u32 tuple4;
		u16 tuple2[2];
		FixedI8 val[4];
	};
	Packed4 top, bot;

	top.val[0] = b8::to_fixedi8(src.v8x16[0].h8x8[0].get0());
	top.val[1] = b8::to_fixedi8(src.v8x16[0].h8x8[0].get1());
	top.val[2] = b8::to_fixedi8(src.v8x16[0].h8x8[1].get0());
	top.val[3] = b8::to_fixedi8(src.v8x16[0].h8x8[1].get1());

	bot.val[0] = b8::to_fixedi8(src.v8x16[1].h8x8[0].get0());
	bot.val[1] = b8::to_fixedi8(src.v8x16[1].h8x8[0].get1());
	bot.val[2] = b8::to_fixedi8(src.v8x16[1].h8x8[1].get0());
	bot.val[3] = b8::to_fixedi8(src.v8x16[1].h8x8[1].get1());

	dst.v8x16[0].data = top.tuple4;
	dst.v8x16[1].data = bot.tuple4;
}

template<bool SHUFFLE, f32 SCALE = 1.0f>
X17_DEVICE void cast(b8::Fragment_16x32<FixedI8> const &src, b32::Fragment_16x32<f32> &dst) {
	X17_UNROLL for (usize i = 0; i < 2; ++i) {
		cast<SHUFFLE, SCALE>(src.h16x16[i], dst.h16x16[i]);
	}
}

template<bool SHUFFLE>
X17_DEVICE void round_clamp_cast(b32::Fragment_16x32<f32> const &src, b8::Fragment_16x32<FixedI8> &dst) {
	X17_UNROLL for (usize i = 0; i < 2; ++i) {
		round_clamp_cast<SHUFFLE>(src.h16x16[i], dst.h16x16[i]);
	}
}

template<bool SHUFFLE, f32 SCALE = 1.0f>
X17_DEVICE void cast(b8::Fragment_32x32<FixedI8> const &src, b32::Fragment_32x32<f32> &dst) {
	X17_UNROLL for (usize j = 0; j < 2; ++j) {
		cast<SHUFFLE, SCALE>(src.v16x32[j], dst.v16x32[j]);
	}
}

template<bool SHUFFLE>
X17_DEVICE void round_clamp_cast(b32::Fragment_32x32<f32> const &src, b8::Fragment_32x32<FixedI8> &dst) {
	X17_UNROLL for (usize j = 0; j < 2; ++j) {
		round_clamp_cast<SHUFFLE>(src.v16x32[j], dst.v16x32[j]);
	}
}

//--------------------------------------------------------------------------------------------------

X17_DEVICE void mma_a_bt(
	b16::Fragment_16x16<bf16> const &a,
	b16::Fragment_16x16<bf16> const &b,
	b32::Fragment_16x16<f32> &c
) {
	sm80::mma_bf16_f32(
		c.v8x16[0].h8x8[0].data0, c.v8x16[0].h8x8[0].data1,
		c.v8x16[1].h8x8[0].data0, c.v8x16[1].h8x8[0].data1,
		a.sub[0][0].val, a.sub[1][0].val, a.sub[0][1].val, a.sub[1][1].val,
		b.sub[0][0].val, b.sub[0][1].val,
		c.v8x16[0].h8x8[0].data0, c.v8x16[0].h8x8[0].data1,
		c.v8x16[1].h8x8[0].data0, c.v8x16[1].h8x8[0].data1
	);
	sm80::mma_bf16_f32(
		c.v8x16[0].h8x8[1].data0, c.v8x16[0].h8x8[1].data1,
		c.v8x16[1].h8x8[1].data0, c.v8x16[1].h8x8[1].data1,
		a.sub[0][0].val, a.sub[1][0].val, a.sub[0][1].val, a.sub[1][1].val,
		b.sub[1][0].val, b.sub[1][1].val,
		c.v8x16[0].h8x8[1].data0, c.v8x16[0].h8x8[1].data1,
		c.v8x16[1].h8x8[1].data0, c.v8x16[1].h8x8[1].data1
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
