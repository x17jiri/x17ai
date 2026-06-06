//------------------------------------------------------------------------------
//
// Copyright 2026 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#pragma once

#include <bit>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdint.h>
#include <numbers>

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
	#define X17_UNROLL    _Pragma("unroll")
	#define X17_NO_UNROLL _Pragma("unroll 1")
#else
	#define X17_UNROLL
	#define X17_NO_UNROLL
#endif

#define X17_DEVICE __forceinline__ __device__
#define X17_HOST_DEVICE __forceinline__ __host__ __device__
#define X17_KERNEL(LAUNCH_BOUNDS) __global__ __launch_bounds__(LAUNCH_BOUNDS)

#ifndef X17_PRECISE_MATH
#define X17_PRECISE_MATH 0
#endif

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using i128 = __int128;

using isize = i32;
using usize = u32;

using f16 = __half;
using bf16 = __nv_bfloat16;
using f32 = float;
using f64 = double;

constexpr usize WARP_SIZE = 32;

//--------------------------------------------------------------------------------------------------

template<const usize CAP>
struct SMemAllocator {
	u32 _ptr;

	X17_DEVICE SMemAllocator(): _ptr(0) {}

	X17_DEVICE u32 alloc(usize size) {
		u32 result = _ptr;
		_ptr += size;
		return result;
	}

	X17_DEVICE void finish() {
		// TODO: assert _ptr == CAP
	}
};

//--------------------------------------------------------------------------------------------------

template<typename To, typename From>
struct Round_cast;

template<>
struct Round_cast<bf16, f32> {
	X17_DEVICE static bf16 cast(f32 x) {
		return __float2bfloat16_rn(x);
	}
};

template<>
struct Round_cast<f32, i8> {
	X17_DEVICE static f32 cast(i8 x) {
		return __int2float_rz(x);
	}
};

template<>
struct Round_cast<bf16, i8> {
	X17_DEVICE static bf16 cast(i8 x) {
		return __float2bfloat16_rn(
			__int2float_rz(x)
		);
	}
};

template<>
struct Round_cast<f32, i32> {
	X17_DEVICE static f32 cast(i32 x) {
		return __int2float_rn(x);
	}
};

template<>
struct Round_cast<f32, u8> {
	X17_DEVICE static f32 cast(u8 x) {
		return __int2float_rz(x);
	}
};

template<>
struct Round_cast<u8, f32> {
	X17_DEVICE static u8 cast(f32 x) {
		return __float2int_rn(x);
	}
};

template<typename To, typename From>
X17_DEVICE To round_cast(From x) {
	return Round_cast<To, From>::cast(x);
}

//--------------------------------------------------------------------------------------------------

namespace math {

	X17_DEVICE f32 max(f32 a, f32 b) {
		return fmaxf(a, b);
	}

	template<const size_t N>
	requires(N > 0)
	X17_DEVICE f32 max(const f32 (&arr)[N]) {
		f32 result = arr[0];
		X17_UNROLL for (size_t i = 1; i < N; ++i) {
			result = fmaxf(result, arr[i]);
		}
		return result;
	}

	X17_DEVICE f32 fma(f32 mul1, f32 mul2, f32 add) {
		return __fmaf_rn(mul1, mul2, add);
	}

	constexpr f64 constexpr_sqrt(f64 x) {
		if (x < 0.0 || x != x) { return std::numeric_limits<f64>::quiet_NaN(); }
		if (x == 0.0) { return 0.0; }
		if (x == std::numeric_limits<f64>::infinity()) { return x; }

		f64 above = std::max(x, 1.0);
		f64 below = 0.0;
		bool stop = false;
		while (!stop) {
			f64 v = (0.5 * above) + (0.5 * below);
			f64 t = x / v;

			f64 new_above = t <= v ? v : above;
			f64 new_below = t >= v ? v : below;

			stop = (new_above == above) && (new_below == below);

			above = new_above;
			below = new_below;
		}

		f64 above_err = (above * above) - x;
		f64 below_err = x - (below * below);
		if (above_err > below_err) {
			return below;
		} else {
			return above;
		}
	}

	constexpr f64 constexpr_inv_sqrt(f64 x) {
		if (x < 0.0 || x != x) { return std::numeric_limits<f64>::quiet_NaN(); }
		if (x == 0.0) { return std::numeric_limits<f64>::infinity(); }
		if (x == std::numeric_limits<f64>::infinity()) { return 0.0; }

		f64 above = 2.0 / constexpr_sqrt(x);
		f64 below = 0.0;
		bool stop = false;
		while (!stop) {
			f64 v = (0.5 * above) + (0.5 * below);

			f64 t = (x * v) * v;

			f64 new_above = t >= 1.0 ? v : above;
			f64 new_below = t <= 1.0 ? v : below;

			stop = (new_above == above) && (new_below == below);

			above = new_above;
			below = new_below;
		}

		f64 above_err = ((x * above) * above) - 1.0;
		f64 below_err = 1.0 - ((x * below) * below);
		if (above_err > below_err) {
			return below;
		} else {
			return above;
		}
	}

	/// Compile-time `log2(x)` for positive finite `x`.
	///
	/// The algorithm splits `x = m * 2^e` by decoding the IEEE-754 bits, so the
	/// exponent contributes exactly as `e`. The mantissa term is then evaluated with
	///
	///   `ln(m) = 2 * (y + y^3/3 + y^5/5 + ...)`, where `y = (m - 1) / (m + 1)`.
	///
	/// Since `m` is normalized into `[1, 2)`, `|y| <= 1/3` and the odd-power series
	/// converges quickly enough for a small fixed number of terms.
	consteval f64 constexpr_log2(f64 x) {
		u64 bits = std::bit_cast<u64>(x);
		i32 exponent = i32((bits >> 52) & 0x7ffull) - 1023;
		f64 mantissa = std::bit_cast<f64>((bits & 0x000fffffffffffffull) | 0x3ff0000000000000ull);
		f64 y = (mantissa - 1.0) / (mantissa + 1.0);
		f64 y_sq = y * y;
		f64 term = y;
		f64 series = 0.0;
		for (i32 n = 1; n <= 21; n += 2) {
			series += term / f64(n);
			term *= y_sq;
		}
		return f64(exponent) + 2.0 * std::numbers::log2e_v<f64> * series;
	}

	struct UnaryResult {
		f32 val;
		f32 dVal;
	};

	namespace fast {
		/// Our underlying exp and log functions use this base.
		/// It was chosen to be fast and may change in the future.
		constexpr f64 b = 2.0;

		/// logb(e) = logb(2.71828...)
		constexpr f64 logb_e = std::numbers::log2e_v<f64>;

		/// logb(2) = 1.0 since b = 2.0
		constexpr f64 logb_2 = 1.0;

		/// logb(4) = 2.0 since b = 2.0
		constexpr f64 logb_4 = 2.0;

		/// Calculates `b^x` where `b` is our underlying base.
		/// The underlying base was chosen to be fast and may change in the future.
		///
		/// To calculate `B^x` for some other base `B`, use `expb(x * logb(B))`.
		X17_DEVICE f32 expb(f32 x) {
			#if X17_PRECISE_MATH
				return exp2f(x);
			#else
				f32 result;
				asm ("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
				return result;
			#endif
		}

		/// Calculate `logb(x)` where `b` is our underlying base.
		/// The underlying base was chosen to be fast and may change in the future.
		///
		/// To calculate `logB(x)` for some other base `B`, use `logb(x) / logb(B)`.
		X17_DEVICE f32 logb(f32 x) {
			#if X17_PRECISE_MATH
				return log2f(x);
			#else
				f32 result;
				asm ("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
				return result;
			#endif
		}

		consteval f64 constexpr_logb(f64 x) {
			return constexpr_log2(x) / constexpr_log2(math::fast::b);
		}

		consteval f64 constexpr_expb(f64 x) {
			i64 exponent = i64(x);
			if (x != f64(exponent)) {
				throw "constexpr_expb only supports integer exponents";
			}

			f64 factor = exponent >= 0 ? b : (1.0 / b);
			f64 result = 1.0;
			i64 steps = exponent >= 0 ? exponent : -exponent;
			for (i64 i = 0; i < steps; ++i) {
				result *= factor;
			}
			return result;
		}

		/// Single-instruction reciprocal approximation.
		X17_DEVICE f32 recip(f32 x) {
			#if X17_PRECISE_MATH
				return 1.0f / x;
			#else
				f32 result;
				asm ("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
				return result;
			#endif
		}

		X17_DEVICE f32 divide(f32 numerator, f32 denominator) {
			#if X17_PRECISE_MATH
				return numerator / denominator;
			#else
				f32 result;
				asm ("div.approx.ftz.f32 %0, %1, %2;\n" : "=f"(result) : "f"(numerator), "f"(denominator));
				return result;
			#endif
		}

		X17_DEVICE f32 sin(f32 x) {
			#if X17_PRECISE_MATH
				return sinf(x);
			#else
				f32 result;
				asm ("sin.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
				return result;
			#endif
		}

		X17_DEVICE f32 cos(f32 x) {
			#if X17_PRECISE_MATH
				return cosf(x);
			#else
				f32 result;
				asm ("cos.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
				return result;
			#endif
		}

		X17_DEVICE UnaryResult tanh(f32 x) {
			f32 val;
			#if X17_PRECISE_MATH
				val = tanhf(x);
			#else
				asm ("tanh.approx.f32 %0, %1;\n" : "=f"(val) : "f"(x));
			#endif
			f32 dVal = -math::fma(val, val, -1.0);
			return UnaryResult {
				.val = val,
				.dVal = dVal
			};
		}

		X17_DEVICE f32 sqrt(f32 x) {
			#if X17_PRECISE_MATH
				return sqrtf(x);
			#else
				f32 result;
				asm ("sqrt.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
				return result;
			#endif
		}

		X17_DEVICE f32 rsqrt(f32 x) {
			#if X17_PRECISE_MATH
				return rsqrtf(x);
			#else
				f32 result;
				asm ("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
				return result;
			#endif
		}

		X17_DEVICE f32 erf(f32 x) {
			#if X17_PRECISE_MATH
				return erff(x);
			#else
				f32 result;
				asm ("erf.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
				return result;
			#endif
		}

		X17_DEVICE f32 normal_cdf(f32 x) {
			// 1.0 / sqrt(2) == sqrt(2) / 2
			static constexpr f32 RSQRT_2 = std::numbers::sqrt2_v<f32> / 2.0f;
			return 0.5f * (1.0f + erf(x * RSQRT_2));
		}

		X17_DEVICE f32 sigmoid(f32 x) {
			return math::fma(0.5f, math::fast::tanh(0.5f * x).val, 0.5f);
		}

		X17_DEVICE f32 sigmoid_base4(f32 x) {
			return math::fma(0.5f, math::fast::tanh(std::numbers::ln2_v<f32> * x).val, 0.5f);
		}

		X17_DEVICE f32 silu(f32 x, f32 beta = 1.0f) {
			return math::fma(0.5f * x, math::fast::tanh((beta * 0.5f) * x).val, 0.5f * x);
		}

		// If gelu is applied to random input with normal distribution and unit variance,
		// it blocks about half of the values and so the output variance drops.
		// This multiplier restores it back to 1.
		constexpr f64 GELU_VAR_FIX_2 =
			1.0 / (
				(1.0 / 3.0)
				+ (1.0 / 2.0) * std::numbers::inv_pi_v<f64> * std::numbers::inv_sqrt3_v<f64>
			);

		/// Gaussian Error Linear Unit
		///
		/// The input is scaled by sqrt(INP_SCALE_2)
		/// The output is scaled by sqrt(OUT_SCALE_2 * VAR_FIX_2)
		///
		/// The scaling is folded into other constants at compile time and so is "for free"
		template<
			const f64 INP_SCALE_2 = 1.0,
			const f64 OUT_SCALE_2 = 1.0,
			const f64 VAR_FIX_2 = GELU_VAR_FIX_2
		>
		X17_DEVICE UnaryResult gelu(f32 x) {
			constexpr f64 k = constexpr_sqrt(INP_SCALE_2);
			constexpr f64 k2 = INP_SCALE_2;
			constexpr f64 ck = constexpr_sqrt(INP_SCALE_2 * 2.0 * std::numbers::inv_pi_v<f64>);
			constexpr f64 ck3 = 0.044715 * ck * k2;
			constexpr f64 Y_SCALE = 0.5 * constexpr_sqrt(OUT_SCALE_2 * VAR_FIX_2) * k;
			f32 x2 = x * x;
			f32 s = math::fma(f32(ck3) * x, x2, f32(ck) * x);
			auto t = math::fast::tanh(s);
			f32 x_ds_dx = math::fma(f32(3.0 * ck3) * x, x2, f32(ck) * x);
			return UnaryResult {
				.val = math::fma(f32(Y_SCALE) * x, t.val, f32(Y_SCALE) * x),
				.dVal = math::fma(
					f32(Y_SCALE) * t.dVal, x_ds_dx,
					math::fma(f32(Y_SCALE), t.val, f32(Y_SCALE))
				)
			};
		}

		/// `softplus(x) = log(1 + exp(x))`
		///
		/// We use numerically stable variant that doesn't overflow for large x
		///
		/// This function doesn't use `log1p`, because CUDA doesn't have a fast approximation.
		/// This causes imprecision.
		///
		/// Imprecision analysis:
		/// - Around zero, the exp result is comparable to `1.0` so `1.0 + exp` is precise
		/// - As the magnitude of `x` gets larger (both positive and negative), the `exp` becomes
		///   much smaller than `1.0` and will eventually get lost, i.e., `1.0 + exp == 1.0`
		/// - The `log` becomes `0.0` and we return just `x` or `0` depending on sign
		/// - What this means is we get to the asymptotes a bit sooner
		X17_DEVICE f32 imprecise_softplus(f32 x, f32 beta = 1.0f) {
			f32 scale = f32(logb_e) * beta;
			return fmaxf(0.0f, x) + logb(1.0f + expb(-fabsf(x * scale))) * recip(scale);
		}

		X17_DEVICE f32 imprecise_softplus_base4(f32 x, f32 beta = 1.0f) {
			f32 scale = f32(logb_4) * beta;
			return fmaxf(0.0f, x) + logb(1.0f + expb(-fabsf(x * scale))) * recip(scale);
		}

		/// Symmetric smooth cap: clamps x to [-C, +C] with a smooth transition.
		/// Exact zero at `x == 0`, exponentially sharp transition near +-C.
		/// Uses `softplus(x-C)` and `softplus(-x-C)` which cancel symmetrically near zero,
		/// preserving high precision.
		X17_DEVICE f32 smooth_cap(f32 x, f32 C = 16.0f, f32 beta = 1.0f) {
			// The imprecise_softplus is ok here. The errors of the two softplus calls cancel out
			// and we get very high precision
			return x - (imprecise_softplus(x - C, beta) - imprecise_softplus(-x - C, beta));
		}
	}
}

//--------------------------------------------------------------------------------------------------

X17_DEVICE void sync_threads() {
	__syncthreads();
}

X17_DEVICE void sync_warp() {
	__syncwarp();
}

X17_DEVICE bool any_sync(bool predicate) {
	return __any_sync(0xffffffff, predicate);
}

X17_DEVICE bool all_sync(bool predicate) {
	return __all_sync(0xffffffff, predicate);
}

template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE T shuffle_xor_sync(T val, int lane_mask) {
	return __shfl_xor_sync(0xffffffff, val, lane_mask);
}

template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE T shuffle_sync(T val, int src_lane) {
	return __shfl_sync(0xffffffff, val, src_lane);
}

X17_DEVICE void store_shared_4x32b(u32 ptr, f32 a, f32 b, f32 c, f32 d) {
	asm volatile(
		"st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
		:
		: "r"(ptr), "f"(a), "f"(b), "f"(c), "f"(d)
		: "memory"
	);
}

X17_DEVICE void store_shared_4x32b(u32 ptr, u32 a, u32 b, u32 c, u32 d) {
	asm volatile(
		"st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
		:
		: "r"(ptr), "r"(a), "r"(b), "r"(c), "r"(d)
		: "memory"
	);
}

template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE void store_shared_1x32b(u32 ptr, T value) {
	asm volatile(
		"st.shared.f32 [%0], %1;\n"
		:
		: "r"(ptr), "f"(value)
		: "memory"
	);
}

template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE void load_shared_4x32b(u32 ptr, T &a, T &b, T &c, T &d) {
	asm volatile(
		"ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
		: "=f"(a), "=f"(b), "=f"(c), "=f"(d)
		: "r"(ptr)
	);
}

X17_DEVICE void load_shared_4x32b(u32 ptr, u32 &a, u32 &b, u32 &c, u32 &d) {
	asm volatile(
		"ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
		: "=r"(a), "=r"(b), "=r"(c), "=r"(d)
		: "r"(ptr)
	);
}

template<typename T>
requires(sizeof(T) == 2)
X17_DEVICE void load_shared_2x16b(u32 ptr, T &a, T &b) {
	u32 word;
	asm volatile(
		"ld.shared.f32 %0, [%1];\n"
		: "=r"(word)
		: "r"(ptr)
	);

	union {
		u32 val;
		T halves[2];
	} u;
	u.val = word;
	a = u.halves[0];
	b = u.halves[1];
}

template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE void load_shared_2x32b(u32 ptr, T &a, T &b) {
	asm volatile(
		"ld.shared.v2.f32 {%0, %1}, [%2];\n"
		: "=f"(a), "=f"(b)
		: "r"(ptr)
	);
}

template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE T load_shared_1x32b(u32 ptr) {
	T value;
	asm volatile(
		"ld.shared.f32 %0, [%1];\n"
		: "=f"(value)
		: "r"(ptr)
	);
	return value;
}

template<const usize N, typename T>
requires(sizeof(T) == 4 && (N == 1 || N == 2 || N == 4))
X17_DEVICE void load_shared_Nx32b(u32 ptr, T (&values)[N]) {
	if constexpr (N == 1) {
		values[0] = load_shared_1x32b<T>(ptr);
	} else if constexpr (N == 2) {
		load_shared_2x32b(ptr, values[0], values[1]);
	} else {
		load_shared_4x32b(ptr, values[0], values[1], values[2], values[3]);
	}
}

/// Load one f32 value from global memory in a single 32-bit transaction.
template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE T load_gmem_1x32b(const T *ptr) {
	T value;
	asm volatile(
		"ld.global.f32 %0, [%1];\n"
		: "=f"(value)
		: "l"(ptr)
	);
	return value;
}

/// Load two consecutive f32 values from global memory in a single 64-bit transaction.
template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE void load_gmem_2x32b(const T *ptr, T &a, T &b) {
	asm volatile(
		"ld.global.v2.f32 {%0, %1}, [%2];\n"
		: "=f"(a), "=f"(b)
		: "l"(ptr)
	);
}

template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE void load_gmem_4x32b(const T *ptr, T &a, T &b, T &c, T &d) {
	asm volatile(
		"ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
		: "=f"(a), "=f"(b), "=f"(c), "=f"(d)
		: "l"(ptr)
	);
}

template<const usize N, typename T>
requires(sizeof(T) == 4 && (N == 1 || N == 2 || N == 4))
X17_DEVICE void load_gmem_Nx32b(const T *ptr, T (&values)[N]) {
	if constexpr (N == 1) {
		values[0] = load_gmem_1x32b(ptr);
	} else if constexpr (N == 2) {
		load_gmem_2x32b(ptr, values[0], values[1]);
	} else {
		load_gmem_4x32b(ptr, values[0], values[1], values[2], values[3]);
	}
}

//--------------------------------------------------------------------------------------------------

X17_DEVICE u32 cast_smem_ptr_to_uint(void const *const ptr) {
	return static_cast<u32>(__cvta_generic_to_shared(ptr));
}

//--------------------------------------------------------------------------------------------------

namespace sm75 {
	/// `smem_src` must be 16-byte aligned.
	X17_DEVICE void ldmatrix_8x8xu16_x2(
		u32 smem_src,
		u32 &dst0, u32 &dst1
	) {
		asm volatile (
			"\nldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
			: "=r"(dst0), "=r"(dst1)
			: "r"(smem_src)
		);
	}

	/// `smem_src` must be 16-byte aligned.
	X17_DEVICE void ldmatrix_8x8xu16_x4(
		u32 smem_src,
		u32 &dst0, u32 &dst1, u32 &dst2, u32 &dst3
	) {
		asm volatile (
			"\nldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
			: "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
			: "r"(smem_src)
		);
	}

	/// `smem_src` must be 16-byte aligned.
	X17_DEVICE void ldmatrix_t_8x8xu16_x4(
		u32 smem_src,
		u32 &dst0, u32 &dst1, u32 &dst2, u32 &dst3
	) {
		asm volatile (
			"\nldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
			: "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
			: "r"(smem_src)
		);
	}

	X17_DEVICE void movmatrix(uint32_t src, uint32_t &dst) {
		asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
			: "=r"(dst)
			:  "r"(src));
	}
}

//--------------------------------------------------------------------------------------------------

namespace sm80 {
	using namespace sm75;

	/// Both `gmem_src` and `smem_dst` must be 16-byte (128-bit) aligned.
	template<typename T>
	X17_DEVICE void cp_async(T const *gmem_src, u32 smem_dst) {
		asm volatile (
			"\ncp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
			:
			: "r"(smem_dst), "l"(gmem_src), "n"(sizeof(u128))
		);
	}

	X17_DEVICE void cp_async_commit() {
		asm volatile("\ncp.async.commit_group;\n" : :);
	}

	/// Blocks until all but N previous cp.async.commit_group operations have committed.
	template<int N = 0>
	X17_DEVICE void cp_async_wait() {
		if constexpr (N == 0) {
			asm volatile("\ncp.async.wait_all;\n" : :);
		} else {
			asm volatile("\ncp.async.wait_group %0;\n" : : "n"(N));
		}
	}

	/// Named barrier sync: syncs exactly THREAD_COUNT threads at barrier `bar_id`.
	/// Entire warps must participate, so THREAD_COUNT must be a multiple of WARP_SIZE.
	/// bar_id 0 is reserved for sync_threads(); use 1..15 for custom barriers.
	/// SM80+ supports 16 named barriers (0-15).
	template<u32 THREAD_COUNT>
	requires(THREAD_COUNT % WARP_SIZE == 0)
	X17_DEVICE void bar_sync(u32 bar_id) {
		if constexpr (THREAD_COUNT == WARP_SIZE) {
			sync_warp();
		} else {
			asm volatile("bar.sync %0, %1;" : : "r"(bar_id), "n"(THREAD_COUNT));
		}
	}

	X17_DEVICE void mma_bf16_f32(
		u32       &d0, u32       &d1, u32       &d2, u32       &d3,
		u32 const &a0, u32 const &a1, u32 const &a2, u32 const &a3,
		u32 const &b0, u32 const &b1,
		u32 const &c0, u32 const &c1, u32 const &c2, u32 const &c3
	) {
		asm volatile(
			"\nmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
			"{%0,  %1,  %2,  %3},"
			"{%4,  %5,  %6,  %7},"
			"{%8,  %9},"
			"{%10, %11, %12, %13};\n"
			:
				"=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
			:
				"r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
				"r"(b0),  "r"(b1),
				"r"(c0),  "r"(c1),  "r"(c2),  "r"(c3)
		);
	}

	X17_DEVICE void mma_i8_i32(
		u32       &d0, u32       &d1, u32       &d2, u32       &d3,
		u32 const &a0, u32 const &a1, u32 const &a2, u32 const &a3,
		u32 const &b0, u32 const &b1,
		u32 const &c0, u32 const &c1, u32 const &c2, u32 const &c3
	) {
		asm volatile(
			"\nmma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
			"{%0,  %1,  %2,  %3},"
			"{%4,  %5,  %6,  %7},"
			"{%8,  %9},"
			"{%10, %11, %12, %13};\n"
			:
				"=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
			:
				"r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
				"r"(b0),  "r"(b1),
				"r"(c0),  "r"(c1),  "r"(c2),  "r"(c3)
		);
	}

	X17_DEVICE void mma_u8_i8_i32(
		u32       &d0, u32       &d1, u32       &d2, u32       &d3,
		u32 const &a0, u32 const &a1,
		u32 const &b0,
		u32 const &c0, u32 const &c1, u32 const &c2, u32 const &c3
	) {
		asm volatile(
			"\nmma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 "
			"{%0,  %1,  %2,  %3},"
			"{%4,  %5},"
			"{%6},"
			"{%7,  %8,  %9,  %10};\n"
			:
				"=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
			:
				"r"(a0),  "r"(a1),
				"r"(b0),
				"r"(c0),  "r"(c1),  "r"(c2),  "r"(c3)
		);
	}
}

//--------------------------------------------------------------------------------------------------

struct StrideBytes {
	usize value;

	X17_HOST_DEVICE constexpr StrideBytes(usize value): value(value) {}
};

template<typename T, const usize M, const usize N>
struct GMatrix {
	T *_ptr;
	StrideBytes _stride_bytes;

	X17_HOST_DEVICE constexpr GMatrix(T *ptr):
		_ptr(ptr),
		_stride_bytes(N * sizeof(T))
	{}
	X17_HOST_DEVICE constexpr GMatrix(T *ptr, usize stride):
		_ptr(ptr),
		_stride_bytes(StrideBytes(stride * sizeof(T)))
	{}
	X17_HOST_DEVICE constexpr GMatrix(T *ptr, StrideBytes stride_bytes):
		_ptr(ptr),
		_stride_bytes(stride_bytes)
	{}

	X17_HOST_DEVICE constexpr usize m_rows() const { return M; }
	X17_HOST_DEVICE constexpr usize n_cols() const { return N; }
	X17_HOST_DEVICE constexpr usize stride_bytes() const { return _stride_bytes.value; }
	X17_HOST_DEVICE constexpr usize elems() const { return M * N; }

	template<const usize TILE_M>
	X17_HOST_DEVICE constexpr GMatrix<T, TILE_M, N> tile_m(usize tile_idx) const {
		return GMatrix<T, TILE_M, N>{
			reinterpret_cast<T *>(
				reinterpret_cast<u8 *>(_ptr)
				+ size_t(TILE_M) * size_t(tile_idx) * size_t(_stride_bytes.value)
			),
			_stride_bytes
		};
	}

	template<const usize NEW_N>
	X17_HOST_DEVICE constexpr GMatrix<T, M, NEW_N> slice_n(usize col_offset) const {
		return GMatrix<T, M, NEW_N>{_ptr + col_offset, _stride_bytes};
	}
};

template<typename T, const usize N>
struct GMatrixDynSize {
	T *_ptr;
	StrideBytes _stride_bytes;

	X17_HOST_DEVICE constexpr GMatrixDynSize(T *ptr):
		_ptr(ptr),
		_stride_bytes(N * sizeof(T))
	{}
	X17_HOST_DEVICE constexpr GMatrixDynSize(T *ptr, usize stride):
		_ptr(ptr),
		_stride_bytes(StrideBytes(stride * sizeof(T)))
	{}
	X17_HOST_DEVICE constexpr GMatrixDynSize(T *ptr, StrideBytes stride_bytes):
		_ptr(ptr),
		_stride_bytes(stride_bytes)
	{}

	X17_HOST_DEVICE constexpr usize n_cols() const { return N; }
	X17_HOST_DEVICE constexpr usize stride_bytes() const { return _stride_bytes.value; }

	template<const usize TILE_M>
	X17_HOST_DEVICE constexpr GMatrix<T, TILE_M, N> tile_m(size_t tile_idx) const {
		return GMatrix<T, TILE_M, N>{
			reinterpret_cast<T *>(
				reinterpret_cast<u8 *>(_ptr)
				+ size_t(TILE_M) * size_t(tile_idx) * size_t(_stride_bytes.value)
			),
			_stride_bytes
		};
	}

	template<const usize NEW_N>
	X17_HOST_DEVICE constexpr GMatrixDynSize<T, NEW_N> slice_n(usize col_offset) const {
		return GMatrixDynSize<T, NEW_N>{_ptr + col_offset, _stride_bytes};
	}
};

//--------------------------------------------------------------------------------------------------

/// Each fragment has 8 columns of 2-byte type, but a thread always has
/// 2 consecutive values. So let's assume each fragment has 4 double-columns.
///
/// We have 4 fragments, so 4x4 = 16 double-columns total.
///
/// Let's say the values in the first row are 0, 1, 2, ..., 15, where:
/// - thread 0 has values 0, 4, 8, 12
/// - thread 1 has values 1, 5, 9, 13
/// - ...
/// Then after this shuffle:
/// - thread 0 has values 0, 1, 2, 3
/// - thread 1 has values 4, 5, 6, 7
/// - ...
X17_DEVICE void shuffle_4x4(u32 &r0, u32 &r1, u32 &r2, u32 &r3) {
	usize tid = threadIdx.x;

	u32 u0 = (tid & 1) == 0 ? r0 : r1;
	u32 u1 = (tid & 1) == 0 ? r1 : r0;
	u32 u2 = (tid & 1) == 0 ? r2 : r3;
	u32 u3 = (tid & 1) == 0 ? r3 : r2;

	r0 = u0; r1 = u1; r2 = u2; r3 = u3;

	u0 = (tid & 2) == 0 ? r0 : r2;
	u2 = (tid & 2) == 0 ? r2 : r0;
	u1 = (tid & 2) == 0 ? r1 : r3;
	u3 = (tid & 2) == 0 ? r3 : r1;

	r0 = u0; r1 = u1; r2 = u2; r3 = u3;

	r1 = shuffle_xor_sync(r1, 1);
	r2 = shuffle_xor_sync(r2, 2);
	r3 = shuffle_xor_sync(r3, 3);

	u0 = (tid & 1) == 0 ? r0 : r1;
	u1 = (tid & 1) == 0 ? r1 : r0;
	u2 = (tid & 1) == 0 ? r2 : r3;
	u3 = (tid & 1) == 0 ? r3 : r2;

	r0 = u0; r1 = u1; r2 = u2; r3 = u3;

	u0 = (tid & 2) == 0 ? r0 : r2;
	u2 = (tid & 2) == 0 ? r2 : r0;
	u1 = (tid & 2) == 0 ? r1 : r3;
	u3 = (tid & 2) == 0 ? r3 : r1;

	r0 = u0; r1 = u1; r2 = u2; r3 = u3;
}

//--------------------------------------------------------------------------------------------------

/// This intentionally ignores NaN and Inf
X17_DEVICE f32 f8_to_f32(u8 x) {
	u32 nosign = u32(x) & 0x7Fu;

	u32 subnormal = __float_as_uint(
		__uint_as_float(nosign | 0x46800000u) - __uint_as_float(0x46800000u)
	);

	u32 normal = u32(nosign << 20) + 0x3C000000u;

	u32 sign = u32(x & 0x80u) << 24;
	u32 exp = u32(x) & 0x78u;
	u32 value = sign ^ (exp == 0 ? subnormal : normal);
	return __uint_as_float(value);
}

/// This intentionally ignores NaN and Inf
X17_DEVICE bf16 f8_to_bf16(u8 x) {
	return __ushort_as_bfloat16(x);
	u16 sign = u16(x & 0x80u) << 8;
	return
		(x & 0x78u) == 0
			? __ushort_as_bfloat16(sign)
			: __ushort_as_bfloat16((sign | u16(u16(x & 0x7Fu) << 4)) + 0x3C00u);
/*
	union {
		f32 value;
		struct { u16 low; bf16 high; } parts;
	} t;
	t.value = f8_to_f32(x);
	return t.parts.high;
*/
}

/// makes sure that 4 bytes read from memory are treated as little endian
X17_HOST_DEVICE constexpr u32 from_le32(u32 val) {
	#if defined(__CUDA_ARCH__)
		return val;
	#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
		return __builtin_bswap32(val);
	#else
		return val;
	#endif
}

//--------------------------------------------------------------------------------------------------

X17_HOST_DEVICE constexpr u32 greatest_common_divisor(u32 a, u32 b) {
	while (b != 0) {
		u32 temp = b;
		b = a % b;
		a = temp;
	}
	return a;
}

X17_HOST_DEVICE constexpr u32 least_common_multiple(u32 a, u32 b) {
  return a * (b / greatest_common_divisor(a, b));
}

//--------------------------------------------------------------------------------------------------

template<typename T, const usize N>
struct SVector {
	u32 _ptr;

	X17_DEVICE constexpr SVector() : _ptr(0) {}
	X17_DEVICE constexpr SVector(void *ptr): SVector(cast_smem_ptr_to_uint(ptr)) {}
	X17_DEVICE constexpr SVector(u32 ptr): _ptr(ptr) {}

	X17_DEVICE constexpr usize elems() const { return N; }
	X17_DEVICE constexpr usize bytes() const { return N * sizeof(T); }

	template<const usize THREADS_PER_BLOCK>
	X17_DEVICE void cp_async_from(usize tid, T const *src) {
		__builtin_assume(tid < THREADS_PER_BLOCK);

		constexpr usize CP_BYTES = sizeof(u128);
		constexpr usize CP_COUNT = (N * sizeof(T)) / CP_BYTES;
		static_assert((N * sizeof(T)) % CP_BYTES == 0);
		static_assert(CP_COUNT <= THREADS_PER_BLOCK);

		if (tid < CP_COUNT) {
			u8 const *src_ptr = reinterpret_cast<u8 const *>(src) + tid * CP_BYTES;
			u32 dst_ptr = _ptr + tid * CP_BYTES;
			sm80::cp_async(reinterpret_cast<u128 const *>(src_ptr), dst_ptr);
		}
	}
};

X17_DEVICE void async_load_commit() {
	sm80::cp_async_commit();
}

template<int N = 0>
X17_DEVICE void async_load_wait() {
	sm80::cp_async_wait<N>();
}

using sm80::bar_sync;

//--------------------------------------------------------------------------------------------------
