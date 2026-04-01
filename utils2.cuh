//------------------------------------------------------------------------------
//
// Copyright 2026 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdint.h>
#include <numbers>

/*#if defined(__CUDACC_RTC__) || defined(__clang__)
	#define X17_UNROLL    _Pragma("unroll")
	#define X17_NO_UNROLL _Pragma("unroll 1")
#elif defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
	#define X17_UNROLL    #pragma unroll
	#define X17_NO_UNROLL #pragma unroll 1
#else*/
	#define X17_UNROLL
	#define X17_NO_UNROLL
/*#endif*/

#define X17_DEVICE __forceinline__ __device__
#define X17_HOST_DEVICE __forceinline__ __host__ __device__

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

namespace math {

	X17_DEVICE f32 max(f32 a, f32 b) {
		return fmaxf(a, b);
	}

	template<const size_t N>
	requires(N > 0)
	X17_DEVICE f32 max(const f32 (&arr)[N]) {
		f32 result = arr[0];
		X17_UNROLL for (size_t i = 1; i < N; i++) {
			result = fmaxf(result, arr[i]);
		}
		return result;
	}

	X17_DEVICE f32 fma(f32 mul1, f32 mul2, f32 add) {
		return __fmaf_rn(mul1, mul2, add);
	}

	namespace fast {
		/// Our underlying exp and log functions use this base.
		/// It was chosen to be fast and may change in the future.
		constexpr f64 b = 2.0;

		/// logb(e) = logb(2.71828...)
		constexpr f64 logb_e = std::numbers::log2e_v<f64>;

		/// logb(2) = 1.0 since b = 2.0
		constexpr f64 logb_2 = 1.0;

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

		X17_DEVICE f32 tanh(f32 x) {
			#if X17_PRECISE_MATH
				return tanhf(x);
			#else
				f32 result;
				asm ("tanh.approx.f32 %0, %1;\n" : "=f"(result) : "f"(x));
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
			return math::fma(0.5f, math::fast::tanh(0.5f * x), 0.5f);
		}

		X17_DEVICE f32 silu(f32 x, f32 beta = 1.0f) {
			return math::fma(0.5f * x, math::fast::tanh((beta * 0.5f) * x), 0.5f * x);
		}

		/// Gaussian Error Linear Unit (GELU) approximation
		X17_DEVICE f32 gelu(f32 x) {
			return silu(x, 1.702f);
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

template<typename T>
requires(sizeof(T) == 4)
X17_DEVICE void store_shared_4x32b(u32 ptr, T a, T b, T c, T d) {
	asm volatile(
		"st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
		:
		: "r"(ptr), "f"(a), "f"(b), "f"(c), "f"(d)
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

//--------------------------------------------------------------------------------------------------

X17_DEVICE u32 cast_smem_ptr_to_uint(void const *const ptr) {
	return static_cast<u32>(__cvta_generic_to_shared(ptr));
}

consteval f64 constexpr_sqrt(f64 x) {
	f64 r = x;
	for (int i = 0; i < 32; i++) r = 0.5f * (r + x / r);
	return r;
}

//--------------------------------------------------------------------------------------------------

namespace sm75 {
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
		f32       &d0, f32       &d1, f32       &d2, f32       &d3,
		u32 const &a0, u32 const &a1, u32 const &a2, u32 const &a3,
		u32 const &b0, u32 const &b1,
		f32 const &c0, f32 const &c1, f32 const &c2, f32 const &c3
	) {
		asm volatile(
			"\nmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
			"{%0,  %1,  %2,  %3},"
			"{%4,  %5,  %6,  %7},"
			"{%8,  %9},"
			"{%10, %11, %12, %13};\n"
			:
				"=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
			:
				"r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
				"r"(b0),  "r"(b1),
				"f"(c0),  "f"(c1),  "f"(c2),  "f"(c3)
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
	usize _m;
	StrideBytes _stride_bytes;

	X17_HOST_DEVICE constexpr GMatrixDynSize(T *ptr, usize m):
		_ptr(ptr),
		_m(m),
		_stride_bytes(N * sizeof(T))
	{}
	X17_HOST_DEVICE constexpr GMatrixDynSize(T *ptr, usize m, usize stride):
		_ptr(ptr),
		_m(m),
		_stride_bytes(StrideBytes(stride * sizeof(T)))
	{}
	X17_HOST_DEVICE constexpr GMatrixDynSize(T *ptr, usize m, StrideBytes stride_bytes):
		_ptr(ptr),
		_m(m),
		_stride_bytes(stride_bytes)
	{}

	X17_HOST_DEVICE constexpr usize m_rows() const { return _m; }
	X17_HOST_DEVICE constexpr usize n_cols() const { return N; }
	X17_HOST_DEVICE constexpr usize stride_bytes() const { return _stride_bytes.value; }
	X17_HOST_DEVICE constexpr usize elems() const { return _m * N; }

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
		return GMatrixDynSize<T, NEW_N>{_ptr + col_offset, _m, _stride_bytes};
	}
};

//--------------------------------------------------------------------------------------------------

template<typename T, const usize T_SIZE = sizeof(T)>
struct FragmentReg;

template<typename T>
requires(sizeof(T) == 2)
struct FragmentReg<T, 2> {
	u32 val;

	X17_DEVICE T first() const {
		union {
			u32 val;
			T halves[2];
		} a;
		a.val = val;
		return a.halves[0];
	}

	X17_DEVICE T second() const {
		union {
			u32 val;
			T halves[2];
		} a;
		a.val = val;
		return a.halves[1];
	}

	X17_DEVICE void set(T first, T second) {
		union {
			u32 val;
			T halves[2];
		} a;
		a.halves[0] = first;
		a.halves[1] = second;
		val = a.val;
	}

	X17_DEVICE void zero_() {
		val = 0;
	}
};

template<typename T>
requires(sizeof(T) == 4)
struct FragmentReg<T, 4> {
	T val0;
	T val1;

	X17_DEVICE T first() const {
		return val0;
	}

	X17_DEVICE T second() const {
		return val1;
	}

	X17_DEVICE void set(T first, T second) {
		val0 = first;
		val1 = second;
	}

	X17_DEVICE void zero_() {
		val0 = T();
		val1 = T();
	}
};

/// An 8x8 tile that is held in registers by the whole warp.
///
/// The first thread holds the first two columns on row zero,
/// the second thread holds the next two columns, ...
template<typename T>
struct Fragment_8x8: FragmentReg<T> {
	template<typename U>
	X17_DEVICE FragmentReg<U> cast_reg() const {
		FragmentReg<U> result;
		result.set(
			static_cast<U>(this->first()),
			static_cast<U>(this->second())
		);
		return result;
	}

	X17_DEVICE void scale_(T scale) {
		this->set(
			this->first() * scale,
			this->second() * scale
		);
	}

	template<typename F>
	X17_DEVICE void elemwise_(F const &fn) {
		this->set(
			fn(this->first()),
			fn(this->second())
		);
	}

	X17_DEVICE void transpose_() requires(sizeof(T) == 4) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize row = tid / 4;
		usize col_pair = tid % 4;

		usize src_lane0 = (2 * col_pair + 0) * 4 + (row / 2);
		usize src_lane1 = (2 * col_pair + 1) * 4 + (row / 2);
		bool take_second = (row & 1) != 0;

		T src00 = shuffle_sync(this->val0, int(src_lane0));
		T src01 = shuffle_sync(this->val1, int(src_lane0));
		T src10 = shuffle_sync(this->val0, int(src_lane1));
		T src11 = shuffle_sync(this->val1, int(src_lane1));

		this->set(
			take_second ? src01 : src00,
			take_second ? src11 : src10
		);
	}
};

template<typename T>
struct Fragment_16x16 {
	Fragment_8x8<T> sub[2][2];

	X17_DEVICE void zero_() {
		sub[0][0].zero_();
		sub[0][1].zero_();
		sub[1][0].zero_();
		sub[1][1].zero_();
	}

	X17_DEVICE void fill_(T v) {
		sub[0][0].set(v, v);
		sub[0][1].set(v, v);
		sub[1][0].set(v, v);
		sub[1][1].set(v, v);
	}

	X17_DEVICE void scale_(T scale) {
		sub[0][0].scale_(scale);
		sub[0][1].scale_(scale);
		sub[1][0].scale_(scale);
		sub[1][1].scale_(scale);
	}

	X17_DEVICE void scale_(T top, T bot) {
		sub[0][0].scale_(top);
		sub[0][1].scale_(top);
		sub[1][0].scale_(bot);
		sub[1][1].scale_(bot);
	}

	X17_DEVICE void scale_top_(T top) {
		sub[0][0].scale_(top);
		sub[0][1].scale_(top);
	}

	X17_DEVICE void scale_bottom_(T bot) {
		sub[1][0].scale_(bot);
		sub[1][1].scale_(bot);
	}

	template<typename F>
	X17_DEVICE void elemwise_(F const &fn) {
		sub[0][0].elemwise_(fn);
		sub[0][1].elemwise_(fn);
		sub[1][0].elemwise_(fn);
		sub[1][1].elemwise_(fn);
	}

	template<typename F>
	X17_DEVICE void elemwise_top_(F const &fn) {
		sub[0][0].elemwise_(fn);
		sub[0][1].elemwise_(fn);
	}

	template<typename F>
	X17_DEVICE void elemwise_bot_(F const &fn) {
		sub[1][0].elemwise_(fn);
		sub[1][1].elemwise_(fn);
	}

	X17_DEVICE void acc_(const Fragment_16x16 &o) {
		sub[0][0].val0 += o.sub[0][0].val0;
		sub[0][0].val1 += o.sub[0][0].val1;
		sub[0][1].val0 += o.sub[0][1].val0;
		sub[0][1].val1 += o.sub[0][1].val1;
		sub[1][0].val0 += o.sub[1][0].val0;
		sub[1][0].val1 += o.sub[1][0].val1;
		sub[1][1].val0 += o.sub[1][1].val0;
		sub[1][1].val1 += o.sub[1][1].val1;
	}

	X17_DEVICE void transpose_() requires(sizeof(T) == 2) {
		sm80::movmatrix(sub[0][0].val, sub[0][0].val);
		Fragment_8x8<T> temp = sub[1][0];
		sm80::movmatrix(sub[0][1].val, sub[1][0].val);
		sm80::movmatrix(temp.val     , sub[0][1].val);
		sm80::movmatrix(sub[1][1].val, sub[1][1].val);
	}

	X17_DEVICE void transpose_() requires(sizeof(T) == 4) {
		sub[0][0].transpose_();
		sub[1][1].transpose_();
		Fragment_8x8<T> temp = sub[0][1];
		sub[0][1] = sub[1][0];
		sub[1][0] = temp;
		sub[0][1].transpose_();
		sub[1][0].transpose_();
	}
};

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
	usize tid = threadIdx.x % WARP_SIZE;

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

/// Stores 4 horizontally-adjacent 8x8 fragments (32 cols × 8 rows) to GMEM.
/// Uses shuffle_4x4 so each thread holds 16 contiguous bytes, then a single 128-bit store.
/// 4 threads per row × 16 bytes = 64B coalesced per row.
template<typename U, typename T, const usize M, const usize N>
requires(sizeof(U) == 2)
X17_DEVICE void store_1x4_8x8(
	GMatrix<U, M, N> const &dst,
	usize m_idx, usize n_idx,
	Fragment_8x8<T> const &f0,
	Fragment_8x8<T> const &f1,
	Fragment_8x8<T> const &f2,
	Fragment_8x8<T> const &f3
) {
	// Cast to output type (e.g., f32 → bf16)
	Fragment_8x8<U> g0, g1, g2, g3;
	g0.val = f0.template cast_reg<U>().val;
	g1.val = f1.template cast_reg<U>().val;
	g2.val = f2.template cast_reg<U>().val;
	g3.val = f3.template cast_reg<U>().val;

	// Rearrange so each thread holds 4 consecutive double-columns
	shuffle_4x4(g0.val, g1.val, g2.val, g3.val);

	// 128-bit store per thread, 64B coalesced per row
	usize tid = threadIdx.x % WARP_SIZE;
	u8 *base = reinterpret_cast<u8 *>(dst._ptr);
	usize stride = dst.stride_bytes();
	usize off = (m_idx + tid / 4) * stride + n_idx * usize(sizeof(U)) + (tid % 4) * 16;

	*reinterpret_cast<uint4 *>(base + off) = make_uint4(g0.val, g1.val, g2.val, g3.val);
}

/// Stores a 16x16 tile (2×2 grid of 8x8 fragments) to GMEM.
/// Uses shuffle_4x4 so each thread holds 16 contiguous bytes, then a single 128-bit store.
/// Threads 0,1,4,5,8,9,... write top 8 rows; threads 2,3,6,7,... write bottom 8 rows.
/// 2 threads per row × 16 bytes = 32B coalesced per row (full row for 16-col bf16).
template<typename U, typename T, const usize M, const usize N>
requires(sizeof(U) == 2)
X17_DEVICE void store_2x2_8x8(
	GMatrix<U, M, N> const &dst,
	usize m_idx, usize n_idx,
	Fragment_8x8<T> const &f0, Fragment_8x8<T> const &f1,
	Fragment_8x8<T> const &f2, Fragment_8x8<T> const &f3
) {
	// Cast to output type (e.g., f32 → bf16)
	u32 g0 = f0.template cast_reg<U>().val;
	u32 g1 = f1.template cast_reg<U>().val;
	u32 g2 = f2.template cast_reg<U>().val;
	u32 g3 = f3.template cast_reg<U>().val;

	// f0=top-left, f1=top-right, f2=bottom-left, f3=bottom-right
	// After shuffle_4x4:
	//   t%4==0: all of tl's row (cols 0-7)   → top row
	//   t%4==1: all of tr's row (cols 8-15)  → top row
	//   t%4==2: all of bl's row (cols 0-7)   → bottom row
	//   t%4==3: all of br's row (cols 8-15)  → bottom row
	shuffle_4x4(g0, g1, g2, g3);

	usize tid = threadIdx.x % WARP_SIZE;
	u8 *base = reinterpret_cast<u8 *>(dst._ptr);
	usize stride = dst.stride_bytes();
	usize row = (tid & 2) ? (m_idx + 8 + tid / 4) : (m_idx + tid / 4);
	usize col_off = n_idx * usize(sizeof(U)) + (tid & 1) * 16;

	*reinterpret_cast<uint4 *>(base + row * stride + col_off) = make_uint4(g0, g1, g2, g3);
}

/// Stores 8 horizontally-adjacent 8x8 fragments (64 cols × 8 rows) to GMEM.
/// shuffle_4x4 on left (f0-f3) and right (f4-f7) groups independently,
/// then XOR-4 shuffle merges them so 8 consecutive threads cover one row.
/// Each thread writes 2 × 16B stores; 8 threads × 16B = 128B coalesced per row.
template<typename U, typename T, const usize M, const usize N>
requires(sizeof(U) == 2)
X17_DEVICE void store_1x8_8x8(
	GMatrix<U, M, N> const &dst,
	usize m_idx, usize n_idx,
	Fragment_8x8<T> const &f0, Fragment_8x8<T> const &f1,
	Fragment_8x8<T> const &f2, Fragment_8x8<T> const &f3,
	Fragment_8x8<T> const &f4, Fragment_8x8<T> const &f5,
	Fragment_8x8<T> const &f6, Fragment_8x8<T> const &f7
) {
	// Cast to output type (e.g., f32 → bf16)
	u32 g0 = f0.template cast_reg<U>().val;
	u32 g1 = f1.template cast_reg<U>().val;
	u32 g2 = f2.template cast_reg<U>().val;
	u32 g3 = f3.template cast_reg<U>().val;
	u32 g4 = f4.template cast_reg<U>().val;
	u32 g5 = f5.template cast_reg<U>().val;
	u32 g6 = f6.template cast_reg<U>().val;
	u32 g7 = f7.template cast_reg<U>().val;

	// Rearrange left and right groups independently
	shuffle_4x4(g0, g1, g2, g3);
	shuffle_4x4(g4, g5, g6, g7);

	// XOR-4 shuffle: swap right group (g4-g7) of bit2=0 threads
	// with left group (g0-g3) of bit2=1 threads.
	// This makes 8 consecutive threads cover the same row.
	usize tid = threadIdx.x % WARP_SIZE;
	bool bit2 = (tid & 4) != 0;
	u32 recv;

	recv = shuffle_xor_sync(bit2 ? g0 : g4, 4);
	g0 = bit2 ? recv : g0;
	g4 = bit2 ? g4 : recv;

	recv = shuffle_xor_sync(bit2 ? g1 : g5, 4);
	g1 = bit2 ? recv : g1;
	g5 = bit2 ? g5 : recv;

	recv = shuffle_xor_sync(bit2 ? g2 : g6, 4);
	g2 = bit2 ? recv : g2;
	g6 = bit2 ? g6 : recv;

	recv = shuffle_xor_sync(bit2 ? g3 : g7, 4);
	g3 = bit2 ? recv : g3;
	g7 = bit2 ? g7 : recv;

	// g0-g3 = even row data, g4-g7 = odd row data
	// 8 consecutive threads × 16B = 128B coalesced per row
	u8 *base = reinterpret_cast<u8 *>(dst._ptr);
	usize stride = dst.stride_bytes();
	usize col_off = n_idx * usize(sizeof(U)) + (tid % 8) * 16;
	usize even_row = (m_idx + (tid / 8) * 2) * stride;
	usize odd_row = even_row + stride;

	*reinterpret_cast<uint4 *>(base + even_row + col_off) = make_uint4(g0, g1, g2, g3);
	*reinterpret_cast<uint4 *>(base + odd_row  + col_off) = make_uint4(g4, g5, g6, g7);
}

/// Phase 1 of loading 8 horizontally-adjacent 8x8 fragments (64 cols × 8 rows) from GMEM.
/// Performs 128-bit coalesced loads only; data remains in shuffled layout in the fragments.
/// Call load_unshuffle_1x8_8x8 later to rearrange into MMA register layout.
template<typename U, const usize M, const usize N>
requires(sizeof(U) == 2)
X17_DEVICE void load_shuffled_1x8_8x8(
	GMatrix<U, M, N> const &src,
	usize m_idx, usize n_idx,
	Fragment_8x8<U> &f0, Fragment_8x8<U> &f1,
	Fragment_8x8<U> &f2, Fragment_8x8<U> &f3,
	Fragment_8x8<U> &f4, Fragment_8x8<U> &f5,
	Fragment_8x8<U> &f6, Fragment_8x8<U> &f7
) {
	// 128-bit load per thread, 8 threads per row, 128B coalesced per row
	usize tid = threadIdx.x % WARP_SIZE;
	u8 *base = reinterpret_cast<u8 *>(src._ptr);
	usize stride = src.stride_bytes();
	usize col_off = n_idx * usize(sizeof(U)) + (tid % 8) * 16;
	usize even_row = (m_idx + (tid / 8) * 2) * stride;
	usize odd_row = even_row + stride;

	uint4 even_data = *reinterpret_cast<uint4 const *>(base + even_row + col_off);
	uint4 odd_data  = *reinterpret_cast<uint4 const *>(base + odd_row  + col_off);

	f0.val = even_data.x; f1.val = even_data.y; f2.val = even_data.z; f3.val = even_data.w;
	f4.val = odd_data.x;  f5.val = odd_data.y;  f6.val = odd_data.z;  f7.val = odd_data.w;
}

/// Phase 2 of loading 8 horizontally-adjacent 8x8 fragments.
/// Reverses the shuffled layout from load_shuffled_1x8_8x8 back to MMA register layout.
/// XOR-4 and shuffle_4x4 are self-inverse, so the same operations as store undo the rearrangement.
template<typename U>
requires(sizeof(U) == 2)
X17_DEVICE void load_unshuffle_1x8_8x8(
	Fragment_8x8<U> &f0, Fragment_8x8<U> &f1,
	Fragment_8x8<U> &f2, Fragment_8x8<U> &f3,
	Fragment_8x8<U> &f4, Fragment_8x8<U> &f5,
	Fragment_8x8<U> &f6, Fragment_8x8<U> &f7
) {
	usize tid = threadIdx.x % WARP_SIZE;
	bool bit2 = (tid & 4) != 0;
	u32 recv;

	recv = shuffle_xor_sync(bit2 ? f0.val : f4.val, 4);
	f0.val = bit2 ? recv : f0.val;
	f4.val = bit2 ? f4.val : recv;

	recv = shuffle_xor_sync(bit2 ? f1.val : f5.val, 4);
	f1.val = bit2 ? recv : f1.val;
	f5.val = bit2 ? f5.val : recv;

	recv = shuffle_xor_sync(bit2 ? f2.val : f6.val, 4);
	f2.val = bit2 ? recv : f2.val;
	f6.val = bit2 ? f6.val : recv;

	recv = shuffle_xor_sync(bit2 ? f3.val : f7.val, 4);
	f3.val = bit2 ? recv : f3.val;
	f7.val = bit2 ? f7.val : recv;

	shuffle_4x4(f0.val, f1.val, f2.val, f3.val);
	shuffle_4x4(f4.val, f5.val, f6.val, f7.val);
}

/// Loads K horizontally-adjacent 16x16 tiles from GMEM in shuffled layout.
/// K must be divisible by 4. Call load_unshuffle() to finalize.
template<typename U, const usize M, const usize N, const usize K>
requires(sizeof(U) == 2 && K % 4 == 0)
X17_DEVICE void load_shuffled(
	Fragment_16x16<U> (&tiles)[K],
	GMatrix<U, M, N> const &src,
	usize m_idx, usize n_idx
) {
	X17_UNROLL for (usize i = 0; i < K; i += 4) {
		load_shuffled_1x8_8x8(src, m_idx, n_idx + i*16,
			tiles[i].sub[0][0], tiles[i].sub[0][1],
			tiles[i+1].sub[0][0], tiles[i+1].sub[0][1],
			tiles[i+2].sub[0][0], tiles[i+2].sub[0][1],
			tiles[i+3].sub[0][0], tiles[i+3].sub[0][1]);
		load_shuffled_1x8_8x8(src, m_idx + 8, n_idx + i*16,
			tiles[i].sub[1][0], tiles[i].sub[1][1],
			tiles[i+1].sub[1][0], tiles[i+1].sub[1][1],
			tiles[i+2].sub[1][0], tiles[i+2].sub[1][1],
			tiles[i+3].sub[1][0], tiles[i+3].sub[1][1]);
	}
}

/// Unshuffles K horizontally-adjacent 16x16 tiles previously loaded with load_shuffled().
/// K must be divisible by 4.
template<typename U, const usize K>
requires(sizeof(U) == 2 && K % 4 == 0)
X17_DEVICE void load_unshuffle(
	Fragment_16x16<U> (&tiles)[K]
) {
	X17_UNROLL for (usize i = 0; i < K; i += 4) {
		load_unshuffle_1x8_8x8(
			tiles[i].sub[0][0], tiles[i].sub[0][1],
			tiles[i+1].sub[0][0], tiles[i+1].sub[0][1],
			tiles[i+2].sub[0][0], tiles[i+2].sub[0][1],
			tiles[i+3].sub[0][0], tiles[i+3].sub[0][1]);
		load_unshuffle_1x8_8x8(
			tiles[i].sub[1][0], tiles[i].sub[1][1],
			tiles[i+1].sub[1][0], tiles[i+1].sub[1][1],
			tiles[i+2].sub[1][0], tiles[i+2].sub[1][1],
			tiles[i+3].sub[1][0], tiles[i+3].sub[1][1]);
	}
}

/// Generic store for an array of K horizontally-adjacent 16x16 tiles.
/// Dispatches to store_1x8_8x8 (4 tiles), store_1x4_8x8 (2 tiles), store_2x2_8x8 (1 tile).
template<typename U, typename T, const usize M, const usize N, const usize K>
requires(sizeof(U) == 2)
X17_DEVICE void store(
	Fragment_16x16<T> const (&tiles)[K],
	GMatrix<U, M, N> const &dst,
	usize m_idx, usize n_idx
) {
	usize i = 0;
	if constexpr (K >= 4) {
		X17_UNROLL for (; i + 4 <= K; i += 4) {
			store_1x8_8x8(dst, m_idx, n_idx + i*16,
				tiles[i].sub[0][0], tiles[i].sub[0][1],
				tiles[i+1].sub[0][0], tiles[i+1].sub[0][1],
				tiles[i+2].sub[0][0], tiles[i+2].sub[0][1],
				tiles[i+3].sub[0][0], tiles[i+3].sub[0][1]);
			store_1x8_8x8(dst, m_idx + 8, n_idx + i*16,
				tiles[i].sub[1][0], tiles[i].sub[1][1],
				tiles[i+1].sub[1][0], tiles[i+1].sub[1][1],
				tiles[i+2].sub[1][0], tiles[i+2].sub[1][1],
				tiles[i+3].sub[1][0], tiles[i+3].sub[1][1]);
		}
	}
	if constexpr (K % 4 >= 2) {
		store_1x4_8x8(dst, m_idx, n_idx + i*16,
			tiles[i].sub[0][0], tiles[i].sub[0][1],
			tiles[i+1].sub[0][0], tiles[i+1].sub[0][1]);
		store_1x4_8x8(dst, m_idx + 8, n_idx + i*16,
			tiles[i].sub[1][0], tiles[i].sub[1][1],
			tiles[i+1].sub[1][0], tiles[i+1].sub[1][1]);
		i += 2;
	}
	if constexpr (K % 2 == 1) {
		store_2x2_8x8(dst, m_idx, n_idx + i*16,
			tiles[i].sub[0][0], tiles[i].sub[0][1],
			tiles[i].sub[1][0], tiles[i].sub[1][1]);
	}
}

template<typename U, typename T, const usize M, const usize N>
requires(sizeof(U) == 2)
X17_DEVICE void store2(
	GMatrix<U, M, N> const &dst,
	usize m_idx, usize n_idx,
	Fragment_16x16<T> const &f0,
	Fragment_16x16<T> const &f1
) {
	usize tid = threadIdx.x % WARP_SIZE;
	bool bit2 = (tid & 4) != 0;
	bool bit3 = (tid & 8) != 0;
	u32 recv;

	u32 tl0 = f0.sub[0][0].template cast_reg<U>().val;
	u32 tr0 = f0.sub[0][1].template cast_reg<U>().val;
	u32 bl0 = f0.sub[1][0].template cast_reg<U>().val;
	u32 br0 = f0.sub[1][1].template cast_reg<U>().val;

	recv = shuffle_xor_sync(bit2 ? tl0 : tr0, 4);
	tl0 = bit2 ? recv : tl0;
	tr0 = bit2 ? tr0 : recv;

	recv = shuffle_xor_sync(bit2 ? bl0 : br0, 4);
	bl0 = bit2 ? recv : bl0;
	br0 = bit2 ? br0 : recv;

	//---

	u32 tl1 = f1.sub[0][0].template cast_reg<U>().val;
	u32 tr1 = f1.sub[0][1].template cast_reg<U>().val;
	u32 bl1 = f1.sub[1][0].template cast_reg<U>().val;
	u32 br1 = f1.sub[1][1].template cast_reg<U>().val;

	recv = shuffle_xor_sync(bit2 ? tl1 : tr1, 4);
	tl1 = bit2 ? recv : tl1;
	tr1 = bit2 ? tr1 : recv;

	recv = shuffle_xor_sync(bit2 ? bl1 : br1, 4);
	bl1 = bit2 ? recv : bl1;
	br1 = bit2 ? br1 : recv;

	//---

	recv = shuffle_xor_sync(bit3 ? tl0 : tl1, 8);
	tl0 = bit3 ? recv : tl0;
	tl1 = bit3 ? tl1 : recv;

	recv = shuffle_xor_sync(bit3 ? tr0 : tr1, 8);
	tr0 = bit3 ? recv : tr0;
	tr1 = bit3 ? tr1 : recv;

	recv = shuffle_xor_sync(bit3 ? bl0 : bl1, 8);
	bl0 = bit3 ? recv : bl0;
	bl1 = bit3 ? bl1 : recv;

	recv = shuffle_xor_sync(bit3 ? br0 : br1, 8);
	br0 = bit3 ? recv : br0;
	br1 = bit3 ? br1 : recv;

	//---

	u8 *top_base = reinterpret_cast<u8 *>(dst.template tile_m<16>(m_idx / 16)._ptr);
	usize stride = dst.stride_bytes();
	usize col_off = n_idx * usize(sizeof(U)) + (tid % 16) * 4;
	usize row_base = (tid / 16) * 4 * stride;

	*reinterpret_cast<u32 *>(top_base + row_base + 0*stride + col_off) = tl0;
	*reinterpret_cast<u32 *>(top_base + row_base + 1*stride + col_off) = tr0;
	*reinterpret_cast<u32 *>(top_base + row_base + 2*stride + col_off) = tl1;
	*reinterpret_cast<u32 *>(top_base + row_base + 3*stride + col_off) = tr1;

	u8 *bot_base = top_base + 8 * stride;
	*reinterpret_cast<u32 *>(bot_base + row_base + 0*stride + col_off) = bl0;
	*reinterpret_cast<u32 *>(bot_base + row_base + 1*stride + col_off) = br0;
	*reinterpret_cast<u32 *>(bot_base + row_base + 2*stride + col_off) = bl1;
	*reinterpret_cast<u32 *>(bot_base + row_base + 3*stride + col_off) = br1;
}


template<typename F, typename T>
X17_DEVICE void cast(Fragment_16x16<F> const &src, Fragment_16x16<T> &dst) {
	X17_UNROLL for (usize j = 0; j < 2; j++) {
		X17_UNROLL for (usize i = 0; i < 2; i++) {
			dst.sub[j][i].set(
				static_cast<T>(src.sub[j][i].first()),
				static_cast<T>(src.sub[j][i].second())
			);
		}
	}
}

template<typename T>
X17_DEVICE void zero_(Fragment_16x16<T> &f) {
	f.zero_();
}

template<typename T, const usize K>
X17_DEVICE void zero_(T (&arr)[K]) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		zero_(arr[i]);
	}
}

template<typename... T>
X17_DEVICE void zero_(T&... args) {
	(zero_(args), ...);
}

template<typename T>
X17_DEVICE void fill_(Fragment_16x16<T> &f, T v) {
	f.fill_(v);
}

template<typename T, const usize K>
X17_DEVICE void fill_(T (&arr)[K], T v) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		fill_(arr[i], v);
	}
}

template<typename... T, typename S>
X17_DEVICE void fill_(T&... args, S v) {
	(fill_(args, v), ...);
}

template<typename T>
X17_DEVICE void scale_(Fragment_16x16<T> &f, T s) {
	f.scale_(s);
}

template<typename T, typename F>
X17_DEVICE void elemwise_(Fragment_16x16<T> &f, F const &fn) {
	f.elemwise_(fn);
}

template<typename T, const usize K, typename F>
X17_DEVICE void elemwise_(T (&arr)[K], F const &fn) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		elemwise_(arr[i], fn);
	}
}

template<typename T, const usize K>
X17_DEVICE void scale_(T (&arr)[K], T s) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		scale_(arr[i], s);
	}
}

template<typename... T, typename S>
X17_DEVICE void scale_(T&... args, S s) {
	(scale_(args, s), ...);
}

template<typename T>
X17_DEVICE void scale_top_(Fragment_16x16<T> &f, T s) {
	f.scale_top_(s);
}

template<typename T, typename F>
X17_DEVICE void elemwise_top_(Fragment_16x16<T> &f, F const &fn) {
	f.elemwise_top_(fn);
}

template<typename T, const usize K, typename F>
X17_DEVICE void elemwise_top_(Fragment_16x16<T> (&arr)[K], F const &fn) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		arr[i].elemwise_top_(fn);
	}
}

template<typename T, const usize K>
X17_DEVICE void scale_top_(Fragment_16x16<T> (&arr)[K], T s) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		arr[i].scale_top_(s);
	}
}

template<typename... T, typename S>
X17_DEVICE void scale_top_(T&... args, S s) {
	(scale_top_(args, s), ...);
}

template<typename T>
X17_DEVICE void scale_bottom_(Fragment_16x16<T> &f, T s) {
	f.scale_bottom_(s);
}

template<typename T, typename F>
X17_DEVICE void elemwise_bot_(Fragment_16x16<T> &f, F const &fn) {
	f.elemwise_bot_(fn);
}

template<typename T, const usize K, typename F>
X17_DEVICE void elemwise_bot_(Fragment_16x16<T> (&arr)[K], F const &fn) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		arr[i].elemwise_bot_(fn);
	}
}

template<typename T, const usize K>
X17_DEVICE void scale_bottom_(Fragment_16x16<T> (&arr)[K], T s) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		arr[i].scale_bottom_(s);
	}
}

template<typename... T, typename S>
X17_DEVICE void scale_bottom_(T&... args, S s) {
	(scale_bottom_(args, s), ...);
}

template<typename T, const usize K>
X17_DEVICE void acc_(Fragment_16x16<T> (&dst)[K], Fragment_16x16<T> const (&src)[K]) {
	X17_UNROLL for (usize i = 0; i < K; i++) {
		dst[i].acc_(src[i]);
	}
}

//--------------------------------------------------------------------------------------------------

struct SoftmaxStats {
	f32 sum;
	f32 max;

	X17_DEVICE SoftmaxStats() {
		sum = 0.0f;
		max = -std::numeric_limits<f32>::infinity();
	}

	X17_DEVICE void store_shared(u32 ptr) const {
		asm volatile(
			"st.shared.v2.f32 [%0], {%1, %2};\n"
			:
			: "r"(ptr), "f"(sum), "f"(max)
			: "memory"
		);
	}

	X17_DEVICE void load_shared(u32 ptr) {
		asm volatile(
			"ld.shared.v2.f32 {%0, %1}, [%2];\n"
			: "=f"(sum), "=f"(max)
			: "r"(ptr)
			: "memory"
		);
	}
};

//--------------------------------------------------------------------------------------------------

template<typename T, const usize M, const usize N>
requires(
	M > 0 && M % 16 == 0
	&& N > 0 && N % 16 == 0
)
struct RMatrix {
	Fragment_16x16<T> tiles[M / 16][N / 16];

	X17_DEVICE constexpr usize m_rows() const {
		return M;
	}

	X17_DEVICE constexpr usize n_cols() const {
		return N;
	}

	X17_DEVICE constexpr usize elems() const {
		return M * N;
	}

	X17_DEVICE void zero_() {
		X17_UNROLL for (usize j = 0; j < M / 16; j++) {
			X17_UNROLL for (usize i = 0; i < N / 16; i++) {
				tiles[j][i].zero_();
			}
		}
	}
};

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

template<
	typename T,
	const usize M,
	const usize N
>
requires(
	sizeof(T) == 2
	&& M >= 0 && M % 16 == 0
	&& N * sizeof(T) % 128 == 0
)
struct SMatrix {
	u32 _ptr;

	X17_DEVICE constexpr SMatrix() : _ptr(0) {}

	X17_DEVICE constexpr SMatrix(void *ptr): SMatrix(cast_smem_ptr_to_uint(ptr)) {}

	X17_DEVICE constexpr SMatrix(u32 ptr): _ptr(ptr) {}

	X17_DEVICE constexpr usize m_rows() const { return M; }
	X17_DEVICE constexpr usize n_cols() const { return N; }
	X17_DEVICE constexpr usize elems() const { return M * N; }
	X17_DEVICE constexpr usize bytes() const { return M * N * sizeof(T); }

	constexpr static usize ROW_BYTES = N * sizeof(T);

	template<const usize TILE_M>
	requires(TILE_M > 0 && TILE_M % 8 == 0 && M % TILE_M == 0)
	X17_DEVICE constexpr SMatrix<T, TILE_M, N> tile_m(usize tile_idx) const {
		return SMatrix<T, TILE_M, N>{
			_ptr + (tile_idx * TILE_M * ROW_BYTES)
		};
	}

	/// Copy from a (possibly smaller) GMEM matrix into a sub-region of this SMEM matrix.
	/// Data is placed starting at (dst_row, dst_col) within this SMEM matrix.
	/// dst_row and dst_col must be multiples of 16.
	template<const usize THREADS_PER_BLOCK, const usize GM, const usize GN>
	requires(GM <= M && GN <= N)
	X17_DEVICE void cp_async_from(usize tid, GMatrix<T, GM, GN> src, usize dst_row, usize dst_col) const {
		if constexpr (GN > 0 && GM > 0) {
			__builtin_assume(tid < THREADS_PER_BLOCK);

			constexpr usize GMEM_ROW_BYTES = GN * sizeof(T);
			static_assert(GMEM_ROW_BYTES % 16 == 0);

			constexpr usize CP_PER_ROW = GMEM_ROW_BYTES / 16;
			static_assert(THREADS_PER_BLOCK % CP_PER_ROW == 0);
			constexpr usize ROWS_PER_STEP = THREADS_PER_BLOCK / CP_PER_ROW;
			constexpr usize STEPS = GM / ROWS_PER_STEP;

			if constexpr (STEPS == 0) {
				if constexpr (GM % ROWS_PER_STEP == 0) {
					return;
				}
				if (tid >= (GM % ROWS_PER_STEP) * CP_PER_ROW) {
					return;
				}
			}

			// Thread's position within a step is fixed
			usize col_in_row = dst_col * sizeof(T) + (tid % CP_PER_ROW) * 16;
			usize gmem_col = (tid % CP_PER_ROW) * 16;
			usize row_in_step = tid / CP_PER_ROW;

			constexpr usize REPEAT_AFTER = least_common_multiple(8, ROWS_PER_STEP) / ROWS_PER_STEP;
			usize off[REPEAT_AFTER];
			X17_UNROLL for (usize i = 0; i < REPEAT_AFTER; i++) {
				usize row = dst_row + i * ROWS_PER_STEP + row_in_step;
				off[i] = col_in_row ^ ((row & 7) << 4);
			}

			u8 const *src_ptr =
				reinterpret_cast<u8 const *>(src._ptr)
				+ row_in_step * src.stride_bytes()
				+ gmem_col;
			usize src_step = ROWS_PER_STEP * src.stride_bytes();

			usize dst_ptr = _ptr + (dst_row + row_in_step) * ROW_BYTES;
			usize dst_step = ROWS_PER_STEP * ROW_BYTES;

			if constexpr (STEPS > 0) {
				X17_UNROLL for (usize step = 0; step < STEPS; step++) {
					sm80::cp_async(src_ptr, dst_ptr + off[step % REPEAT_AFTER]);
					src_ptr += src_step;
					dst_ptr += dst_step;
				}
			}
			if constexpr (GM % ROWS_PER_STEP != 0) {
				usize step = STEPS;
				if (tid < (GM % ROWS_PER_STEP) * CP_PER_ROW) {
					sm80::cp_async(src_ptr, dst_ptr + off[step % REPEAT_AFTER]);
				}
			}
		}
	}

	/// Both `m_idx` and `n_idx` must be multiples of 16.
	X17_DEVICE void load_tile_to_fragment(
		usize m_idx, usize n_idx,
		Fragment_16x16<T> &dst
	) const requires(sizeof(T) == 2) {
		if constexpr (N > 0) {
			usize tid = threadIdx.x;
			usize row = m_idx + (tid & 15);
			usize swizzle = ((threadIdx.x & 7) << 4) ^ (threadIdx.x & 16);
			usize byte_col = n_idx * sizeof(T);
			u32 addr = _ptr + (row * ROW_BYTES) + (byte_col ^ swizzle);

			sm80::ldmatrix_8x8xu16_x4(
				addr,
				dst.sub[0][0].val, dst.sub[1][0].val, dst.sub[0][1].val, dst.sub[1][1].val
			);
		}
	}

	/// Loads a 16x16 tile from SMEM, transposed, into MMA registers.
	/// Uses ldmatrix.trans to transpose each 8x8 sub-tile during load,
	/// plus swaps off-diagonal destinations for a full 16x16 transpose.
	X17_DEVICE void load_tile_to_fragment_trans(
		usize m_idx, usize n_idx,
		Fragment_16x16<T> &dst
	) const requires(sizeof(T) == 2) {
		if constexpr (N > 0) {
			usize tid = threadIdx.x;
			usize row = m_idx + (tid & 15);
			usize swizzle = ((threadIdx.x & 7) << 4) ^ (threadIdx.x & 16);
			usize byte_col = n_idx * sizeof(T);
			u32 addr = _ptr + (row * ROW_BYTES) + (byte_col ^ swizzle);

			sm80::ldmatrix_t_8x8xu16_x4(
				addr,
				dst.sub[0][0].val, dst.sub[0][1].val, dst.sub[1][0].val, dst.sub[1][1].val
			);
		}
	}

};

template<
	typename T,
	const usize M,
	const usize N
>
requires(sizeof(T) == 4 && M >= 0)
struct SMatrix_32b {
	u32 _ptr;

	X17_DEVICE constexpr SMatrix_32b() : _ptr(0) {}

	X17_DEVICE constexpr SMatrix_32b(void *ptr): SMatrix_32b(cast_smem_ptr_to_uint(ptr)) {}

	X17_DEVICE constexpr SMatrix_32b(u32 ptr): _ptr(ptr) {}

	X17_DEVICE constexpr usize m_rows() const { return M; }
	X17_DEVICE constexpr usize n_cols() const { return N; }
	X17_DEVICE constexpr usize elems() const { return M * N; }
	X17_DEVICE constexpr usize bytes() const { return M * N * sizeof(T); }

	constexpr static usize ROW_BYTES = N * sizeof(T);

	template<const usize TILE_M>
	requires(TILE_M > 0 && M % TILE_M == 0)
	X17_DEVICE constexpr SMatrix_32b<T, TILE_M, N> tile_m(usize tile_idx) const {
		return SMatrix_32b<T, TILE_M, N>{
			_ptr + (tile_idx * TILE_M * ROW_BYTES)
		};
	}
};

// Standalone tile_m: avoids `.template tile_m<>()` in dependent contexts.
template<const usize TILE_M, typename Mat>
X17_HOST_DEVICE constexpr auto tile_m(Mat mat, usize tile_idx)
	-> decltype(mat.template tile_m<TILE_M>(tile_idx))
{
	return mat.template tile_m<TILE_M>(tile_idx);
}

template<
	const usize THREADS_PER_BLOCK,
	typename T,
	const usize SM, const usize SN,
	const usize GM, const usize GN
>
X17_DEVICE void cp_async_gmem_to_smem(
	usize tid,
	GMatrix<T, GM, GN> src,
	SMatrix<T, SM, SN> dst,
	usize dst_row = 0,
	usize dst_col = 0
) {
	dst.template cp_async_from<THREADS_PER_BLOCK>(tid, src, dst_row, dst_col);
}

template<
	const usize THREADS_PER_BLOCK,
	typename T,
	const usize SM,
	const usize SN,
	const usize GN
>
requires(sizeof(T) == 4 && GN * sizeof(T) % 16 == 0)
X17_DEVICE void cp_async_gmem_to_smem(
	usize tid,
	GMatrix<T, 1, GN> src,
	SMatrix_32b<T, SM, SN> dst,
	usize dst_row = 0,
	usize dst_col = 0
) {
	__builtin_assume(tid < THREADS_PER_BLOCK);

	constexpr usize CP_PER_ROW = GN * sizeof(T) / 16;
	if (tid < CP_PER_ROW) {
		u8 const *src_ptr = reinterpret_cast<u8 const *>(src._ptr) + tid * sizeof(u128);
		u32 dst_ptr = dst._ptr + dst_row * dst.ROW_BYTES + dst_col * sizeof(T) + tid * sizeof(u128);
		sm80::cp_async(reinterpret_cast<u128 const *>(src_ptr), dst_ptr);
	}
}

template<
	typename T,
	const usize M,
	const usize N
>
X17_DEVICE void smem_tile_to_fragment(
	SMatrix<T, M, N> const &src,
	usize m_idx, usize n_idx,
	Fragment_16x16<T> &dst
) {
	src.load_tile_to_fragment(m_idx, n_idx, dst);
}

template<
	typename T,
	const usize M,
	const usize N
>
X17_DEVICE void smem_tile_to_fragment_trans(
	SMatrix<T, M, N> const &src,
	usize m_idx, usize n_idx,
	Fragment_16x16<T> &dst
) {
	src.load_tile_to_fragment_trans(m_idx, n_idx, dst);
}

using sm80::cp_async_commit;
using sm80::cp_async_wait;
using sm80::bar_sync;

//--------------------------------------------------------------------------------------------------

template<const usize M, const usize N, const usize K>
requires(M == 16 && N == K * 16)
X17_DEVICE void fragments_to_smem(
	Fragment_16x16<f32> const (&src)[K],
	SMatrix_32b<f32, M, N> const &dst
) {
	usize tid = threadIdx.x % WARP_SIZE;
	constexpr u32 TILE_STRIDE = 2 * WARP_SIZE * 4 * sizeof(f32); // 1024 bytes per 16x16 f32 tile

	X17_UNROLL for (usize i = 0; i < K; i++) {
		u32 base = dst._ptr + i * TILE_STRIDE;
		u32 p0 = base + tid * 4 * sizeof(f32);
		u32 p1 = p0 + WARP_SIZE * 4 * sizeof(f32);

		store_shared_4x32b(
			p0,
			src[i].sub[0][0].val0, src[i].sub[0][0].val1,
			src[i].sub[0][1].val0, src[i].sub[0][1].val1
		);
		store_shared_4x32b(
			p1,
			src[i].sub[1][0].val0, src[i].sub[1][0].val1,
			src[i].sub[1][1].val0, src[i].sub[1][1].val1
		);
	}
}

/// Loads MMA-layout f32 fragments from shared memory (inverse of fragments_to_smem).
template<const usize M, const usize N, const usize K>
requires(M == 16 && N == K * 16)
X17_DEVICE void smem_to_fragments(
	Fragment_16x16<f32> (&dst)[K],
	SMatrix_32b<f32, M, N> const &src
) {
	usize tid = threadIdx.x % WARP_SIZE;
	constexpr u32 TILE_STRIDE = 2 * WARP_SIZE * 4 * sizeof(f32); // 1024 bytes per 16x16 f32 tile
	X17_UNROLL for (usize i = 0; i < K; i++) {
		u32 base = src._ptr + i * TILE_STRIDE;
		u32 p0 = base + tid * 4 * sizeof(f32);
		u32 p1 = p0 + WARP_SIZE * 4 * sizeof(f32);

		load_shared_4x32b(
			p0,
			dst[i].sub[0][0].val0, dst[i].sub[0][0].val1,
			dst[i].sub[0][1].val0, dst[i].sub[0][1].val1
		);
		load_shared_4x32b(
			p1,
			dst[i].sub[1][0].val0, dst[i].sub[1][0].val1,
			dst[i].sub[1][1].val0, dst[i].sub[1][1].val1
		);
	}
}

//--------------------------------------------------------------------------------------------------

X17_DEVICE void mma_a_bt(
	Fragment_16x16<bf16> const &a,
	Fragment_16x16<bf16> const &b,
	Fragment_16x16<f32> &c
) {
    sm80::mma_bf16_f32(
		c.sub[0][0].val0, c.sub[0][0].val1, c.sub[1][0].val0, c.sub[1][0].val1,
		a.sub[0][0].val , a.sub[1][0].val , a.sub[0][1].val , a.sub[1][1].val ,
		b.sub[0][0].val , b.sub[0][1].val ,
		c.sub[0][0].val0, c.sub[0][0].val1, c.sub[1][0].val0, c.sub[1][0].val1
	);
    sm80::mma_bf16_f32(
		c.sub[0][1].val0, c.sub[0][1].val1, c.sub[1][1].val0, c.sub[1][1].val1,
		a.sub[0][0].val , a.sub[1][0].val , a.sub[0][1].val , a.sub[1][1].val ,
		b.sub[1][0].val , b.sub[1][1].val ,
		c.sub[0][1].val0, c.sub[0][1].val1, c.sub[1][1].val0, c.sub[1][1].val1
	);
}

//--------------------------------------------------------------------------------------------------
