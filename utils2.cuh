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
			f32 result;
			asm ("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
			return result;
		}

		/// Calculate `logb(x)` where `b` is our underlying base.
		/// The underlying base was chosen to be fast and may change in the future.
		///
		/// To calculate `logB(x)` for some other base `B`, use `logb(x) / logb(B)`.
		X17_DEVICE f32 logb(f32 x) {
			f32 result;
			asm ("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
			return result;
		}

		/// Single-instruction reciprocal
		/// Precision: <= 1 ULP, round-to-nearest
		X17_DEVICE f32 recip(f32 x) {
			return __frcp_rn(x);
		}
	}

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

	X17_DEVICE f32 fma(f32 a, f32 b, f32 c) {
		return __fmaf_rn(a, b, c);
	}
}

//--------------------------------------------------------------------------------------------------

X17_DEVICE bool any_sync(bool predicate) {
	return __any_sync(0xffffffff, predicate);
}

X17_DEVICE bool all_sync(bool predicate) {
	return __all_sync(0xffffffff, predicate);
}

X17_DEVICE f32 shfl_xor_sync(f32 val, int lane_mask) {
	return __shfl_xor_sync(0xffffffff, val, lane_mask);
}

X17_DEVICE void store_shared_4(u32 ptr, f32 a, f32 b, f32 c, f32 d) {
	asm volatile(
		"st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
		:
		: "r"(ptr), "f"(a), "f"(b), "f"(c), "f"(d)
		: "memory"
	);
}

X17_DEVICE void load_shared_4(u32 ptr, f32 &a, f32 &b, f32 &c, f32 &d) {
	asm volatile(
		"ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
		: "=f"(a), "=f"(b), "=f"(c), "=f"(d)
		: "r"(ptr)
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

	X17_DEVICE void mma_bf16_f32(
		f32       &d0, f32       &d1, f32       &d2, f32       &d3,
		u32 const &a0, u32 const &a1, u32 const &a2, u32 const &a3,
		u32 const &b0, u32 const &b1
	) {
		asm volatile(
			"\nmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.zero "
			"{%0,  %1,  %2,  %3},"
			"{%4,  %5,  %6,  %7},"
			"{%8,  %9};\n"
			:
				"=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
			:
				"r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
				"r"(b0),  "r"(b1)
		);
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

	X17_DEVICE void transpose_() {
		sm80::movmatrix(sub[0][0].val, sub[0][0].val);
		Fragment_8x8<T> temp = sub[1][0];
		sm80::movmatrix(sub[0][1].val, sub[1][0].val);
		sm80::movmatrix(temp.val     , sub[0][1].val);
		sm80::movmatrix(sub[1][1].val, sub[1][1].val);
	}

	/// Stores a 16x16 tile from MMA registers to GMEM with 32-byte coalesced writes.
	/// Even-row threads (tid & 4 == 0) write left 8 cols, odd-row threads write right 8 cols.
	/// Sender picks what to send so only 2 shuffles needed (top + bottom half).
	template<typename U, const usize M, const usize N>
	requires(sizeof(U) == 2)
	X17_DEVICE void store(GMatrix<U, M, N> const &dst, usize m_idx, usize n_idx) const {
		usize tid = threadIdx.x % WARP_SIZE;
		bool is_even = (tid & 4) == 0;

		// Cast all sub-fragments to output type
		u32 my_tl = sub[0][0].template cast_reg<U>().val;
		u32 my_tr = sub[0][1].template cast_reg<U>().val;
		u32 my_bl = sub[1][0].template cast_reg<U>().val;
		u32 my_br = sub[1][1].template cast_reg<U>().val;

		// Sender picks: even threads send tr (partner needs it), odd send tl
		u32 top_recv = __shfl_xor_sync(0xffffffff, is_even ? my_tr : my_tl, 4);
		u32 bot_recv = __shfl_xor_sync(0xffffffff, is_even ? my_br : my_bl, 4);

		my_tl = is_even ? my_tl : top_recv;
		my_tr = is_even ? top_recv : my_tr;

		my_bl = is_even ? my_bl : bot_recv;
		my_br = is_even ? bot_recv : my_br;

		u8 *top_base = reinterpret_cast<u8 *>(dst.template tile_m<16>(m_idx / 16)._ptr);
		usize stride = dst.stride_bytes();
		usize col_off = n_idx * usize(sizeof(U)) + (is_even ? 0 : 16) + (tid & 3) * 4;
		usize even_row = (tid >> 3) * 2 * stride;
		usize odd_row = even_row + stride;
		u8 *bot_base = top_base + 8 * stride;

		// Top half
		*reinterpret_cast<u32 *>(top_base + even_row + col_off) = my_tl;
		*reinterpret_cast<u32 *>(top_base + odd_row  + col_off) = my_tr;
		// Bottom half
		*reinterpret_cast<u32 *>(bot_base + even_row + col_off) = my_bl;
		*reinterpret_cast<u32 *>(bot_base + odd_row  + col_off) = my_br;
	}
/*
	/// Loads a 16x16 tile from GMEM directly into MMA-compatible register layout.
	/// m_idx and n_idx must be multiples of 16.
	template<const usize M, const usize N>
	X17_DEVICE void load(GMatrix<T, M, N> const &src, usize m_idx, usize n_idx) requires(sizeof(T) == 2) {
		usize tid = threadIdx.x;
		GMatrix<T, 16, N> src_tile = src.template tile_m<16>(m_idx / 16);
		usize src_off = n_idx * usize(sizeof(T));

		src_off += (tid & 0x1c) * (src.stride_bytes() / 4) + (tid & 3) * 4;
		sub[0][0].val = __ldg(reinterpret_cast<u32 const *>(
			reinterpret_cast<u8 const *>(src_tile._ptr) + src_off
		));

		src_off += 16;
		sub[0][1].val = __ldg(reinterpret_cast<u32 const *>(
			reinterpret_cast<u8 const *>(src_tile._ptr) + src_off
		));

		src_off += 8 * src.stride_bytes() - 16;
		sub[1][0].val = __ldg(reinterpret_cast<u32 const *>(
			reinterpret_cast<u8 const *>(src_tile._ptr) + src_off
		));

		src_off += 16;
		sub[1][1].val = __ldg(reinterpret_cast<u32 const *>(
			reinterpret_cast<u8 const *>(src_tile._ptr) + src_off
		));
	}
*/
	/// Loads two consecutive 16x16 tiles from GMEM with improved coalescing.
	/// f0 gets tile at (m_idx, n_idx), f1 gets tile at (m_idx, n_idx + 16).
	/// Thread mapping: tile = (tid>>2)&1, row_group = tid>>3, col = tid&3.
	/// Data is in scrambled layout; call shuffle_load2() before use.
	template<const usize M, const usize N>
	X17_DEVICE static void load2(
		Fragment_16x16<T> &f0, Fragment_16x16<T> &f1,
		GMatrix<T, M, N> const &src, usize m_idx, usize n_idx
	) requires(sizeof(T) == 2) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize tile = (tid >> 2) & 1;
		usize rg = tid >> 3;
		usize col = tid & 3;

		GMatrix<T, 16, N> src_tile = src.template tile_m<16>(m_idx / 16);
		u8 const *ptr = reinterpret_cast<u8 const *>(src_tile._ptr);
		usize stride = src.stride_bytes();
		usize base = (n_idx + tile * 16) * usize(sizeof(T)) + col * 4;
		usize even_off = base + rg * 2 * stride;
		usize odd_off = even_off + stride;
		usize bot = 8 * stride;

		// Even rows, top half
		f0.sub[0][0].val = __ldg(reinterpret_cast<u32 const *>(ptr + even_off));
		f0.sub[0][1].val = __ldg(reinterpret_cast<u32 const *>(ptr + even_off + 16));
		// Odd rows, top half
		f1.sub[0][0].val = __ldg(reinterpret_cast<u32 const *>(ptr + odd_off));
		f1.sub[0][1].val = __ldg(reinterpret_cast<u32 const *>(ptr + odd_off + 16));
		// Even rows, bottom half
		f0.sub[1][0].val = __ldg(reinterpret_cast<u32 const *>(ptr + even_off + bot));
		f0.sub[1][1].val = __ldg(reinterpret_cast<u32 const *>(ptr + even_off + bot + 16));
		// Odd rows, bottom half
		f1.sub[1][0].val = __ldg(reinterpret_cast<u32 const *>(ptr + odd_off + bot));
		f1.sub[1][1].val = __ldg(reinterpret_cast<u32 const *>(ptr + odd_off + bot + 16));
	}

	/// Shuffles two fragments loaded by load2() into correct MMA layout.
	/// After this, f0 = tile 0 data, f1 = tile 1 data.
	X17_DEVICE static void shuffle_load2(
		Fragment_16x16<T> &f0, Fragment_16x16<T> &f1
	) requires(sizeof(T) == 2) {
		usize tid = threadIdx.x % WARP_SIZE;
		bool is_tile1 = (tid >> 2) & 1;
		// bit2=0 threads: f0 has tile0 even rows, f1 has tile0 odd rows → keep f0, replace f1
		// bit2=1 threads: f0 has tile1 even rows, f1 has tile1 odd rows → replace f0, keep f1
		X17_UNROLL for (usize h = 0; h < 2; h++) {
			X17_UNROLL for (usize c = 0; c < 2; c++) {
				u32 send = is_tile1 ? f0.sub[h][c].val : f1.sub[h][c].val;
				u32 recv = __shfl_xor_sync(0xffffffff, send, 4);
				if (is_tile1) f0.sub[h][c].val = recv;
				else          f1.sub[h][c].val = recv;
			}
		}
	}
};

template<typename U, typename T, const usize M, const usize N>
requires(sizeof(U) == 2)
X17_DEVICE void store_x(
	GMatrix<U, M, N> const &dst,
	usize m_idx, usize n_idx,
	Fragment_16x16<T> f0,
	Fragment_16x16<T> f1
) {
	usize tid = threadIdx.x % WARP_SIZE;
	bool is_even = (tid & 4) == 0;

	u32 my_tl0 = f0.sub[0][0].template cast_reg<U>().val;
	u32 my_tr0 = f0.sub[0][1].template cast_reg<U>().val;
	u32 my_bl0 = f0.sub[1][0].template cast_reg<U>().val;
	u32 my_br0 = f0.sub[1][1].template cast_reg<U>().val;

	u32 top_recv0 = __shfl_xor_sync(0xffffffff, is_even ? my_tr0 : my_tl0, 4);
	u32 bot_recv0 = __shfl_xor_sync(0xffffffff, is_even ? my_br0 : my_bl0, 4);

	my_tl0 = is_even ? my_tl0 : top_recv0;
	my_tr0 = is_even ? top_recv0 : my_tr0;

	my_bl0 = is_even ? my_bl0 : bot_recv0;
	my_br0 = is_even ? bot_recv0 : my_br0;

	//---

	u32 my_tl1 = f1.sub[0][0].template cast_reg<U>().val;
	u32 my_tr1 = f1.sub[0][1].template cast_reg<U>().val;
	u32 my_bl1 = f1.sub[1][0].template cast_reg<U>().val;
	u32 my_br1 = f1.sub[1][1].template cast_reg<U>().val;

	u32 top_recv1 = __shfl_xor_sync(0xffffffff, is_even ? my_tr1 : my_tl1, 4);
	u32 bot_recv1 = __shfl_xor_sync(0xffffffff, is_even ? my_br1 : my_bl1, 4);

	my_tl1 = is_even ? my_tl1 : top_recv1;
	my_tr1 = is_even ? top_recv1 : my_tr1;

	my_bl1 = is_even ? my_bl1 : bot_recv1;
	my_br1 = is_even ? bot_recv1 : my_br1;

	//---

	u32 top_recv2 = __shfl_xor_sync(0xffffffff, (tid & 8) == 0 ? my_tl1 : my_tl0, 8);
	u32 bot_recv2 = __shfl_xor_sync(0xffffffff, (tid & 8) == 0 ? my_bl1 : my_bl0, 8);

	my_tl0 = (tid & 8) == 0 ? my_tl0 : top_recv2;
	my_bl0 = (tid & 8) == 0 ? my_bl0 : bot_recv2;

	my_tl1 = (tid & 8) == 0 ? my_tl1 : top_recv2;
	my_bl1 = (tid & 8) == 0 ? my_bl1 : bot_recv2;

	//---

}

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

	r1 = __shfl_xor_sync(0xffffffff, r1, 1);
	r2 = __shfl_xor_sync(0xffffffff, r2, 2);
	r3 = __shfl_xor_sync(0xffffffff, r3, 3);

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

	recv = __shfl_xor_sync(0xffffffff, bit2 ? g0 : g4, 4);
	g0 = bit2 ? recv : g0;
	g4 = bit2 ? g4 : recv;

	recv = __shfl_xor_sync(0xffffffff, bit2 ? g1 : g5, 4);
	g1 = bit2 ? recv : g1;
	g5 = bit2 ? g5 : recv;

	recv = __shfl_xor_sync(0xffffffff, bit2 ? g2 : g6, 4);
	g2 = bit2 ? recv : g2;
	g6 = bit2 ? g6 : recv;

	recv = __shfl_xor_sync(0xffffffff, bit2 ? g3 : g7, 4);
	g3 = bit2 ? recv : g3;
	g7 = bit2 ? g7 : recv;

	// g0-g3 = even row data, g4-g7 = odd row data
	// 8 consecutive threads × 16B = 128B coalesced per row
	u8 *base = reinterpret_cast<u8 *>(dst._ptr);
	usize stride = dst.stride_bytes();
	usize col_off = n_idx * usize(sizeof(U)) + (tid % 8) * 16;
	usize even_row = (m_idx + (tid >> 3) * 2) * stride;
	usize odd_row = even_row + stride;

	*reinterpret_cast<uint4 *>(base + even_row + col_off) = make_uint4(g0, g1, g2, g3);
	*reinterpret_cast<uint4 *>(base + odd_row  + col_off) = make_uint4(g4, g5, g6, g7);
}

/// Generic store for an array of K horizontally-adjacent 16x16 tiles.
/// Dispatches to store_1x8_8x8 (4 tiles), store_1x4_8x8 (2 tiles), store_2x2_8x8 (1 tile).
template<typename U, typename T, const usize M, const usize N, const usize K>
requires(sizeof(U) == 2)
X17_DEVICE void store(
	GMatrix<U, M, N> const &dst,
	usize m_idx, usize n_idx,
	Fragment_16x16<T> const (&tiles)[K]
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

	recv = __shfl_xor_sync(0xffffffff, bit2 ? tl0 : tr0, 4);
	tl0 = bit2 ? recv : tl0;
	tr0 = bit2 ? tr0 : recv;

	recv = __shfl_xor_sync(0xffffffff, bit2 ? bl0 : br0, 4);
	bl0 = bit2 ? recv : bl0;
	br0 = bit2 ? br0 : recv;

	//---

	u32 tl1 = f1.sub[0][0].template cast_reg<U>().val;
	u32 tr1 = f1.sub[0][1].template cast_reg<U>().val;
	u32 bl1 = f1.sub[1][0].template cast_reg<U>().val;
	u32 br1 = f1.sub[1][1].template cast_reg<U>().val;

	recv = __shfl_xor_sync(0xffffffff, bit2 ? tl1 : tr1, 4);
	tl1 = bit2 ? recv : tl1;
	tr1 = bit2 ? tr1 : recv;

	recv = __shfl_xor_sync(0xffffffff, bit2 ? bl1 : br1, 4);
	bl1 = bit2 ? recv : bl1;
	br1 = bit2 ? br1 : recv;

	//---

	recv = __shfl_xor_sync(0xffffffff, bit3 ? tl0 : tl1, 8);
	tl0 = bit3 ? recv : tl0;
	tl1 = bit3 ? tl1 : recv;

	recv = __shfl_xor_sync(0xffffffff, bit3 ? tr0 : tr1, 8);
	tr0 = bit3 ? recv : tr0;
	tr1 = bit3 ? tr1 : recv;

	recv = __shfl_xor_sync(0xffffffff, bit3 ? bl0 : bl1, 8);
	bl0 = bit3 ? recv : bl0;
	bl1 = bit3 ? bl1 : recv;

	recv = __shfl_xor_sync(0xffffffff, bit3 ? br0 : br1, 8);
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

template<typename T, typename... U>
X17_DEVICE void zero_(Fragment_16x16<T> &f, U&... rest) {
	f.zero_();
	zero_(rest...);
}

template<typename T, const usize K>
X17_DEVICE void zero_(Fragment_16x16<T> (&arr)[K]) {
	for (usize i = 0; i < K; i++) {
		arr[i].zero_();
	}
}

template<typename T>
X17_DEVICE void scale_(Fragment_16x16<T> &f, T s) {
	f.scale_(s);
}

template<typename T, const usize K>
X17_DEVICE void scale_(Fragment_16x16<T> (&arr)[K], T s) {
	for (usize i = 0; i < K; i++) {
		arr[i].scale_(s);
	}
}

template<typename T>
X17_DEVICE void scale_top_(Fragment_16x16<T> &f, T s) {
	f.scale_top_(s);
}

template<typename T, const usize K>
X17_DEVICE void scale_top_(Fragment_16x16<T> (&arr)[K], T s) {
	for (usize i = 0; i < K; i++) {
		arr[i].scale_top_(s);
	}
}

template<typename T>
X17_DEVICE void scale_bottom_(Fragment_16x16<T> &f, T s) {
	f.scale_bottom_(s);
}

template<typename T, const usize K>
X17_DEVICE void scale_bottom_(Fragment_16x16<T> (&arr)[K], T s) {
	for (usize i = 0; i < K; i++) {
		arr[i].scale_bottom_(s);
	}
}

template<typename T, const usize K>
X17_DEVICE void acc_(Fragment_16x16<T> (&dst)[K], Fragment_16x16<T> const (&src)[K]) {
	for (usize i = 0; i < K; i++) {
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
	M > 0 && M % 16 == 0
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

	template<const usize THREADS_PER_BLOCK>
	X17_DEVICE void cp_async_from(usize tid, GMatrix<T, M, N> src) const {
		if constexpr (N > 0) {
			__builtin_assume(tid < THREADS_PER_BLOCK);
			static_assert(ROW_BYTES % 128 == 0);

			constexpr usize CP_PER_ROW = ROW_BYTES / 16;
			static_assert(THREADS_PER_BLOCK % CP_PER_ROW == 0);
			constexpr usize ROWS_PER_STEP = THREADS_PER_BLOCK / CP_PER_ROW;
			constexpr usize STEPS = M / ROWS_PER_STEP;

			if constexpr (STEPS == 0) {
				if constexpr (M % ROWS_PER_STEP == 0) {
					return;
				}
				if (tid >= (M % ROWS_PER_STEP) * CP_PER_ROW) {
					return;
				}
			}

			// Thread's position within a step is fixed
			usize col_in_row = (tid % CP_PER_ROW) * 16;  // byte offset within row
			usize row_in_step = tid / CP_PER_ROW;

			constexpr usize REPEAT_AFTER = least_common_multiple(8, ROWS_PER_STEP) / ROWS_PER_STEP;
			usize off[REPEAT_AFTER];
			X17_UNROLL for (usize i = 0; i < REPEAT_AFTER; i++) {
				usize row = i * ROWS_PER_STEP + row_in_step;
				off[i] = col_in_row ^ ((row & 7) << 4);
			}

			u8 const *src_ptr =
				reinterpret_cast<u8 const *>(src._ptr)
				+ row_in_step * src.stride_bytes()
				+ col_in_row;
			usize src_step = ROWS_PER_STEP * src.stride_bytes();

			usize dst_ptr = _ptr + row_in_step * ROW_BYTES;
			usize dst_step = ROWS_PER_STEP * ROW_BYTES;

			if constexpr (STEPS > 0) {
				X17_UNROLL for (usize step = 0; step < STEPS; step++) {
					sm80::cp_async(src_ptr, dst_ptr + off[step % REPEAT_AFTER]);
					src_ptr += src_step;
					dst_ptr += dst_step;
				}
			}
			if constexpr (M % ROWS_PER_STEP != 0) {
				usize step = STEPS;
				if (tid < (M % ROWS_PER_STEP) * CP_PER_ROW) {
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

			sm80::ldmatrix_8x8xu16_x4(
				addr,
				dst.sub[0][0].val, dst.sub[0][1].val, dst.sub[1][0].val, dst.sub[1][1].val
			);
		}
	}

};

template<
	const usize THREADS_PER_BLOCK,
	typename T,
	const usize M,
	const usize N
>
X17_DEVICE void cp_async_gmem_to_smem(
	usize tid,
	GMatrix<T, M, N> src,
	SMatrix<T, M, N> dst
) {
	dst.template cp_async_from<THREADS_PER_BLOCK>(tid, src);
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

using sm80::cp_async_commit;
using sm80::cp_async_wait;

//--------------------------------------------------------------------------------------------------

template<const usize M, const usize N, const usize K>
requires(M == 16 && N == K * 16)
X17_DEVICE void fragments_to_smem(
	Fragment_16x16<f32> const (&src)[K],
	SMatrix<f32, M, N> const &dst
) {
	usize tid = threadIdx.x % WARP_SIZE;
	constexpr u32 TILE_STRIDE = 2 * WARP_SIZE * 4 * sizeof(f32); // 1024 bytes per 16x16 f32 tile

	for (usize i = 0; i < K; i++) {
		u32 base = dst._ptr + i * TILE_STRIDE;
		u32 p0 = base + tid * 4 * sizeof(f32);
		u32 p1 = p0 + WARP_SIZE * 4 * sizeof(f32);

		store_shared_4(
			p0,
			src[i].sub[0][0].val0, src[i].sub[0][0].val1,
			src[i].sub[0][1].val0, src[i].sub[0][1].val1
		);
		store_shared_4(
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
	SMatrix<f32, M, N> const &src
) {
	usize tid = threadIdx.x % WARP_SIZE;
	constexpr u32 TILE_STRIDE = 2 * WARP_SIZE * 4 * sizeof(f32); // 1024 bytes per 16x16 f32 tile
	for (usize i = 0; i < K; i++) {
		u32 base = src._ptr + i * TILE_STRIDE;
		u32 p0 = base + tid * 4 * sizeof(f32);
		u32 p1 = p0 + WARP_SIZE * 4 * sizeof(f32);

		load_shared_4(
			p0,
			dst[i].sub[0][0].val0, dst[i].sub[0][0].val1,
			dst[i].sub[0][1].val0, dst[i].sub[0][1].val1
		);
		load_shared_4(
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
