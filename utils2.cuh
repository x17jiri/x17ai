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

X17_DEVICE u32 cast_smem_ptr_to_uint(void const *const ptr) {
	return static_cast<u32>(__cvta_generic_to_shared(ptr));
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

	X17_DEVICE u32 ldmatrix_swizzle(u32 byte_offset) {
		return byte_offset ^ ((byte_offset >> 3) & 0x70);
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

template<typename T, const usize M, const usize N>
struct GMatrix {
	T *_ptr;

	X17_HOST_DEVICE constexpr GMatrix(T *ptr) : _ptr(ptr) {}

	X17_DEVICE constexpr usize m_rows() const { return M; }
	X17_DEVICE constexpr usize n_cols() const { return N; }
	X17_DEVICE constexpr usize stride() const { return N; }
	X17_DEVICE constexpr usize elems() const { return M * N; }

	template<const usize TILE_M>
	X17_DEVICE constexpr GMatrix<T, TILE_M, N> tile_m(usize tile_idx) const {
		return GMatrix<T, TILE_M, N>{_ptr + TILE_M * N * tile_idx};
	}
};

template<typename T, const usize N>
struct GMatrixDynSize {
	T *_ptr;
	usize _m;

	X17_HOST_DEVICE constexpr GMatrixDynSize(T *ptr, usize m) : _ptr(ptr), _m(m) {}

	X17_DEVICE constexpr usize m_rows() const { return _m; }
	X17_DEVICE constexpr usize n_cols() const { return N; }
	X17_DEVICE constexpr usize stride() const { return N; }
	X17_DEVICE constexpr usize elems() const { return _m * N; }

	template<const usize TILE_M>
	X17_DEVICE constexpr GMatrix<T, TILE_M, N> tile_m(size_t tile_idx) const {
		return GMatrix<T, TILE_M, N>{
			reinterpret_cast<T *>(
				reinterpret_cast<u8 *>(_ptr)
				+ size_t(TILE_M) * size_t(N) * size_t(tile_idx) * sizeof(T)
			)
		};
	}
};

//--------------------------------------------------------------------------------------------------

template<typename T, const usize M, const usize N, const usize STRIDE = N>
struct SMatrix {
	u32 _base_ptr;
	u32 _off;

	X17_DEVICE constexpr SMatrix(void *ptr):
		_base_ptr(cast_smem_ptr_to_uint(ptr)),
		_off(0)
	{}

	X17_DEVICE constexpr SMatrix(u32 base_ptr, u32 off):
		_base_ptr(base_ptr),
		_off(off)
	{}

	X17_DEVICE constexpr usize m_rows() const { return M; }
	X17_DEVICE constexpr usize n_cols() const { return N; }
	X17_DEVICE constexpr usize stride() const { return STRIDE; }
	X17_DEVICE constexpr usize elems() const { return M * N; }

	template<const usize TILE_M>
	requires(TILE_M > 0 && M % TILE_M == 0)
	X17_DEVICE constexpr SMatrix<T, TILE_M, N, STRIDE> tile_m(usize tile_idx) const {
		return SMatrix<T, TILE_M, N, STRIDE>{
			_base_ptr,
			_off + TILE_M * STRIDE * tile_idx
		};
	}

	template<const usize TILE_N>
	requires(TILE_N > 0 && N % TILE_N == 0)
	X17_DEVICE constexpr SMatrix<T, M, TILE_N, STRIDE> tile_n(usize tile_idx) const {
		return SMatrix<T, M, TILE_N, STRIDE>{
			_base_ptr,
			_off + TILE_N * usize(sizeof(T)) * tile_idx
		};
	}
};

//--------------------------------------------------------------------------------------------------

template<const usize BLOCK_DIM, typename T, const usize M, const usize N, const usize STRIDE>
requires(STRIDE == N)
X17_DEVICE void cp_async(GMatrix<T, M, N> src, SMatrix<T, M, N, STRIDE> dst) {
	constexpr usize BYTES = sizeof(T) * M * N;
	constexpr usize CP_ASYNC_CNT = BYTES / 16;
	static_assert(BYTES % 16 == 0, "cp.async size must be multiple of 16 bytes");

	usize tid = threadIdx.x;
	usize src_off = tid * 16;
	u8 *src_ptr = reinterpret_cast<u8 *>(src._ptr) + src_off;

	usize dst_off = sm80::ldmatrix_swizzle(src_off);
	usize dst_ptr = dst._base_ptr + dst_off;

	static_assert(BLOCK_DIM * 16 % 1024 == 0, "The swizzle pattern repeats after 1024 bytes. This assumption allows us to simply add a constant offset in the loop and not recalculate the swizzle");

	constexpr usize ITERATIONS = CP_ASYNC_CNT / BLOCK_DIM;
	if constexpr (ITERATIONS > 0) {
		X17_UNROLL for (usize i = 0; i < ITERATIONS; i++) {
			sm80::cp_async(src_ptr, dst_ptr);
			src_ptr += usize(BLOCK_DIM * 16);
			dst_ptr += usize(BLOCK_DIM * 16);
		}
	}
	if constexpr (CP_ASYNC_CNT % BLOCK_DIM != 0) {
		if (tid < CP_ASYNC_CNT % BLOCK_DIM) {
			sm80::cp_async(src_ptr, dst_ptr);
		}
	}
}

using sm80::cp_async_commit;
using sm80::cp_async_wait;

//--------------------------------------------------------------------------------------------------

enum MatrixLayout {
	RowMajor,
	ColumnMajor
};

//--------------------------------------------------------------------------------------------------

/// A fragment is an 8x8 tile held in registers by the whole warp.
/// Each thread holds 2 elements.
///
/// Which thread holds which matrix element:
/// row 0: |  0 |  0 |  1 |  1 |  2 |  2 |  3 |  3 |
/// row 1: |  4 |  4 |  5 |  5 |  6 |  6 |  7 |  7 |
/// ...
/// row 7: | 28 | 28 | 29 | 29 | 30 | 30 | 31 | 31 |
///
/// For 16-bit type, the two elements are packed into a single 32-bit register.
template<typename T>
requires(sizeof(T) == 2)
struct Fragment_8x8_u16 {
	u32 reg;

	X17_DEVICE T first() const {
		union {
			u32 reg;
			T halves[2];
		} a;
		a.reg = reg;
		return a.halves[0];
	}

	X17_DEVICE T second() const {
		union {
			u32 reg;
			T halves[2];
		} a;
		a.reg = reg;
		return a.halves[1];
	}

	X17_DEVICE void set(T first, T second) {
		union {
			u32 reg;
			T halves[2];
		} a;
		a.halves[0] = first;
		a.halves[1] = second;
		reg = a.reg;
	}
};

template<typename T>
requires(sizeof(T) == 4)
struct Fragment_8x8_u32 {
	T reg0;
	T reg1;

	X17_DEVICE T first() const {
		return reg0;
	}

	X17_DEVICE T second() const {
		return reg1;
	}

	X17_DEVICE void set(T first, T second) {
		reg0 = first;
		reg1 = second;
	}
};

//--------------------------------------------------------------------------------------------------

template<
	typename T,
	const isize M, const isize N,
	const usize MAJOR_DIM, const usize MINOR_DIM,
	const usize T_SIZE = sizeof(T)
>
struct RMatrix_impl;

template<
	typename T,
	const isize M, const isize N,
	const usize MAJOR_DIM, const usize MINOR_DIM
>
requires(
	M > 0 && M % 8 == 0
	&& N > 0 && N % 8 == 0
)
struct RMatrix_impl<T, M, N, MAJOR_DIM, MINOR_DIM, 2> {
	Fragment_8x8_u16<T> tiles[MINOR_DIM / 8][MAJOR_DIM / 8];

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
		X17_UNROLL for (usize j = 0; j < M / 8; j++) {
			X17_UNROLL for (usize i = 0; i < N / 8; i++) {
				tiles[j][i].set(T(), T());
			}
		}
	}
};

template<
	typename T,
	const isize M, const isize N,
	const usize MAJOR_DIM, const usize MINOR_DIM
>
requires(
	M > 0 && M % 8 == 0
	&& N > 0 && N % 8 == 0
)
struct RMatrix_impl<T, M, N, MAJOR_DIM, MINOR_DIM, 4> {
	Fragment_8x8_u32<T> tiles[MINOR_DIM / 8][MAJOR_DIM / 8];

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
		X17_UNROLL for (usize j = 0; j < M / 8; j++) {
			X17_UNROLL for (usize i = 0; i < N / 8; i++) {
				tiles[j][i].set(T(), T());
			}
		}
	}
};

template<typename T, const isize M, const isize N, const MatrixLayout L = RowMajor>
requires(
	sizeof(T) == 2 || sizeof(T) == 4
	&& M > 0 && M % 8 == 0
	&& N > 0 && N % 8 == 0
)
struct RMatrix: RMatrix_impl<
	T,
	M, N,
	(L == RowMajor ? N : M),
	(L == RowMajor ? M : N),
	sizeof(T)
> {};

//--------------------------------------------------------------------------------------------------

template<typename T, const usize STRIDE>
requires(sizeof(T) == 2)
X17_DEVICE void ldmatrix(SMatrix<T, 16, 16, STRIDE> src, RMatrix<T, 16, 16, RowMajor> &dst) {
	usize tid = threadIdx.x;
	usize thread_off = (tid & 15) * STRIDE * usize(sizeof(T)) + (tid & 16);
	usize off = src._off + thread_off;
	sm80::ldmatrix_8x8xu16_x4(
		src._base_ptr + sm80::ldmatrix_swizzle(off),
		dst.tiles[0][0].reg, dst.tiles[1][0].reg, dst.tiles[0][1].reg, dst.tiles[1][1].reg
	);
}

template<typename T, const usize STRIDE>
requires(sizeof(T) == 2)
X17_DEVICE void ldmatrix(SMatrix<T, 16, 16, STRIDE> src, RMatrix<T, 16, 16, ColumnMajor> &dst) {
	usize tid = threadIdx.x;
	usize thread_off = (tid & 15) * STRIDE * usize(sizeof(T)) + (tid & 16);
	usize off = src._off + thread_off;
	sm80::ldmatrix_t_8x8xu16_x4(
		src._base_ptr + sm80::ldmatrix_swizzle(off),
		dst.tiles[0][0].reg, dst.tiles[0][1].reg, dst.tiles[1][0].reg, dst.tiles[1][1].reg
	);
}

template<typename T, const usize STRIDE>
requires(sizeof(T) == 2)
X17_DEVICE void ldmatrix_t(SMatrix<T, 16, 16, STRIDE> src, RMatrix<T, 16, 16, RowMajor> &dst) {
	usize tid = threadIdx.x;
	usize thread_off = (tid & 15) * STRIDE * usize(sizeof(T)) + (tid & 16);
	usize off = src._off + thread_off;
	sm80::ldmatrix_t_8x8xu16_x4(
		src._base_ptr + sm80::ldmatrix_swizzle(off),
		dst.tiles[0][0].reg, dst.tiles[0][1].reg, dst.tiles[1][0].reg, dst.tiles[1][1].reg
	);
}

template<typename T, const usize STRIDE>
requires(sizeof(T) == 2)
X17_DEVICE void ldmatrix_t(SMatrix<T, 16, 16, STRIDE> src, RMatrix<T, 16, 16, ColumnMajor> &dst) {
	usize tid = threadIdx.x;
	usize thread_off = (tid & 15) * STRIDE * usize(sizeof(T)) + (tid & 16);
	usize off = src._off + thread_off;
	sm80::ldmatrix_8x8xu16_x4(
		src._base_ptr + sm80::ldmatrix_swizzle(off),
		dst.tiles[0][0].reg, dst.tiles[1][0].reg, dst.tiles[0][1].reg, dst.tiles[1][1].reg
	);
}

//--------------------------------------------------------------------------------------------------
