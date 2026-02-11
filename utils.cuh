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

X17_DEVICE u32 cast_smem_ptr_to_uint(void const *const ptr) {
	return static_cast<u32>(__cvta_generic_to_shared(ptr));
}

namespace sm75 {
	/// `smem_src` must be 16-byte aligned.
	X17_DEVICE void ldmatrix_8x8xu16_x2(
		void const *smem_src,
		u32 &dst0, u32 &dst1
	) {
		u32 smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
		asm volatile (
			"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
			: "=r"(dst0), "=r"(dst1)
			: "r"(smem_int_ptr)
		);
	}

	/// `smem_src` must be 16-byte aligned.
	X17_DEVICE void ldmatrix_8x8xu16_x4(
		void const *smem_src,
		u32 &dst0, u32 &dst1, u32 &dst2, u32 &dst3
	) {
		u32 smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
		asm volatile (
			"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
			: "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
			: "r"(smem_int_ptr)
		);
	}

	/// `smem_src` must be 16-byte aligned.
	X17_DEVICE void ldmatrix_t_8x8xu16_x4(
		void const *smem_src,
		u32 &dst0, u32 &dst1, u32 &dst2, u32 &dst3
	) {
		u32 smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
		asm volatile (
			"ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
			: "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
			: "r"(smem_int_ptr)
		);
	}

	/// `smem_ptr` must be 16-byte aligned.
	/// `offset * sizeof(T)` must be a multiple of 16.
	template<typename T>
	X17_DEVICE T *ldmatrix_swizzle(T *smem_ptr, u32 byte_offset) {
		byte_offset ^= ((byte_offset & (7 << 7)) >> 3);
		return reinterpret_cast<T *>(
			reinterpret_cast<u8 *>(smem_ptr) + byte_offset
		);
	}
}

namespace sm80 {
	using namespace sm75;

	/// Both `gmem_src` and `smem_dst` must be 16-byte (128-bit) aligned.
	template<typename T>
	X17_DEVICE void cp_async(T const *gmem_src, T *smem_dst) {
		u32 smem_int_ptr = cast_smem_ptr_to_uint(smem_dst);
		asm volatile (
			"cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
			:
			: "r"(smem_int_ptr), "l"(gmem_src), "n"(sizeof(u128))
		);
	}

	X17_DEVICE void cp_async_commit() {
		asm volatile("cp.async.commit_group;\n" : :);
	}

	/// Blocks until all but N previous cp.async.commit_group operations have committed.
	template<int N = 0>
	X17_DEVICE void cp_async_wait() {
		if constexpr (N == 0) {
			asm volatile("cp.async.wait_all;\n" : :);
		} else {
			asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
		}
	}

	X17_DEVICE void mma_bf16_f32(
		f32       &d0, f32       &d1, f32       &d2, f32       &d3,
		u32 const &a0, u32 const &a1, u32 const &a2, u32 const &a3,
		u32 const &b0, u32 const &b1
	) {
		asm volatile(
			"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.zero "
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
			"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
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

template<typename T>
struct GPtr {
	using value_type = T;

	T *_ptr;

	X17_HOST_DEVICE GPtr(T *p): _ptr(p) {}

	X17_DEVICE GPtr with_byte_offset(size_t offset) const {
		return GPtr(
			reinterpret_cast<T *>(
				reinterpret_cast<u8 *>(_ptr) + offset
			)
		);
	}

	X17_DEVICE T const *get() const {
		return _ptr;
	}
};

template<typename T>
struct SwizzledSptr {
	using value_type = T;

	T *_ptr;
	usize _byte_offset;

	X17_DEVICE SwizzledSptr(T *p, usize byte_offset = 0):
		_ptr(p),
		_byte_offset(byte_offset)
	{}

	X17_DEVICE SwizzledSptr with_byte_offset(usize byte_offset) const {
		return SwizzledSptr(_ptr, _byte_offset + byte_offset);
	}

	X17_DEVICE T *get() const {
		return sm80::ldmatrix_swizzle(_ptr, _byte_offset);
	}
};

template<typename Data>
struct MatrixData {
	Data data;
};

template<const isize V>
struct ConstExtent {
	X17_DEVICE constexpr usize value() const noexcept {
		return V;
	}
};

struct DynamicExtent {
	usize v;
	X17_DEVICE constexpr usize value() const noexcept {
		return v;
	}
};

template<const isize V>
struct Extent: std::conditional_t<(V >= 0), ConstExtent<V>, DynamicExtent> {};

template<const isize M>
struct MatrixRowCount: Extent<M> {};

template<const isize N>
struct MatrixColCount: Extent<N> {};

enum MatrixLayout {
	RowMajor,
	ColumnMajor
};

template<
	typename Data,
	const isize M, // number of rows
	const isize N, // number of columns
	const MatrixLayout L = RowMajor,
	const usize STRIDE = (L == RowMajor ? N : M)
>
requires(
	(M >= 0 || N >= 0) // at least one dimension must be known
	&& ( !(L == RowMajor) || N >= 0 ) // if row-major, N must be known
	&& ( !(L == ColumnMajor) || M >= 0 ) // if column-major, M must be known
	&& ( !(L == RowMajor) || STRIDE >= N ) // if row-major, stride must be >= N
	&& ( !(L == ColumnMajor) || STRIDE >= M ) // if column-major, stride must be >= M
)
struct Matrix:
	MatrixData<Data>,
	MatrixRowCount<M>,
	MatrixColCount<N>
{
	X17_HOST_DEVICE Matrix(Data d) requires(M >= 0 && N >= 0):
		MatrixData<Data>{d},
		MatrixRowCount<M>{},
		MatrixColCount<N>{}
	{}

	X17_HOST_DEVICE Matrix(Data d, usize m_rows) requires(M < 0 && N >= 0 && L == RowMajor):
		MatrixData<Data>{d},
		MatrixRowCount<M>{m_rows},
		MatrixColCount<N>{}
	{}

	X17_HOST_DEVICE Matrix(Data d, usize n_cols) requires(M >= 0 && N < 0 && L == ColumnMajor):
		MatrixData<Data>{d},
		MatrixRowCount<M>{},
		MatrixColCount<N>{n_cols}
	{}

	X17_DEVICE usize m_rows() const {
		return MatrixRowCount<M>::value();
	}

	X17_DEVICE usize n_cols() const {
		return MatrixColCount<N>::value();
	}

	X17_DEVICE usize elems() const {
		return m_rows() * n_cols();
	}

	X17_DEVICE constexpr usize stride() const {
		return STRIDE;
	}

	X17_DEVICE constexpr MatrixLayout layout() const {
		return L;
	}

	X17_DEVICE Matrix<Data, N, M, ColumnMajor> t() const requires(
		M >= 0 && N >= 0
		&& L == RowMajor
	) {
		return Matrix<Data, N, M, ColumnMajor, STRIDE>{MatrixData<Data>::data};
	}

	template<const isize M_TILE, const isize N_TILE>
	X17_DEVICE Matrix<Data, M_TILE, N_TILE, L, STRIDE> tile(
		usize m_tile_idx, usize n_tile_idx
	) const requires(
		M_TILE >= 0 && N_TILE >= 0
		&& M >= 0 && M % M_TILE == 0
		&& N >= 0 && N % N_TILE == 0
	) {
		return Matrix<Data, M_TILE, N_TILE, L, STRIDE>{
			MatrixData<Data>::data.with_byte_offset(
				L == RowMajor
					?
						(
							m_tile_idx * usize(M_TILE) * usize(STRIDE)
							+ n_tile_idx * usize(N_TILE)
						) * usize(sizeof(typename Data::value_type))
					:
						(
							m_tile_idx * usize(M_TILE)
							+ n_tile_idx * usize(N_TILE) * usize(STRIDE)
						) * usize(sizeof(typename Data::value_type))
			)
		};
	}

	template<const isize M_TILE, const isize N_TILE>
	X17_DEVICE Matrix<Data, M_TILE, N_TILE, RowMajor, STRIDE> tile(
		usize m_tile_idx, usize n_tile_idx
	) const requires(
		M_TILE >= 0 && N_TILE >= 0
		&& M < 0 && L == RowMajor
		&& N >= 0 && N % N_TILE == 0
	) {
		return Matrix<Data, M_TILE, N_TILE, RowMajor>{
			MatrixData<Data>::data.with_byte_offset(
				(
					m_tile_idx * usize(M_TILE) * usize(STRIDE)
					+ n_tile_idx * usize(N_TILE)
				) * usize(sizeof(typename Data::value_type))
			)
		};
	}

	template<const isize M_TILE>
	X17_DEVICE Matrix<Data, M_TILE, N, L, STRIDE> tile_m(usize m_tile_idx) const
	requires(M_TILE >= 0 && N >= 0) {
		return tile<M_TILE, N>(m_tile_idx, 0);
	}

	template<const isize N_TILE>
	X17_DEVICE Matrix<Data, M, N_TILE, L, STRIDE> tile_n(usize n_tile_idx) const
	requires(N_TILE >= 0 && M >= 0) {
		return tile<M, N_TILE>(0, n_tile_idx);
	}
};

template<
	typename T,
	const isize M, // number of rows
	const isize N, // number of columns
	const MatrixLayout L = RowMajor,
	const usize STRIDE = (L == RowMajor ? N : M)
>
using GMatrix = Matrix<GPtr<T>, M, N, L, STRIDE>;

template<
	typename T,
	const isize M, // number of rows
	const isize N, // number of columns
	const MatrixLayout L = RowMajor,
	const usize STRIDE = (L == RowMajor ? N : M)
>
using SMatrix = Matrix<SwizzledSptr<T>, M, N, L, STRIDE>;

template<
	const usize BLOCK_DIM,
	const isize M, const isize N,
	typename T,
	const usize S1, const usize S2
>
X17_DEVICE void cp_async(
	usize thread_idx,
	GMatrix<T, M, N, RowMajor, S1> const &src,
	SMatrix<T, M, N, RowMajor, S2> const &dst
) requires(BLOCK_DIM > 0 && M > 0 && N > 0 && S1 == N && S2 == N) {
	constexpr usize BYTES = sizeof(T) * M * N;
	constexpr usize CP_ASYNC_CNT = BYTES / 16;
	static_assert(BYTES % 16 == 0, "cp.async size must be multiple of 16 bytes");

	GPtr<T> gptr = src.data.with_byte_offset(threadIdx.x * usize(16));
	SwizzledSptr<T> sptr = dst.data.with_byte_offset(threadIdx.x * usize(16));

	constexpr usize ITERATIONS = CP_ASYNC_CNT / BLOCK_DIM;
	if constexpr (ITERATIONS > 0) {
		X17_UNROLL for (usize i = 0; i < ITERATIONS; i++) {
			sm80::cp_async(gptr.get(), sptr.get());
			gptr = gptr.with_byte_offset(BLOCK_DIM * usize(16));
			sptr = sptr.with_byte_offset(BLOCK_DIM * usize(16));
		}
	}
	if constexpr (CP_ASYNC_CNT % BLOCK_DIM != 0) {
		if (thread_idx < CP_ASYNC_CNT % BLOCK_DIM) {
			sm80::cp_async(gptr.get(), sptr.get());
		}
	}
}

using sm80::cp_async_commit;
using sm80::cp_async_wait;

/// The whole warp holds one 8x8 matrix = 64 elements of u16.
///
/// So each thread holds 64 / 32 = 2 elements of u16 stored in one u32 register.
///
/// Which thread holds which matrix element:
/// row 0: |  0 |  0 |  1 |  1 |  2 |  2 |  3 |  3 |
/// row 1: |  4 |  4 |  5 |  5 |  6 |  6 |  7 |  7 |
/// ...
/// row 7: | 28 | 28 | 29 | 29 | 30 | 30 | 31 | 31 |
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

template<
	typename T,
	const isize M, const isize N,
	const usize MAJOR_DIM, const usize MINOR_DIM,
	const usize SZ = sizeof(T)
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

template<typename T, const usize STRIDE, const MatrixLayout L>
requires(sizeof(T) == 2)
X17_DEVICE void ldmatrix(
	usize thread_idx,
	SMatrix<T, 16, 16, L, STRIDE> const &src,
	RMatrix<T, 16, 16, L> &dst
) {
	u32 byte_offset =
		((thread_idx & 15) * STRIDE * sizeof(T))
		+ ((thread_idx & 16) / 2 * sizeof(T));
	sm80::ldmatrix_8x8xu16_x4(
		src.data.with_byte_offset(byte_offset).get(),
		dst.tiles[0][0].reg, dst.tiles[1][0].reg, dst.tiles[0][1].reg, dst.tiles[1][1].reg
	);
}

template<typename T, const usize STRIDE, const MatrixLayout L1, const MatrixLayout L2>
requires(sizeof(T) == 2 && L1 != L2)
X17_DEVICE void ldmatrix(
	usize thread_idx,
	SMatrix<T, 16, 16, L1, STRIDE> const &src,
	RMatrix<T, 16, 16, L2> &dst
) {
	u32 byte_offset =
		((thread_idx & 15) * STRIDE * sizeof(T))
		+ ((thread_idx & 16) / 2 * sizeof(T));
	sm80::ldmatrix_t_8x8xu16_x4(
		src.data.with_byte_offset(byte_offset).get(),
		dst.tiles[0][0].reg, dst.tiles[0][1].reg, dst.tiles[1][0].reg, dst.tiles[1][1].reg
	);
}

// The basic form is:
//     A: row major, B: column major, C: row major
//
// If A is col major, it is transposed.
// If B is row major, it is transposed.
// If C's layout is different from A, the result is transposed.
//
//  A   | B   | C   | operation
// -----+-----+-----+----------------
//  row | col | row | C = A x B
//  row | col | col | C = (A x B).T = B.T x A.T
//  row | row | row | C = A x B.T
//  row | row | col | C = (A x B.T).T = B x A.T
//  col | col | row | C = (A.T x B).T = B.T x A
//  col | col | col | C = A.T x B
//  col | row | row | C = (A.T x B.T).T = B x A
//  col | row | col | C = A.T x B.T
template<const MatrixLayout LA, const MatrixLayout LB, const MatrixLayout LC>
X17_DEVICE void gemm(
	RMatrix<f32, 16, 16, LA> &c,
	RMatrix<bf16, 16, 16, LB> const &a,
	RMatrix<bf16, 16, 16, LC> const &b
) {
    sm80::mma_bf16_f32(
		c.tiles[0][0].reg0, c.tiles[0][0].reg1, c.tiles[1][0].reg0, c.tiles[1][0].reg1,
		a.tiles[0][0].reg, a.tiles[1][0].reg, a.tiles[0][1].reg, a.tiles[1][1].reg,
		b.tiles[0][0].reg, b.tiles[0][1].reg,
		c.tiles[0][0].reg0, c.tiles[0][0].reg1, c.tiles[1][0].reg0, c.tiles[1][0].reg1
	);
    sm80::mma_bf16_f32(
		c.tiles[0][1].reg0, c.tiles[0][1].reg1, c.tiles[1][1].reg0, c.tiles[1][1].reg1,
		a.tiles[0][0].reg, a.tiles[1][0].reg, a.tiles[0][1].reg, a.tiles[1][1].reg,
		b.tiles[1][0].reg, b.tiles[1][1].reg,
		c.tiles[0][1].reg0, c.tiles[0][1].reg1, c.tiles[1][1].reg0, c.tiles[1][1].reg1
	);
}
