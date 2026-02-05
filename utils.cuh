//------------------------------------------------------------------------------
//
// Copyright 2026 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdint.h>

#if defined(__CUDACC_RTC__) || defined(__clang__)
	#define X17_UNROLL    _Pragma("unroll")
	#define X17_NO_UNROLL _Pragma("unroll 1")
#elif defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
	#define X17_UNROLL    #pragma unroll
	#define X17_NO_UNROLL #pragma unroll 1
#else
	#define X17_UNROLL
	#define X17_NO_UNROLL
#endif

#define X17_DEVICE __forceinline__ __device__

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

X17_DEVICE u32 cast_smem_ptr_to_uint(void const *const ptr) {
	return static_cast<u32>(__cvta_generic_to_shared(ptr));
}

namespace sm75 {
	X17_DEVICE void ldmatrix_8x8xu16_x4(
		u128 const *smem_src,
		u32 &dst0, u32 &dst1, u32 &dst2, u32 &dst3
	) {
		u32 smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
		asm volatile (
			"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
			: "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
			: "r"(smem_int_ptr)
		);
	}

	X17_DEVICE void ldmatrix_8x8xu16_t_x4(
		u128 const *smem_src,
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
	X17_DEVICE T *ldmatrix_swizzle(T *smem_ptr, u32 offset) {
		offset *= sizeof(T);
		// 111 000 0000
		offset ^= ((offset & (7 << 7)) >> 3);
		return reinterpret_cast<T *>(
			reinterpret_cast<u8 *>(smem_ptr) + offset
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
	template<int N>
	X17_DEVICE void cp_async_wait() {
		if constexpr (N == 0) {
			asm volatile("cp.async.wait_all;\n" : :);
		} else {
			asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
		}
	}
}

template<typename T>
struct GPtr {
	T *ptr;

	GPtr(T *p): ptr(p) {}

	GPtr with_offset(usize offset) const {
		return GPtr(ptr + offset);
	}
};

template<typename T>
struct SPtr {
	T *ptr;

	SPtr(T *p): ptr(p) {}

	SPtr with_offset(usize offset) const {
		return SPtr(ptr + offset);
	}
};

template<typename Data>
struct MatrixData {
	Data data;
};

template<const isize V>
struct ConstExtent {
	inline constexpr usize value() const noexcept {
		return V;
	}
};

struct DynamicExtent {
	usize v;
	inline constexpr usize value() const noexcept {
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
	inline Matrix(Data d) requires(M >= 0 && N >= 0):
		MatrixData<Data>{d},
		MatrixRowCount<M>{},
		MatrixColCount<N>{}
	{}

	inline Matrix(Data d, usize m_rows) requires(M < 0 && N >= 0 && L == RowMajor):
		MatrixData<Data>{d},
		MatrixRowCount<M>{m_rows},
		MatrixColCount<N>{}
	{}

	inline Matrix(Data d, usize n_cols) requires(M >= 0 && N < 0 && L == ColumnMajor):
		MatrixData<Data>{d},
		MatrixRowCount<M>{},
		MatrixColCount<N>{n_cols}
	{}

	inline usize m_rows() const {
		return MatrixRowCount<M>::value();
	}

	inline usize n_cols() const {
		return MatrixColCount<N>::value();
	}

	inline constexpr usize stride() const {
		return STRIDE;
	}

	inline constexpr MatrixLayout layout() const {
		return L;
	}

	inline Matrix<Data, N, M, ColumnMajor> transpose() const
	requires(M >= 0 && N >= 0 && L == RowMajor) {
		return Matrix<Data, N, M, ColumnMajor, STRIDE>{MatrixData<Data>::data};
	}

	template<const isize M_TILE, const isize N_TILE>
	Matrix<Data, M_TILE, N_TILE, L> tile(usize m_tile_idx, usize n_tile_idx) const
	requires(M_TILE >= 0 && N_TILE >= 0 && M >= 0 && N >= 0 && M % M_TILE == 0 && N % N_TILE == 0) {
		assert(m_tile_idx < (m_rows() / M_TILE));
		assert(n_tile_idx < (n_cols() / N_TILE));
		return Matrix<Data, M_TILE, N_TILE, L>{
			MatrixData<Data>::data.with_offset(L == RowMajor
				? m_tile_idx * M_TILE * stride() + n_tile_idx * N_TILE
				: n_tile_idx * N_TILE * stride() + m_tile_idx * M_TILE
			),
			stride()
		};
	}

	template<const isize M_TILE, const isize N_TILE>
	Matrix<Data, M_TILE, N_TILE, RowMajor> tile(usize m_tile_idx, usize n_tile_idx) const
	requires(M_TILE >= 0 && N_TILE >= 0 && M < 0 && N >= 0 && L == RowMajor && N % N_TILE == 0) {
		assert(m_rows() % M_TILE == 0);
		assert(m_tile_idx < (m_rows() / M_TILE));
		assert(n_tile_idx < (n_cols() / N_TILE));
		return Matrix<Data, M_TILE, N_TILE, RowMajor>{
			MatrixData<Data>::data.with_offset(m_tile_idx * M_TILE * stride() + n_tile_idx * N_TILE),
			stride()
		};
	}
};

template<const usize BLOCK_DIM, const isize M, const isize N, typename T>
X17_DEVICE void cp_asyncx(
	usize thread_idx,
	Matrix<GPtr<T>, M, N, RowMajor, N> const &src,
	Matrix<SPtr<T>, M, N, RowMajor, N> const &dst
) requires(M > 0 && N > 0) {
	T const *gptr = src.data.ptr;
	T *sptr = dst.data.ptr;

	constexpr usize BYTES = sizeof(T) * M * N;
	static_assert(BYTES % 16 == 0, "cp.async size must be multiple of 16 bytes");
	constexpr usize CP_ASYNC_CNT = BYTES / 16;
	if constexpr (CP_ASYNC_CNT < BLOCK_DIM) {
		if (thread_idx < CP_ASYNC_CNT) {
			usize offset = thread_idx * (16 / sizeof(T));
			sm80::cp_async(gptr + offset, sm80::ldmatrix_swizzle(sptr, offset));
		}
	} else if constexpr (CP_ASYNC_CNT == BLOCK_DIM) {
		usize offset = thread_idx * (16 / sizeof(T));
		sm80::cp_async(gptr + offset, sm80::ldmatrix_swizzle(sptr, offset));
	} else {
		constexpr usize ITERATIONS = CP_ASYNC_CNT / BLOCK_DIM;
		X17_UNROLL
		for (usize i = 0; i < ITERATIONS; i++) {
			usize offset = (i * BLOCK_DIM + thread_idx) * (16 / sizeof(T));
			sm80::cp_async(gptr + offset, sm80::ldmatrix_swizzle(sptr, offset));
		}
		if (thread_idx < CP_ASYNC_CNT % BLOCK_DIM) {
			usize offset = (ITERATIONS * BLOCK_DIM + thread_idx) * (16 / sizeof(T));
			sm80::cp_async(gptr + offset, sm80::ldmatrix_swizzle(sptr, offset));
		}
	}
}
