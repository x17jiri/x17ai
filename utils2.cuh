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
}

//--------------------------------------------------------------------------------------------------

namespace sm80 {
	using namespace sm75;

	/// Both `gmem_src` and `smem_dst` must be 16-byte (128-bit) aligned.
	template<typename T>
	X17_DEVICE void cp_async(T const *gmem_src, u32 smem_dst) {
		asm volatile (
			"cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
			:
			: "r"(smem_dst), "l"(gmem_src), "n"(sizeof(u128))
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
	X17_DEVICE constexpr GMatrix<T, TILE_M, N> tile_m(usize tile_idx) const {
		return GMatrix<T, TILE_M, N>{_ptr + TILE_M * N * tile_idx};
	}
};

//--------------------------------------------------------------------------------------------------

template<typename T, const usize M, const usize N>
struct SMatrix {
	u32 _ptr;

	X17_DEVICE constexpr SMatrix(void *ptr) : _ptr(cast_smem_ptr_to_uint(ptr)) {}
	X17_DEVICE constexpr SMatrix(u32 ptr) : _ptr(ptr) {}

	X17_DEVICE constexpr usize m_rows() const { return M; }
	X17_DEVICE constexpr usize n_cols() const { return N; }
	X17_DEVICE constexpr usize elems() const { return M * N; }

	template<const usize TILE_M>
	requires(TILE_M > 0 && TILE_M % 16 == 0)
	X17_DEVICE constexpr SMatrix<T, TILE_M, N> tile_m(usize tile_idx) const {
		return SMatrix<T, TILE_M, N>{_ptr + TILE_M * N * tile_idx};
	}
};

//--------------------------------------------------------------------------------------------------

constexpr u32 greatest_common_divisor(u32 a, u32 b) {
	while (b != 0) {
		u32 temp = b;
		b = a % b;
		a = temp;
	}
	return a;
}

constexpr u32 least_common_multiple(u32 a, u32 b) {
  return a * (b / greatest_common_divisor(a, b));
}

template<typename T, const usize N, const usize BLOCK_DIM>
requires(
	sizeof(T) == 2
	&& N > 0 && N % 16 == 0
	&& BLOCK_DIM % WARP_SIZE == 0
)
struct CpAsync {
	constexpr static usize LINES_PER_M_TILE = 16 * N * sizeof(T) / 16;

	constexpr static usize PRECALC = least_common_multiple(LINES_PER_M_TILE, BLOCK_DIM) / BLOCK_DIM;
	usize _offset[PRECALC];

	X17_DEVICE CpAsync() {
		usize off = threadIdx.x;
		X17_UNROLL for (usize i = 0; i < PRECALC; ++i) {
			// +-----------------------------------+-----------------------------------+
			// |                                   |                                   |
			// | -------0-------- -------8-------- | -------32------- -------40------- |
			// | -------1-------- -------9-------- | -------33------- -------41------- |
			// | -------2-------- -------10------- | -------34------- -------42------- |
			// | -------3-------- -------11------- | -------35------- -------43------- |
			// | -------4-------- -------12------- | -------36------- -------44------- |
			// | -------5-------- -------13------- | -------37------- -------45------- |
			// | -------6-------- -------14------- | -------38------- -------46------- |
			// | -------7-------- -------15------- | -------39------- -------47------- |
			// |                                   |                                   |
			// | -------16------- -------24------- | -------48------- -------56------- |
			// | -------17------- -------25------- | -------49------- -------57------- |
			// | -------18------- -------26------- | -------50------- -------58------- |
			// | -------19------- -------27------- | -------51------- -------59------- |
			// | -------20------- -------28------- | -------52------- -------60------- |
			// | -------21------- -------29------- | -------53------- -------61------- |
			// | -------22------- -------30------- | -------54------- -------62------- |
			// | -------23------- -------31------- | -------55------- -------63------- |
			// |                                   |                                   |
			// +-----------------------------------+-----------------------------------+

			usize y = off / (N * sizeof(T) / 16);
			usize x = off % (N * sizeof(T) / 16);
			usize off_tile = (x & ~1u) << 8;
			usize add_x = (off & 1) << 7;
			usize add_y = (y & 8) << 5;
			_offset[i] = off_tile | add_x | add_y | ((y & 7) << 4);

			off += BLOCK_DIM;
		}
	}

	template<const usize M>
	requires(M > 0 && M % 16 == 0)
	X17_DEVICE void run(GMatrix<T, M, N> src, SMatrix<T, M, N> dst) {
		constexpr static usize LINES_TO_COPY = M * N * sizeof(T) / 16;
		usize off = threadIdx.x * 16;
		usize dst_ptr = dst._ptr;
		X17_UNROLL for (usize i = 0; i < LINES_TO_COPY / BLOCK_DIM; ++i) {
			sm80::cp_async(
				reinterpret_cast<u8 *>(src._ptr) + off,
				dst_ptr + _offset[i % PRECALC]
			);
			off += BLOCK_DIM * 16;
			if ((i + 1) % PRECALC == 0) {
				dst_ptr += PRECALC * BLOCK_DIM * 16;
			}
		}
		if constexpr (LINES_TO_COPY % BLOCK_DIM != 0) {
			if (off < LINES_TO_COPY * 16) {
				usize i = LINES_TO_COPY / BLOCK_DIM;
				sm80::cp_async(
					reinterpret_cast<u8 *>(src._ptr) + off,
					dst_ptr + _offset[i % PRECALC]
				);
			}
		}
	}

	X17_DEVICE void commit() {
		sm80::cp_async_commit();
	}

	template<int CNT = 0>
	X17_DEVICE void wait() {
		sm80::cp_async_wait<CNT>();
	}
};

//--------------------------------------------------------------------------------------------------
