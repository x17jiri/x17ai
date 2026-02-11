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

//--------------------------------------------------------------------------------------------------

template<typename T, const usize M, const usize N>
struct GMatrix {
	T *_ptr;

	X17_HOST_DEVICE constexpr GMatrix(T *ptr) : _ptr(ptr) {}

	X17_DEVICE constexpr usize m_rows() const { return M; }
	X17_DEVICE constexpr usize n_cols() const { return N; }
	X17_DEVICE constexpr usize stride() const { return N; }
	X17_DEVICE constexpr usize elems() const { return M * N; }
};

//--------------------------------------------------------------------------------------------------

template<typename T, const usize M, const usize N>
struct SMatrix {
	u32 _ptr;

	X17_DEVICE constexpr SMatrix(void *ptr) : _ptr(cast_smem_ptr_to_uint(ptr)) {}

	X17_DEVICE constexpr usize m_rows() const { return M; }
	X17_DEVICE constexpr usize n_cols() const { return N; }
	X17_DEVICE constexpr usize elems() const { return M * N; }
};

//--------------------------------------------------------------------------------------------------

template<typename T, const usize M, const usize N, const usize BLOCK_DIM>
requires(sizeof(T) == 2 && M == 16 && N > 0 && N % 16 == 0 && BLOCK_DIM % 32 == 0)
struct CpAsync {
	constexpr static usize PER_THREAD = (M * N * sizeof(T) + BLOCK_DIM - 1) / BLOCK_DIM;
	usize _offset[PER_THREAD];

	X17_DEVICE CpAsync() {
		usize off = 16 * threadIdx.x;
		X17_UNROLL for (usize i = 0; i < PER_THREAD; ++i) {
			usize y = off / (N * sizeof(T));
			usize x = off % (N * sizeof(T));
			usize off_tile = (x & ~31u) << 4;

			usize add_x = (off & 16) << 3;
			usize add_y = (y & 8) << 5;

			_offset[i] = off_tile + add_x + add_y;

			// +----------+----------+
			// |          |          |
			// |   0      |  128     |
			// |          |          |
			// +----------+----------+
			// |          |          |
			// |   256    |  384     |
			// |          |          |
			// +----------+----------+

			off += 16 * BLOCK_DIM;
		}
	}
};


//--------------------------------------------------------------------------------------------------
