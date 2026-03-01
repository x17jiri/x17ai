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

	X17_DEVICE u32 ldmatrix_swizzle(u32 byte_offset) {
		return byte_offset ^ ((byte_offset >> 3) & 0x70);
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

	X17_DEVICE void transpose_() {
		sm80::movmatrix(sub[0][0].val, sub[0][0].val);
		Fragment_8x8<T> temp = sub[1][0];
		sm80::movmatrix(sub[0][1].val, sub[1][0].val);
		sm80::movmatrix(temp.val     , sub[0][1].val);
		sm80::movmatrix(sub[1][1].val, sub[1][1].val);
	}

	template<typename U, const usize M, const usize N>
	requires(sizeof(U) == 2)
	X17_DEVICE void store(GMatrix<U, M, N> const &dst, usize m_idx, usize n_idx) const {
		usize tid = threadIdx.x;
		GMatrix<U, 16, N> dst_tile = dst.tile_m<16>(m_idx / 16);
		usize dst_off = n_idx * usize(sizeof(U));

		dst_off += (tid & 0x1c) * (dst.stride_bytes() / 4) + (tid & 3) * 4;
		*reinterpret_cast<u32 *>(
			reinterpret_cast<u8 *>(dst_tile._ptr) + dst_off
		) = sub[0][0].template cast_reg<U>().val;

		dst_off += 16;
		*reinterpret_cast<u32 *>(
			reinterpret_cast<u8 *>(dst_tile._ptr) + dst_off
		) = sub[0][1].template cast_reg<U>().val;

		dst_off += 8 * dst.stride_bytes() - 16;
		*reinterpret_cast<u32 *>(
			reinterpret_cast<u8 *>(dst_tile._ptr) + dst_off
		) = sub[1][0].template cast_reg<U>().val;

		dst_off += 16;
		*reinterpret_cast<u32 *>(
			reinterpret_cast<u8 *>(dst_tile._ptr) + dst_off
		) = sub[1][1].template cast_reg<U>().val;
	}
};

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
	&& N > 0 && N * sizeof(T) % 128 == 0
)
struct SMatrix {
	u32 _ptr;
	u32 _thread_off[4];

	X17_DEVICE constexpr SMatrix() : _ptr(0), _thread_off{0, 0, 0, 0} {}

	X17_DEVICE constexpr SMatrix(void *ptr): SMatrix(cast_smem_ptr_to_uint(ptr)) {}

	X17_DEVICE constexpr SMatrix(u32 ptr):
		_ptr(ptr)
	{
		usize tid = threadIdx.x;
		_thread_off[0] = sm80::ldmatrix_swizzle((tid & 7) * 128  +  (tid & 16) + 0);
		_thread_off[1] = sm80::ldmatrix_swizzle((tid & 7) * 128  +  (tid & 16) + 32);
		_thread_off[2] = sm80::ldmatrix_swizzle((tid & 7) * 128  +  (tid & 16) + 64);
		_thread_off[3] = sm80::ldmatrix_swizzle((tid & 7) * 128  +  (tid & 16) + 96);
	}

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
/*		__builtin_assume(tid < THREADS_PER_BLOCK);

		constexpr usize THREADS_PER_TILE = 64;
		static_assert(THREADS_PER_BLOCK % THREADS_PER_TILE == 0, "TODO");
		constexpr usize TILES_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_TILE;
		static_assert((M / 8) % TILES_PER_BLOCK == 0, "TODO");

		usize src_off = (tid % 8) * 16 + (tid / 8) * ROW_BYTES;
		u8 *src_ptr = reinterpret_cast<u8 *>(src._ptr) + src_off;

		usize dst_off = (tid % 8) * 16 + (tid / 8) * 128;
		dst_off = sm80::ldmatrix_swizzle(dst_off);
		dst_off += (tid / THREADS_PER_TILE) * 8 * (ROW_BYTES - 128);
		usize dst_ptr = _ptr + dst_off;

		constexpr usize h = M / 16; // TODO : fix
		constexpr usize w = ROW_BYTES / 128;
		X17_UNROLL for (usize j = 0; j < h; j += TILES_PER_BLOCK) {
			X17_UNROLL for (usize i = 0; i < w; ++i) {
				sm80::cp_async(src_ptr, dst_ptr);
				src_ptr += 128;
				dst_ptr += 8 * 128;
			}
			src_ptr += 8 * TILES_PER_BLOCK * ROW_BYTES - w * 128;
			dst_ptr += (TILES_PER_BLOCK - 1) * 8 * ROW_BYTES;
		}
		return; // TODO*/

		__builtin_assume(tid < THREADS_PER_BLOCK);
		static_assert(ROW_BYTES % 128 == 0, "TODO");
		constexpr usize BYTES_PER_STEP = THREADS_PER_BLOCK * 16;
		constexpr usize STEPS = M * ROW_BYTES / BYTES_PER_STEP;
		constexpr usize REMAINING = M * ROW_BYTES % BYTES_PER_STEP;
		constexpr usize REPEAT_AFTER = least_common_multiple(8 * ROW_BYTES, BYTES_PER_STEP);

		constexpr usize PRECALC = std::min(
			STEPS + (REMAINING > 0 ? 1 : 0),
			REPEAT_AFTER / BYTES_PER_STEP
		);
		usize precalc[PRECALC];
		for (usize i = 0; i < PRECALC; i++) {
			usize off = i * BYTES_PER_STEP + tid * 16;
			usize row = off / ROW_BYTES;
			usize col = off % ROW_BYTES;

			usize outer_row = row / 8;
			usize outer_col = col / 128;
			usize inner_row = row % 8;
			usize inner_col = col % 128;

			precalc[i] =
				outer_row * 8 * ROW_BYTES
				+ outer_col * 8 * 128
				+ sm80::ldmatrix_swizzle(inner_row * 128 + inner_col);
		}

		u8 const *src_base = reinterpret_cast<u8 const *>(src._ptr);
		usize src_stride_bytes = src.stride_bytes();
		X17_UNROLL for (usize step = 0; step < STEPS; step++) {
			usize off = step * BYTES_PER_STEP + tid * 16;
			usize row = off / ROW_BYTES;
			usize col = off % ROW_BYTES;
			u8 const *src_ptr = src_base + row * src_stride_bytes + col;
			u32 dst_ptr = _ptr + precalc[step % PRECALC]
				+ usize(step / PRECALC) * usize(REPEAT_AFTER);
			sm80::cp_async(src_ptr, dst_ptr);
		}
		static_assert(REMAINING % 16 == 0);
		if constexpr (REMAINING > 0) {
			if (tid < REMAINING / 16) {
				usize step = STEPS;
				usize off = step * BYTES_PER_STEP + tid * 16;
				usize row = off / ROW_BYTES;
				usize col = off % ROW_BYTES;
				u8 const *src_ptr = src_base + row * src_stride_bytes + col;
				u32 dst_ptr = _ptr + precalc[step % PRECALC]
					+ usize(step / PRECALC) * usize(REPEAT_AFTER);
				sm80::cp_async(src_ptr, dst_ptr);
			}
		}
	}

	/// Both `m_idx` and `n_idx` must be multiples of 16.
	X17_DEVICE void load_tile_to_fragment(
		usize m_idx, usize n_idx,
		Fragment_16x16<T> &dst
	) const requires(sizeof(T) == 2) {
		usize tid = threadIdx.x;

		usize a =
			m_idx * ROW_BYTES
			+ (tid & 8) * ROW_BYTES
			+ (n_idx / 64) * (8 * 128);
		usize b = _thread_off[(n_idx % 64) / 16];

		sm80::ldmatrix_8x8xu16_x4(
			_ptr + a + b,
			dst.sub[0][0].val, dst.sub[1][0].val, dst.sub[0][1].val, dst.sub[1][1].val
		);
	}

	X17_DEVICE void load_tile_to_fragment_trans(
		usize m_idx, usize n_idx,
		Fragment_16x16<T> &dst
	) const requires(sizeof(T) == 2) {
		usize tid = threadIdx.x;

		usize a =
			m_idx * ROW_BYTES
			+ (tid & 8) * ROW_BYTES
			+ (n_idx / 64) * (8 * 128);
		usize b = _thread_off[(n_idx % 64) / 16];

		sm80::ldmatrix_8x8xu16_x4(
			_ptr + a + b,
			dst.sub[0][0].val, dst.sub[0][1].val, dst.sub[1][0].val, dst.sub[1][1].val
		);
	}

	X17_DEVICE void load_to_registers(RMatrix<T, M, N> &dst) const requires(sizeof(T) == 2) {
		X17_UNROLL for (usize j = 0; j < M / 16; j++) {
			X17_UNROLL for (usize i = 0; i < N / 16; i++) {
				load_tile_to_fragment(j * 16, i * 16, dst.tiles[j][i]);
			}
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

		f32 a = src[i].sub[0][0].val0;
		f32 b = src[i].sub[0][0].val1;
		f32 c = src[i].sub[0][1].val0;
		f32 d = src[i].sub[0][1].val1;
		asm volatile(
			"st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
			:
			: "r"(p0), "f"(a), "f"(b), "f"(c), "f"(d)
			: "memory"
		);
		a = src[i].sub[1][0].val0;
		b = src[i].sub[1][0].val1;
		c = src[i].sub[1][1].val0;
		d = src[i].sub[1][1].val1;
		asm volatile(
			"st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
			:
			: "r"(p1), "f"(a), "f"(b), "f"(c), "f"(d)
			: "memory"
		);
	}
}

/// Calculates `r = r * r_scale + s * s_scale`
template<const usize M, const usize N, const usize K>
requires(M == 16 && N == K * 16)
X17_DEVICE void rescale_acc(
	Fragment_16x16<f32> (&r)[K],
	f32 r_top_scale, f32 r_bot_scale,
	SMatrix<f32, M, N> const &s,
	f32 s_top_scale, f32 s_bot_scale
) {
	usize tid = threadIdx.x % WARP_SIZE;
	constexpr u32 TILE_STRIDE = 2 * WARP_SIZE * 4 * sizeof(f32); // 1024 bytes per 16x16 f32 tile
	for (usize i = 0; i < K; i++) {
		u32 base = s._ptr + i * TILE_STRIDE;
		u32 p0 = base + tid * 4 * sizeof(f32);
		u32 p1 = p0 + WARP_SIZE * 4 * sizeof(f32);

		f32 a, b, c, d;
		asm volatile(
			"ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
			: "=f"(a), "=f"(b), "=f"(c), "=f"(d)
			: "r"(p0)
		);
		r[i].sub[0][0].val0 = (r_top_scale * r[i].sub[0][0].val0) + (s_top_scale * a);
		r[i].sub[0][0].val1 = (r_top_scale * r[i].sub[0][0].val1) + (s_top_scale * b);
		r[i].sub[0][1].val0 = (r_top_scale * r[i].sub[0][1].val0) + (s_top_scale * c);
		r[i].sub[0][1].val1 = (r_top_scale * r[i].sub[0][1].val1) + (s_top_scale * d);
		asm volatile(
			"ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
			: "=f"(a), "=f"(b), "=f"(c), "=f"(d)
			: "r"(p1)
		);
		r[i].sub[1][0].val0 = (r_bot_scale * r[i].sub[1][0].val0) + (s_bot_scale * a);
		r[i].sub[1][0].val1 = (r_bot_scale * r[i].sub[1][0].val1) + (s_bot_scale * b);
		r[i].sub[1][1].val0 = (r_bot_scale * r[i].sub[1][1].val0) + (s_bot_scale * c);
		r[i].sub[1][1].val1 = (r_bot_scale * r[i].sub[1][1].val1) + (s_bot_scale * d);
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
