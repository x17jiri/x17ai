#pragma once

#include "utils.cuh"

namespace b32 {
	template<typename T>
	requires(sizeof(T) == 4)
	struct Fragment_8x8 {
		T val0, val1;

		X17_DEVICE T get0() const {
			return val0;
		}

		X17_DEVICE T get1() const {
			return val1;
		}

		X17_DEVICE void set0(T new_value) {
			val0 = new_value;
		}

		X17_DEVICE void set1(T new_value) {
			val1 = new_value;
		}

		X17_DEVICE void set(T value0, T value1) {
			val0 = value0;
			val1 = value1;
		}
	};

	template<typename T>
	requires(sizeof(T) == 4)
	struct Fragment_8x16 {
		Fragment_8x8<T> h8x8[2];
	};

	template<typename T>
	requires(sizeof(T) == 4)
	struct Fragment_16x16 {
		Fragment_8x16<T> v8x16[2];
	};

	template<typename T>
	requires(sizeof(T) == 4)
	struct Fragment_16x32 {
		Fragment_16x16<T> h16x16[2];
	};

	template<typename T>
	requires(sizeof(T) == 4)
	struct Fragment_32x32 {
		Fragment_16x32<T> v16x32[2];
	};

	template<typename T>
	X17_DEVICE void zero_(Fragment_8x8<T> &f) {
		f.val0 = 0;
		f.val1 = 0;
	}

	template<typename T>
	X17_DEVICE void zero_(Fragment_8x16<T> &f) {
		zero_(f.h8x8[0]);
		zero_(f.h8x8[1]);
	}

	template<typename T>
	X17_DEVICE void zero_(Fragment_16x16<T> &f) {
		zero_(f.v8x16[0]);
		zero_(f.v8x16[1]);
	}

	template<typename T>
	X17_DEVICE void zero_(Fragment_16x32<T> &f) {
		zero_(f.h16x16[0]);
		zero_(f.h16x16[1]);
	}

	template<typename T>
	X17_DEVICE void zero_(Fragment_32x32<T> &f) {
		zero_(f.v16x32[0]);
		zero_(f.v16x32[1]);
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_8x8<T> &f, T v) {
		f.val0 = v;
		f.val1 = v;
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_8x16<T> &f, T v) {
		fill_(f.h8x8[0], v);
		fill_(f.h8x8[1], v);
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_16x16<T> &f, T v) {
		fill_(f.v8x16[0], v);
		fill_(f.v8x16[1], v);
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_16x32<T> &f, T v) {
		fill_(f.h16x16[0], v);
		fill_(f.h16x16[1], v);
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_32x32<T> &f, T v) {
		fill_(f.v16x32[0], v);
		fill_(f.v16x32[1], v);
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_8x8<T> &f, T s) {
		f.val0 *= s;
		f.val1 *= s;
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_8x16<T> &f, T s) {
		scale_(f.h8x8[0], s);
		scale_(f.h8x8[1], s);
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_16x16<T> &f, T s) {
		scale_(f.v8x16[0], s);
		scale_(f.v8x16[1], s);
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_16x32<T> &f, T s) {
		scale_(f.h16x16[0], s);
		scale_(f.h16x16[1], s);
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_32x32<T> &f, T s) {
		scale_(f.v16x32[0], s);
		scale_(f.v16x32[1], s);
	}

	template<const f64 SCALE = 1.0>
	X17_DEVICE void cast(Fragment_8x8<i32> const &src, Fragment_8x8<f32> &dst) {
		if constexpr (SCALE == 1.0) {
			dst.val0 = f32(src.val0);
			dst.val1 = f32(src.val1);
		} else {
			dst.val0 = f32(src.val0) * f32(SCALE);
			dst.val1 = f32(src.val1) * f32(SCALE);
		}
	}

	template<const f64 SCALE = 1.0>
	X17_DEVICE void cast(Fragment_8x16<i32> const &src, Fragment_8x16<f32> &dst) {
		cast<SCALE>(src.h8x8[0], dst.h8x8[0]);
		cast<SCALE>(src.h8x8[1], dst.h8x8[1]);
	}

	template<const f64 SCALE = 1.0>
	X17_DEVICE void cast(Fragment_16x16<i32> const &src, Fragment_16x16<f32> &dst) {
		cast<SCALE>(src.v8x16[0], dst.v8x16[0]);
		cast<SCALE>(src.v8x16[1], dst.v8x16[1]);
	}

	template<const f64 SCALE = 1.0>
	X17_DEVICE void cast(Fragment_16x32<i32> const &src, Fragment_16x32<f32> &dst) {
		cast<SCALE>(src.h16x16[0], dst.h16x16[0]);
		cast<SCALE>(src.h16x16[1], dst.h16x16[1]);
	}

	template<const f64 SCALE = 1.0>
	X17_DEVICE void cast(Fragment_32x32<i32> const &src, Fragment_32x32<f32> &dst) {
		cast<SCALE>(src.v16x32[0], dst.v16x32[0]);
		cast<SCALE>(src.v16x32[1], dst.v16x32[1]);
	}

	template<
		typename T,
		const usize M,
		const usize N
	>
	requires(sizeof(T) == 4 && M >= 0)
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
		requires(TILE_M > 0 && M % TILE_M == 0)
		X17_DEVICE constexpr SMatrix<T, TILE_M, N> tile_m(usize tile_idx) const {
			return SMatrix<T, TILE_M, N>{
				_ptr + (tile_idx * TILE_M * ROW_BYTES)
			};
		}
	};

	//--------------------------------------------------------------------------------------------------

	template<const usize M, const usize N, const usize K>
	requires(M == 16 && N == K * 16)
	X17_DEVICE void fragments_to_smem(
		Fragment_16x16<f32> const (&src)[K],
		SMatrix<f32, M, N> const &dst
	) {
		usize tid = threadIdx.x % WARP_SIZE;
		constexpr u32 TILE_STRIDE = 2 * WARP_SIZE * 4 * sizeof(f32); // 1024 bytes per 16x16 f32 tile

		X17_UNROLL for (usize i = 0; i < K; ++i) {
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
		SMatrix<f32, M, N> const &src
	) {
		usize tid = threadIdx.x % WARP_SIZE;
		constexpr u32 TILE_STRIDE = 2 * WARP_SIZE * 4 * sizeof(f32); // 1024 bytes per 16x16 f32 tile
		X17_UNROLL for (usize i = 0; i < K; ++i) {
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

}
