#pragma once

#include "utils.cuh"

namespace b32 {
	template<typename T>
	requires(sizeof(T) == 4)
	struct Fragment_8x8 {
		union Union {
			T value;
			u32 data;
		};

		u32 data0, data1;

		X17_DEVICE T get0() const {
			Union tmp;
			tmp.data = data0;
			return tmp.value;
		}

		X17_DEVICE T get1() const {
			Union tmp;
			tmp.data = data1;
			return tmp.value;
		}

		X17_DEVICE void set0(T new_value) {
			Union tmp;
			tmp.value = new_value;
			data0 = tmp.data;
		}

		X17_DEVICE void set1(T new_value) {
			Union tmp;
			tmp.value = new_value;
			data1 = tmp.data;
		}

		X17_DEVICE void set(T value0, T value1) {
			Union tmp;
			tmp.value = value0;
			data0 = tmp.data;
			tmp.value = value1;
			data1 = tmp.data;
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

	//----------------------------------------------------------------------------------------------

	/// Example for thread 0:
	/// - after using EvenOdd matrix as `B` in MMA, we hold columns: 0, 2, 1, 3
	/// - after this function, it is reordered to: 0, 1, 2, 3
	/// This is useful when we later want to convert to b8 fragment without shuffle.
	/// However it is still not the normal b32 order: 0, 1, 8, 9
	template<typename T>
	X17_DEVICE void fix_even_odd_columns_(Fragment_16x16<T> &frag) {
		X17_UNROLL for (usize row = 0; row < 2; ++row) {
			u32 tmp = frag.v8x16[row].h8x8[0].data1;
			frag.v8x16[row].h8x8[0].data1 = frag.v8x16[row].h8x8[1].data0;
			frag.v8x16[row].h8x8[1].data0 = tmp;
		}
	}

	//----------------------------------------------------------------------------------------------

	template<typename T>
	X17_DEVICE void fill_(Fragment_8x8<T> &f, T value) {
		f.set(value, value);
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_8x16<T> &f, T value) {
		fill_(f.h8x8[0], value);
		fill_(f.h8x8[1], value);
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_16x16<T> &f, T value) {
		fill_(f.v8x16[0], value);
		fill_(f.v8x16[1], value);
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_16x32<T> &f, T value) {
		fill_(f.h16x16[0], value);
		fill_(f.h16x16[1], value);
	}

	template<typename T>
	X17_DEVICE void fill_(Fragment_32x32<T> &f, T value) {
		fill_(f.v16x32[0], value);
		fill_(f.v16x32[1], value);
	}

	//----------------------------------------------------------------------------------------------

	template<typename T>
	X17_DEVICE void zero_(Fragment_8x8<T> &f) {
		fill_(f, T());
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

	//----------------------------------------------------------------------------------------------

	template<typename T>
	X17_DEVICE void scale_(Fragment_8x8<T> &f, T s) {
		f.set(f.get0() * s, f.get1() * s);
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

	//----------------------------------------------------------------------------------------------

	template<typename T>
	X17_DEVICE void acc_(Fragment_8x8<T> &dst, Fragment_8x8<T> const &a) {
		dst.set(dst.get0() + a.get0(), dst.get1() * a.get1());
	}

	template<typename T>
	X17_DEVICE void acc_(Fragment_8x16<T> &dst, Fragment_8x16<T> const &a) {
		acc_(dst.h8x8[0], a.h8x8[0]);
		acc_(dst.h8x8[1], a.h8x8[1]);
	}

	template<typename T>
	X17_DEVICE void acc_(Fragment_16x16<T> &dst, Fragment_16x16<T> const &a) {
		acc_(dst.v8x16[0], a.v8x16[0]);
		acc_(dst.v8x16[1], a.v8x16[1]);
	}

	template<typename T>
	X17_DEVICE void acc_(Fragment_16x32<T> &dst, Fragment_16x32<T> const &a) {
		acc_(dst.h16x16[0], a.h16x16[0]);
		acc_(dst.h16x16[1], a.h16x16[1]);
	}

	template<typename T>
	X17_DEVICE void acc_(Fragment_32x32<T> &dst, Fragment_32x32<T> const &a) {
		acc_(dst.v16x32[0], a.v16x32[0]);
		acc_(dst.v16x32[1], a.v16x32[1]);
	}

	//----------------------------------------------------------------------------------------------

	template<const f64 SCALE = 1.0>
	X17_DEVICE void cast(Fragment_8x8<i32> const &src, Fragment_8x8<f32> &dst) {
		if constexpr (SCALE == 1.0) {
			dst.data0 = src.data0;
			dst.data1 = src.data1;
		} else {
			dst.set(src.get0() * f32(SCALE), src.get1() * f32(SCALE));
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

	//----------------------------------------------------------------------------------------------

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

	//----------------------------------------------------------------------------------------------

}
