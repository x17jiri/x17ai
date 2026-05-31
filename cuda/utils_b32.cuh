#include "utils.cuh"

namespace b32 {
	template<typename T>
	requires(sizeof(T) == 4)
	struct Fragment_8x8 {
		T val0, val1;
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
		::zero_(f.h8x8);
	}

	template<typename T>
	X17_DEVICE void zero_(Fragment_16x16<T> &f) {
		::zero_(f.v8x16);
	}

	template<typename T>
	X17_DEVICE void zero_(Fragment_16x32<T> &f) {
		::zero_(f.h16x16);
	}

	template<typename T>
	X17_DEVICE void zero_(Fragment_32x32<T> &f) {
		::zero_(f.v16x32);
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_8x8<T> &f, T s) {
		f.val0 *= s;
		f.val1 *= s;
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_8x16<T> &f, T s) {
		::scale_(f.h8x8, s);
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_16x16<T> &f, T s) {
		::scale_(f.v8x16, s);
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_16x32<T> &f, T s) {
		::scale_(f.h16x16, s);
	}

	template<typename T>
	X17_DEVICE void scale_(Fragment_32x32<T> &f, T s) {
		::scale_(f.v16x32, s);
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
}
