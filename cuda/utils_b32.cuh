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
}
