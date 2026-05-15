#include "utils.cuh"

namespace b8 {

	template<typename T>
	requires(sizeof(T) == 1)
	struct Fragment_8x16 {
		u32 val;
	};

	template<typename T>
	requires(sizeof(T) == 1)
	struct Fragment_16x16 {
		Fragment_8x16<T> sub[2];
	};

	template<
		typename T,
		const usize M,
		const usize N
	>
	requires(
		sizeof(T) == 1
		&& M >= 0 && M % 16 == 0
		&& N * sizeof(T) % 128 == 0
	)
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

		/// Both `m_idx` and `n_idx` must be multiples of 16.
		X17_DEVICE void tile_to_fragment(
			usize m_idx, usize n_idx,
			Fragment_16x16<T> &dst
		) const requires(sizeof(T) == 2) {
			if constexpr (N > 0) {
				usize tid = threadIdx.x;
				usize row = m_idx + (tid & 15);
				usize swizzle = (tid & 7) << 4;
				usize col_off = n_idx * sizeof(T);
				u32 addr = _ptr + (row * ROW_BYTES) + (col_off ^ swizzle);

				sm80::ldmatrix_8x8xu16_x2(addr, dst.sub[0].val, dst.sub[1].val);
			}
		}

		/// Both `m_idx` and `n_idx` must be multiples of 16.
		X17_DEVICE void tile_to_fragment_trans(
			usize m_idx, usize n_idx,
			Fragment_16x16<T> &dst
		) const requires(sizeof(T) == 2) {
			if constexpr (N > 0) {
				usize tid = threadIdx.x;
				usize row = m_idx + (tid & 15);
				usize swizzle = (tid & 7) << 4;
				usize col_off = n_idx * sizeof(T);
				u32 addr = _ptr + (row * ROW_BYTES) + (col_off ^ swizzle);

				sm80::ldmatrix_trans_8x8xu16_x2(addr, dst.sub[0].val, dst.sub[1].val);
			}
		}
	};
}
