#include "utils.cuh"
#include "utils_b32.cuh"

namespace b8 {
	template<typename T>
	requires(sizeof(T) == 1)
	struct Fragment_8x16: FragmentReg<T> {};

	template<typename T>
	requires(sizeof(T) == 1)
	struct Fragment_16x16 {
		Fragment_8x16<T> v8x16[2];

		/// Before:
		///     even = [0, 1, 2, 3]
		///     odd  = [4, 5, 6, 7]
		/// after:
		///     even = [0, 4, 2, 6]
		///     odd  = [1, 5, 3, 7]
		X17_DEVICE void transpose_2x2() {
			u32 even = v8x16[0].val;
			u32 odd = v8x16[1].val;
			v8x16[0].val = __byte_perm(even, odd, 0x6240);
			v8x16[1].val  = __byte_perm(even, odd, 0x7351);
		}
	};

	template<typename T>
	requires(sizeof(T) == 1)
	struct Fragment_16x32 {
		Fragment_16x16<T> h16x16[2];
	};

	template<typename T>
	requires(sizeof(T) == 1)
	struct Fragment_32x32 {
		Fragment_16x32<T> v16x32[2];
	};

	/// Stores 4 horizontally-adjacent 8x16 fragments (64 cols × 8 rows) to GMEM.
	/// Uses shuffle_4x4 so each thread holds 16 contiguous bytes, then a single 128-bit store.
	/// Rows are written at stride 2 starting from m_idx, so callers use m_idx + 0 for even rows
	/// and m_idx + 1 for odd rows of a 16-row block.
	template<typename U, const usize M, const usize N>
	requires(sizeof(U) == 1)
	X17_DEVICE void store_1x4_8x16(
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx,
		u32 f0, u32 f1, u32 f2, u32 f3
	) {
		shuffle_4x4(f0, f1, f2, f3);

		usize tid = threadIdx.x % WARP_SIZE;
		u8 *base = reinterpret_cast<u8 *>(dst._ptr);
		usize stride = dst.stride_bytes();
		usize row = m_idx + ((tid / 2) & (((WARP_SIZE - 1) / 2) ^ 1));
		usize col_off = n_idx * usize(sizeof(U)) + (tid % 4) * 16;

		*reinterpret_cast<uint4 *>(base + row * stride + col_off) = make_uint4(f0, f1, f2, f3);
	}

	/// Stores a 16x32 tile (2×2 grid of 8x16 fragments) to GMEM.
	/// Uses shuffle_4x4 so each thread holds 16 contiguous bytes, then a single 128-bit store.
	/// Threads 0,1,4,5,8,9,... write even rows; threads 2,3,6,7,... write odd rows.
	/// 2 threads per row × 16B = 32B coalesced per row.
	template<typename U, const usize M, const usize N>
	requires(sizeof(U) == 1)
	X17_DEVICE void store_2x2_8x16(
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx,
		u32 f0, u32 f1,
		u32 f2, u32 f3
	) {
		shuffle_4x4(f0, f1, f2, f3);

		usize tid = threadIdx.x;
		u8 *base = reinterpret_cast<u8 *>(dst._ptr);
		usize stride = dst.stride_bytes();
		usize row = m_idx + ((tid % WARP_SIZE) / 2);
		usize col_off = n_idx * usize(sizeof(U)) + (tid % 2) * 16;

		*reinterpret_cast<uint4 *>(base + row * stride + col_off) = make_uint4(f0, f1, f2, f3);
	}

	template<typename U, typename T, const usize M, const usize N, const usize K>
	requires(sizeof(U) == 1)
	X17_DEVICE void store(
		Fragment_32x32<T> const (&tiles)[K],
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx
	) {
		usize i = 0;
		if constexpr (K >= 2) {
			X17_UNROLL for (; i + 2 <= K; i += 2) {
				store_1x4_8x16(
					dst, m_idx + 0, n_idx + i*32,
					tiles[i+0].v16x32[0].h16x16[0].v8x16[0].val,
					tiles[i+0].v16x32[0].h16x16[1].v8x16[0].val,
					tiles[i+1].v16x32[0].h16x16[0].v8x16[0].val,
					tiles[i+1].v16x32[0].h16x16[1].v8x16[0].val
				);
				store_1x4_8x16(
					dst, m_idx + 1, n_idx + i*32,
					tiles[i+0].v16x32[0].h16x16[0].v8x16[1].val,
					tiles[i+0].v16x32[0].h16x16[1].v8x16[1].val,
					tiles[i+1].v16x32[0].h16x16[0].v8x16[1].val,
					tiles[i+1].v16x32[0].h16x16[1].v8x16[1].val
				);
				store_1x4_8x16(
					dst, m_idx + 16, n_idx + i*32,
					tiles[i+0].v16x32[1].h16x16[0].v8x16[0].val,
					tiles[i+0].v16x32[1].h16x16[1].v8x16[0].val,
					tiles[i+1].v16x32[1].h16x16[0].v8x16[0].val,
					tiles[i+1].v16x32[1].h16x16[1].v8x16[0].val
				);
				store_1x4_8x16(
					dst, m_idx + 17, n_idx + i*32,
					tiles[i+0].v16x32[1].h16x16[0].v8x16[1].val,
					tiles[i+0].v16x32[1].h16x16[1].v8x16[1].val,
					tiles[i+1].v16x32[1].h16x16[0].v8x16[1].val,
					tiles[i+1].v16x32[1].h16x16[1].v8x16[1].val
				);
			}
		}
		if constexpr (K % 2 == 1) {
			store_2x2_8x16(
				dst, m_idx + 0, n_idx + i*32,
				tiles[i].v16x32[0].h16x16[0].v8x16[0].val,
				tiles[i].v16x32[0].h16x16[1].v8x16[0].val,
				tiles[i].v16x32[0].h16x16[0].v8x16[1].val,
				tiles[i].v16x32[0].h16x16[1].v8x16[1].val
			);
			store_2x2_8x16(
				dst, m_idx + 16, n_idx + i*32,
				tiles[i].v16x32[1].h16x16[0].v8x16[0].val,
				tiles[i].v16x32[1].h16x16[1].v8x16[0].val,
				tiles[i].v16x32[1].h16x16[0].v8x16[1].val,
				tiles[i].v16x32[1].h16x16[1].v8x16[1].val
			);
		}
	}

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

		/// Copy from a sub-region of a GMEM matrix into a sub-region of this SMEM matrix.
		/// Data is placed starting at (dst_row, dst_col) within this SMEM matrix.
		/// dst_row and dst_col must be multiples of 32
		template<
			const usize THREADS_PER_BLOCK,
			const usize WIDTH, const usize HEIGHT,
			const usize GM, const usize GN
		>
		requires(
			WIDTH <= GN && HEIGHT <= GM
			&& WIDTH <= N && HEIGHT <= M
		)
		X17_DEVICE void cp_async_from(
			usize tid,
			GMatrix<T, GM, GN> src,
			usize src_row,
			usize src_col,
			usize dst_row,
			usize dst_col
		) const {
			if constexpr (WIDTH > 0 && HEIGHT > 0) {
				__builtin_assume(tid < THREADS_PER_BLOCK);

				constexpr usize SRC_ROW_BYTES = WIDTH * sizeof(T);
				constexpr usize CP_BYTES = 16;
				constexpr usize CP_PER_ROW = SRC_ROW_BYTES / CP_BYTES;
				constexpr usize ROWS_PER_STEP = THREADS_PER_BLOCK / CP_PER_ROW;
				constexpr usize STEPS = HEIGHT / ROWS_PER_STEP;

				static_assert(CP_BYTES % sizeof(T) == 0);
				static_assert((WIDTH * sizeof(T)) % CP_BYTES == 0);
				static_assert((N * sizeof(T)) % CP_BYTES == 0);
				static_assert(THREADS_PER_BLOCK % CP_PER_ROW == 0);
				if constexpr (STEPS == 0) {
					if constexpr (HEIGHT % ROWS_PER_STEP == 0) {
						return;
					}
					if (tid >= (HEIGHT % ROWS_PER_STEP) * CP_PER_ROW) {
						return;
					}
				}

				// Thread's position within a step is fixed
				usize col_in_row = dst_col * sizeof(T) + (tid % CP_PER_ROW) * CP_BYTES;
				usize row_in_step = tid / CP_PER_ROW;
				usize src_col_in_row = src_col * sizeof(T) + (tid % CP_PER_ROW) * CP_BYTES;

				u8 const *src_ptr =
					reinterpret_cast<u8 const *>(src._ptr)
					+ (src_row + row_in_step) * src.stride_bytes()
					+ src_col_in_row;

				usize dst_ptr = _ptr + (dst_row + row_in_step) * ROW_BYTES;

				usize row = row_in_step;
				if constexpr (STEPS > 0) {
					X17_UNROLL for (usize step = 0; step < STEPS; ++step) {
						usize swizzle = (row & 14) << 3;
						usize off = col_in_row ^ swizzle;
						sm80::cp_async(src_ptr, dst_ptr + off);
						src_ptr += ROWS_PER_STEP * src.stride_bytes();
						dst_ptr += ROWS_PER_STEP * ROW_BYTES;
						row += ROWS_PER_STEP;
					}
				}
				if constexpr (HEIGHT % ROWS_PER_STEP != 0) {
					if (tid < (HEIGHT % ROWS_PER_STEP) * CP_PER_ROW) {
						usize swizzle = (row & 14) << 3;
						usize off = col_in_row ^ swizzle;
						sm80::cp_async(src_ptr, dst_ptr + off);
					}
				}
			}
		}

		/// Both `m_idx` and `n_idx` must be multiples of 32
		X17_DEVICE void tile_to_fragment(
			usize m_idx, usize n_idx,
			Fragment_32x32<T> &dst
		) const {
			if constexpr (N > 0) {
				usize tid = threadIdx.x;
				usize row = m_idx + (((tid & 15) << 1) | ((tid & 16) >> 4));
				usize swizzle = (tid & 7) << 4;
				usize col_off = n_idx * sizeof(T);
				u32 addr = _ptr + (row * ROW_BYTES) + (col_off ^ swizzle);

				sm80::ldmatrix_8x8xu16_x4(
					addr,
					dst.v16x32[0].h16x16[0].v8x16[0].val,
					dst.v16x32[1].h16x16[0].v8x16[0].val,
					dst.v16x32[0].h16x16[0].v8x16[1].val,
					dst.v16x32[1].h16x16[0].v8x16[1].val
				);

				addr ^= 16;

				sm80::ldmatrix_8x8xu16_x4(
					addr,
					dst.v16x32[0].h16x16[1].v8x16[0].val,
					dst.v16x32[1].h16x16[1].v8x16[0].val,
					dst.v16x32[0].h16x16[1].v8x16[1].val,
					dst.v16x32[1].h16x16[1].v8x16[1].val
				);
			}
		}

		/// Both `m_idx` and `n_idx` must be multiples of 32
		X17_DEVICE void tile_to_fragment_trans(
			usize m_idx, usize n_idx,
			Fragment_32x32<T> &dst
		) const {
			if constexpr (N > 0) {
				usize tid = threadIdx.x;
				usize row = m_idx + (((tid & 15) << 1) | ((tid & 16) >> 4));
				usize swizzle = (tid & 7) << 4;
				usize col_off = n_idx * sizeof(T);
				u32 addr = _ptr + (row * ROW_BYTES) + (col_off ^ swizzle);

				sm80::ldmatrix_t_8x8xu16_x4(
					addr,
					dst.v16x32[0].h16x16[0].v8x16[0].val,
					dst.v16x32[0].h16x16[1].v8x16[0].val,
					dst.v16x32[0].h16x16[0].v8x16[1].val,
					dst.v16x32[0].h16x16[1].v8x16[1].val
				);
				dst.v16x32[0].h16x16[0].transpose_2x2();
				dst.v16x32[0].h16x16[1].transpose_2x2();

				addr ^= 16;

				sm80::ldmatrix_t_8x8xu16_x4(
					addr,
					dst.v16x32[1].h16x16[0].v8x16[0].val,
					dst.v16x32[1].h16x16[1].v8x16[0].val,
					dst.v16x32[1].h16x16[0].v8x16[1].val,
					dst.v16x32[1].h16x16[1].v8x16[1].val
				);
				dst.v16x32[1].h16x16[0].transpose_2x2();
				dst.v16x32[1].h16x16[1].transpose_2x2();
			}
		}
	};

	X17_DEVICE void mma_a_bt(
		Fragment_16x32<i8> const &a,
		Fragment_16x32<i8> const &b,
		b32::Fragment_16x16<i32> &c
	) {
		X17_UNROLL for (int j = 0; j < 2; ++j) {
			sm80::mma_i8_i32(
				c.v8x16[0].h8x8[j].val0,
				c.v8x16[0].h8x8[j].val1,
				c.v8x16[1].h8x8[j].val0,
				c.v8x16[1].h8x8[j].val1,

				a.h16x16[0].v8x16[0].val,
				a.h16x16[0].v8x16[1].val,
				a.h16x16[1].v8x16[0].val,
				a.h16x16[1].v8x16[1].val,

				b.h16x16[0].v8x16[j].val,
				b.h16x16[1].v8x16[j].val,

				c.v8x16[0].h8x8[j].val0,
				c.v8x16[0].h8x8[j].val1,
				c.v8x16[1].h8x8[j].val0,
				c.v8x16[1].h8x8[j].val1
			);
		}
	}

	X17_DEVICE void mma_a_bt(
		Fragment_32x32<i8> const &a,
		Fragment_32x32<i8> const &b,
		b32::Fragment_32x32<i32> &c
	) {
		X17_UNROLL for (int j = 0; j < 2; ++j) {
			X17_UNROLL for (int i = 0; i < 2; ++i) {
				mma_a_bt(a.v16x32[j], b.v16x32[i], c.v16x32[j].h16x16[i]);
			}
		}
	}
}
