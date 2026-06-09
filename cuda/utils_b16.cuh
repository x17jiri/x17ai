#pragma once

#include "utils.cuh"

namespace b16 {

	template<typename T, const usize T_SIZE = sizeof(T)>
	struct FragmentReg;

	template<typename T>
	requires(sizeof(T) == 1)
	struct FragmentReg<T, 1> {
		u32 val;
	};

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

		template<typename F>
		X17_DEVICE void elemwise_(F const &fn) {
			this->set(
				fn(this->first()),
				fn(this->second())
			);
		}

		X17_DEVICE void transpose_() requires(sizeof(T) == 4) {
			usize tid = threadIdx.x % WARP_SIZE;
			usize row = tid / 4;
			usize col_pair = tid % 4;

			usize src_lane0 = (2 * col_pair + 0) * 4 + (row / 2);
			usize src_lane1 = (2 * col_pair + 1) * 4 + (row / 2);
			bool take_second = (row & 1) != 0;

			T src00 = shuffle_sync(this->val0, int(src_lane0));
			T src01 = shuffle_sync(this->val1, int(src_lane0));
			T src10 = shuffle_sync(this->val0, int(src_lane1));
			T src11 = shuffle_sync(this->val1, int(src_lane1));

			this->set(
				take_second ? src01 : src00,
				take_second ? src11 : src10
			);
		}

		X17_DEVICE void transpose_() requires(sizeof(T) == 2) {
			sm80::movmatrix(this->val, this->val);
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

		X17_DEVICE void fill_(T v) {
			sub[0][0].set(v, v);
			sub[0][1].set(v, v);
			sub[1][0].set(v, v);
			sub[1][1].set(v, v);
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

		X17_DEVICE void scale_top_(T top) {
			sub[0][0].scale_(top);
			sub[0][1].scale_(top);
		}

		X17_DEVICE void scale_bottom_(T bot) {
			sub[1][0].scale_(bot);
			sub[1][1].scale_(bot);
		}

		template<typename F>
		X17_DEVICE void elemwise_(F const &fn) {
			sub[0][0].elemwise_(fn);
			sub[0][1].elemwise_(fn);
			sub[1][0].elemwise_(fn);
			sub[1][1].elemwise_(fn);
		}

		template<typename F>
		X17_DEVICE void elemwise_top_(F const &fn) {
			sub[0][0].elemwise_(fn);
			sub[0][1].elemwise_(fn);
		}

		template<typename F>
		X17_DEVICE void elemwise_bot_(F const &fn) {
			sub[1][0].elemwise_(fn);
			sub[1][1].elemwise_(fn);
		}

		X17_DEVICE void acc_(const Fragment_16x16 &o) {
			sub[0][0].val0 += o.sub[0][0].val0;
			sub[0][0].val1 += o.sub[0][0].val1;
			sub[0][1].val0 += o.sub[0][1].val0;
			sub[0][1].val1 += o.sub[0][1].val1;
			sub[1][0].val0 += o.sub[1][0].val0;
			sub[1][0].val1 += o.sub[1][0].val1;
			sub[1][1].val0 += o.sub[1][1].val0;
			sub[1][1].val1 += o.sub[1][1].val1;
		}

		X17_DEVICE void transpose_() requires(sizeof(T) == 2) {
			sm80::movmatrix(sub[0][0].val, sub[0][0].val);
			Fragment_8x8<T> temp = sub[1][0];
			sm80::movmatrix(sub[0][1].val, sub[1][0].val);
			sm80::movmatrix(temp.val     , sub[0][1].val);
			sm80::movmatrix(sub[1][1].val, sub[1][1].val);
		}

		X17_DEVICE void transpose_() requires(sizeof(T) == 4) {
			sub[0][0].transpose_();
			sub[1][1].transpose_();
			Fragment_8x8<T> temp = sub[0][1];
			sub[0][1] = sub[1][0];
			sub[1][0] = temp;
			sub[0][1].transpose_();
			sub[1][0].transpose_();
		}
	};

	template<typename T>
	struct Fragment_16x8 {
		Fragment_8x8<T> sub[2];
	};

	template<typename T>
	struct Fragment_16x32 {
		Fragment_16x16<T> h16x16[2];
	};

	template<typename T>
	struct Fragment_32x32 {
		Fragment_16x32<T> v16x32[2];
	};

	/// Stores 4 horizontally-adjacent 8x8 fragments (32 cols × 8 rows) to GMEM.
	/// Uses shuffle_4x4 so each thread holds 16 contiguous bytes, then a single 128-bit store.
	/// 4 threads per row × 16 bytes = 64B coalesced per row.
	template<typename U, typename T, const usize M, const usize N>
	requires(sizeof(U) == 2)
	X17_DEVICE void store_1x4_8x8(
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx,
		Fragment_8x8<T> const &f0,
		Fragment_8x8<T> const &f1,
		Fragment_8x8<T> const &f2,
		Fragment_8x8<T> const &f3
	) {
		// Cast to output type (e.g., f32 → bf16)
		Fragment_8x8<U> g0, g1, g2, g3;
		g0.val = f0.template cast_reg<U>().val;
		g1.val = f1.template cast_reg<U>().val;
		g2.val = f2.template cast_reg<U>().val;
		g3.val = f3.template cast_reg<U>().val;

		// Rearrange so each thread holds 4 consecutive double-columns
		shuffle_4x4(g0.val, g1.val, g2.val, g3.val);

		// 128-bit store per thread, 64B coalesced per row
		usize tid = threadIdx.x % WARP_SIZE;
		u8 *base = reinterpret_cast<u8 *>(dst._ptr);
		usize stride = dst.stride_bytes();
		usize off = (m_idx + tid / 4) * stride + n_idx * usize(sizeof(U)) + (tid % 4) * 16;

		*reinterpret_cast<uint4 *>(base + off) = make_uint4(g0.val, g1.val, g2.val, g3.val);
	}

	/// Stores a 16x16 tile (2×2 grid of 8x8 fragments) to GMEM.
	/// Uses shuffle_4x4 so each thread holds 16 contiguous bytes, then a single 128-bit store.
	/// Threads 0,1,4,5,8,9,... write top 8 rows; threads 2,3,6,7,... write bottom 8 rows.
	/// 2 threads per row × 16 bytes = 32B coalesced per row (full row for 16-col bf16).
	template<typename U, typename T, const usize M, const usize N>
	requires(sizeof(U) == 2)
	X17_DEVICE void store_2x2_8x8(
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx,
		Fragment_8x8<T> const &f0, Fragment_8x8<T> const &f1,
		Fragment_8x8<T> const &f2, Fragment_8x8<T> const &f3
	) {
		// Cast to output type (e.g., f32 → bf16)
		u32 g0 = f0.template cast_reg<U>().val;
		u32 g1 = f1.template cast_reg<U>().val;
		u32 g2 = f2.template cast_reg<U>().val;
		u32 g3 = f3.template cast_reg<U>().val;

		// f0=top-left, f1=top-right, f2=bottom-left, f3=bottom-right
		// After shuffle_4x4:
		//   t%4==0: all of tl's row (cols 0-7)   → top row
		//   t%4==1: all of tr's row (cols 8-15)  → top row
		//   t%4==2: all of bl's row (cols 0-7)   → bottom row
		//   t%4==3: all of br's row (cols 8-15)  → bottom row
		shuffle_4x4(g0, g1, g2, g3);

		usize tid = threadIdx.x % WARP_SIZE;
		u8 *base = reinterpret_cast<u8 *>(dst._ptr);
		usize stride = dst.stride_bytes();
		usize row = (tid & 2) ? (m_idx + 8 + tid / 4) : (m_idx + tid / 4);
		usize col_off = n_idx * usize(sizeof(U)) + (tid & 1) * 16;

		*reinterpret_cast<uint4 *>(base + row * stride + col_off) = make_uint4(g0, g1, g2, g3);
	}

	/// Stores 8 horizontally-adjacent 8x8 fragments (64 cols × 8 rows) to GMEM.
	/// shuffle_4x4 on left (f0-f3) and right (f4-f7) groups independently,
	/// then XOR-4 shuffle merges them so 8 consecutive threads cover one row.
	/// Each thread writes 2 × 16B stores; 8 threads × 16B = 128B coalesced per row.
	template<typename U, typename T, const usize M, const usize N>
	requires(sizeof(U) == 2)
	X17_DEVICE void store_1x8_8x8(
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx,
		Fragment_8x8<T> const &f0, Fragment_8x8<T> const &f1,
		Fragment_8x8<T> const &f2, Fragment_8x8<T> const &f3,
		Fragment_8x8<T> const &f4, Fragment_8x8<T> const &f5,
		Fragment_8x8<T> const &f6, Fragment_8x8<T> const &f7
	) {
		// Cast to output type (e.g., f32 → bf16)
		u32 g0 = f0.template cast_reg<U>().val;
		u32 g1 = f1.template cast_reg<U>().val;
		u32 g2 = f2.template cast_reg<U>().val;
		u32 g3 = f3.template cast_reg<U>().val;
		u32 g4 = f4.template cast_reg<U>().val;
		u32 g5 = f5.template cast_reg<U>().val;
		u32 g6 = f6.template cast_reg<U>().val;
		u32 g7 = f7.template cast_reg<U>().val;

		// Rearrange left and right groups independently
		shuffle_4x4(g0, g1, g2, g3);
		shuffle_4x4(g4, g5, g6, g7);

		// XOR-4 shuffle: swap right group (g4-g7) of bit2=0 threads
		// with left group (g0-g3) of bit2=1 threads.
		// This makes 8 consecutive threads cover the same row.
		usize tid = threadIdx.x % WARP_SIZE;
		bool bit2 = (tid & 4) != 0;
		u32 recv;

		recv = shuffle_xor_sync(bit2 ? g0 : g4, 4);
		g0 = bit2 ? recv : g0;
		g4 = bit2 ? g4 : recv;

		recv = shuffle_xor_sync(bit2 ? g1 : g5, 4);
		g1 = bit2 ? recv : g1;
		g5 = bit2 ? g5 : recv;

		recv = shuffle_xor_sync(bit2 ? g2 : g6, 4);
		g2 = bit2 ? recv : g2;
		g6 = bit2 ? g6 : recv;

		recv = shuffle_xor_sync(bit2 ? g3 : g7, 4);
		g3 = bit2 ? recv : g3;
		g7 = bit2 ? g7 : recv;

		// g0-g3 = even row data, g4-g7 = odd row data
		// 8 consecutive threads × 16B = 128B coalesced per row
		u8 *base = reinterpret_cast<u8 *>(dst._ptr);
		usize stride = dst.stride_bytes();
		usize col_off = n_idx * usize(sizeof(U)) + (tid % 8) * 16;
		usize even_row = (m_idx + (tid / 8) * 2) * stride;
		usize odd_row = even_row + stride;

		*reinterpret_cast<uint4 *>(base + even_row + col_off) = make_uint4(g0, g1, g2, g3);
		*reinterpret_cast<uint4 *>(base + odd_row  + col_off) = make_uint4(g4, g5, g6, g7);
	}

	/// Phase 1 of loading 8 horizontally-adjacent 8x8 fragments (64 cols × 8 rows) from GMEM.
	/// Performs 128-bit coalesced loads only; data remains in shuffled layout in the fragments.
	/// Call load_unshuffle_1x8_8x8 later to rearrange into MMA register layout.
	template<typename U, const usize M, const usize N>
	requires(sizeof(U) == 2)
	X17_DEVICE void load_shuffled_1x8_8x8(
		GMatrix<U, M, N> const &src,
		usize m_idx, usize n_idx,
		Fragment_8x8<U> &f0, Fragment_8x8<U> &f1,
		Fragment_8x8<U> &f2, Fragment_8x8<U> &f3,
		Fragment_8x8<U> &f4, Fragment_8x8<U> &f5,
		Fragment_8x8<U> &f6, Fragment_8x8<U> &f7
	) {
		// 128-bit load per thread, 8 threads per row, 128B coalesced per row
		usize tid = threadIdx.x % WARP_SIZE;
		u8 *base = reinterpret_cast<u8 *>(src._ptr);
		usize stride = src.stride_bytes();
		usize col_off = n_idx * usize(sizeof(U)) + (tid % 8) * 16;
		usize even_row = (m_idx + (tid / 8) * 2) * stride;
		usize odd_row = even_row + stride;

		uint4 even_data = *reinterpret_cast<uint4 const *>(base + even_row + col_off);
		uint4 odd_data  = *reinterpret_cast<uint4 const *>(base + odd_row  + col_off);

		f0.val = even_data.x; f1.val = even_data.y; f2.val = even_data.z; f3.val = even_data.w;
		f4.val = odd_data.x;  f5.val = odd_data.y;  f6.val = odd_data.z;  f7.val = odd_data.w;
	}

	/// Phase 2 of loading 8 horizontally-adjacent 8x8 fragments.
	/// Reverses the shuffled layout from load_shuffled_1x8_8x8 back to MMA register layout.
	/// XOR-4 and shuffle_4x4 are self-inverse, so the same operations as store undo the rearrangement.
	template<typename U>
	requires(sizeof(U) == 2)
	X17_DEVICE void load_unshuffle_1x8_8x8(
		Fragment_8x8<U> &f0, Fragment_8x8<U> &f1,
		Fragment_8x8<U> &f2, Fragment_8x8<U> &f3,
		Fragment_8x8<U> &f4, Fragment_8x8<U> &f5,
		Fragment_8x8<U> &f6, Fragment_8x8<U> &f7
	) {
		usize tid = threadIdx.x % WARP_SIZE;
		bool bit2 = (tid & 4) != 0;
		u32 recv;

		recv = shuffle_xor_sync(bit2 ? f0.val : f4.val, 4);
		f0.val = bit2 ? recv : f0.val;
		f4.val = bit2 ? f4.val : recv;

		recv = shuffle_xor_sync(bit2 ? f1.val : f5.val, 4);
		f1.val = bit2 ? recv : f1.val;
		f5.val = bit2 ? f5.val : recv;

		recv = shuffle_xor_sync(bit2 ? f2.val : f6.val, 4);
		f2.val = bit2 ? recv : f2.val;
		f6.val = bit2 ? f6.val : recv;

		recv = shuffle_xor_sync(bit2 ? f3.val : f7.val, 4);
		f3.val = bit2 ? recv : f3.val;
		f7.val = bit2 ? f7.val : recv;

		shuffle_4x4(f0.val, f1.val, f2.val, f3.val);
		shuffle_4x4(f4.val, f5.val, f6.val, f7.val);
	}

	/// Loads K horizontally-adjacent 16x16 tiles from GMEM in shuffled layout.
	/// K must be divisible by 4. Call load_unshuffle() to finalize.
	template<typename U, const usize M, const usize N, const usize K>
	requires(sizeof(U) == 2 && K % 4 == 0)
	X17_DEVICE void load_shuffled(
		Fragment_16x16<U> (&tiles)[K],
		GMatrix<U, M, N> const &src,
		usize m_idx, usize n_idx
	) {
		X17_UNROLL for (usize i = 0; i < K; i += 4) {
			load_shuffled_1x8_8x8(src, m_idx, n_idx + i*16,
				tiles[i].sub[0][0], tiles[i].sub[0][1],
				tiles[i+1].sub[0][0], tiles[i+1].sub[0][1],
				tiles[i+2].sub[0][0], tiles[i+2].sub[0][1],
				tiles[i+3].sub[0][0], tiles[i+3].sub[0][1]);
			load_shuffled_1x8_8x8(src, m_idx + 8, n_idx + i*16,
				tiles[i].sub[1][0], tiles[i].sub[1][1],
				tiles[i+1].sub[1][0], tiles[i+1].sub[1][1],
				tiles[i+2].sub[1][0], tiles[i+2].sub[1][1],
				tiles[i+3].sub[1][0], tiles[i+3].sub[1][1]);
		}
	}

	/// Unshuffles K horizontally-adjacent 16x16 tiles previously loaded with load_shuffled().
	/// K must be divisible by 4.
	template<typename U, const usize K>
	requires(sizeof(U) == 2 && K % 4 == 0)
	X17_DEVICE void load_unshuffle(
		Fragment_16x16<U> (&tiles)[K]
	) {
		X17_UNROLL for (usize i = 0; i < K; i += 4) {
			load_unshuffle_1x8_8x8(
				tiles[i].sub[0][0], tiles[i].sub[0][1],
				tiles[i+1].sub[0][0], tiles[i+1].sub[0][1],
				tiles[i+2].sub[0][0], tiles[i+2].sub[0][1],
				tiles[i+3].sub[0][0], tiles[i+3].sub[0][1]);
			load_unshuffle_1x8_8x8(
				tiles[i].sub[1][0], tiles[i].sub[1][1],
				tiles[i+1].sub[1][0], tiles[i+1].sub[1][1],
				tiles[i+2].sub[1][0], tiles[i+2].sub[1][1],
				tiles[i+3].sub[1][0], tiles[i+3].sub[1][1]);
		}
	}

	/// Generic store for an array of K horizontally-adjacent 16x16 tiles.
	/// Dispatches to store_1x8_8x8 (4 tiles), store_1x4_8x8 (2 tiles), store_2x2_8x8 (1 tile).
	template<typename U, typename T, const usize M, const usize N, const usize K>
	requires(sizeof(U) == 2)
	X17_DEVICE void store(
		Fragment_16x16<T> const (&tiles)[K],
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx
	) {
		usize i = 0;
		if constexpr (K >= 4) {
			X17_UNROLL for (; i + 4 <= K; i += 4) {
				store_1x8_8x8(dst, m_idx, n_idx + i*16,
					tiles[i].sub[0][0], tiles[i].sub[0][1],
					tiles[i+1].sub[0][0], tiles[i+1].sub[0][1],
					tiles[i+2].sub[0][0], tiles[i+2].sub[0][1],
					tiles[i+3].sub[0][0], tiles[i+3].sub[0][1]);
				store_1x8_8x8(dst, m_idx + 8, n_idx + i*16,
					tiles[i].sub[1][0], tiles[i].sub[1][1],
					tiles[i+1].sub[1][0], tiles[i+1].sub[1][1],
					tiles[i+2].sub[1][0], tiles[i+2].sub[1][1],
					tiles[i+3].sub[1][0], tiles[i+3].sub[1][1]);
			}
		}
		if constexpr (K % 4 >= 2) {
			store_1x4_8x8(dst, m_idx, n_idx + i*16,
				tiles[i].sub[0][0], tiles[i].sub[0][1],
				tiles[i+1].sub[0][0], tiles[i+1].sub[0][1]);
			store_1x4_8x8(dst, m_idx + 8, n_idx + i*16,
				tiles[i].sub[1][0], tiles[i].sub[1][1],
				tiles[i+1].sub[1][0], tiles[i+1].sub[1][1]);
			i += 2;
		}
		if constexpr (K % 2 == 1) {
			store_2x2_8x8(dst, m_idx, n_idx + i*16,
				tiles[i].sub[0][0], tiles[i].sub[0][1],
				tiles[i].sub[1][0], tiles[i].sub[1][1]);
		}
	}

	template<typename U, typename T, const usize M, const usize N>
	requires(sizeof(U) == 2)
	X17_DEVICE void store(
		Fragment_32x32<T> const &tile,
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx
	) {
		store(tile.v16x32[0].h16x16, dst, m_idx, n_idx);
		store(tile.v16x32[1].h16x16, dst, m_idx + 16, n_idx);
	}

	template<typename U, typename T, const usize M, const usize N, const usize K>
	requires(sizeof(U) == 2)
	X17_DEVICE void store(
		Fragment_32x32<T> const (&tiles)[K],
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx
	) {
		X17_UNROLL for (usize i = 0; i < K; ++i) {
			store(tiles[i], dst, m_idx, n_idx + 32*i);
		}
	}

	template<typename U, typename T, const usize M, const usize N>
	requires(sizeof(U) == 2)
	X17_DEVICE void store2(
		GMatrix<U, M, N> const &dst,
		usize m_idx, usize n_idx,
		Fragment_16x16<T> const &f0,
		Fragment_16x16<T> const &f1
	) {
		usize tid = threadIdx.x % WARP_SIZE;
		bool bit2 = (tid & 4) != 0;
		bool bit3 = (tid & 8) != 0;
		u32 recv;

		u32 tl0 = f0.sub[0][0].template cast_reg<U>().val;
		u32 tr0 = f0.sub[0][1].template cast_reg<U>().val;
		u32 bl0 = f0.sub[1][0].template cast_reg<U>().val;
		u32 br0 = f0.sub[1][1].template cast_reg<U>().val;

		recv = shuffle_xor_sync(bit2 ? tl0 : tr0, 4);
		tl0 = bit2 ? recv : tl0;
		tr0 = bit2 ? tr0 : recv;

		recv = shuffle_xor_sync(bit2 ? bl0 : br0, 4);
		bl0 = bit2 ? recv : bl0;
		br0 = bit2 ? br0 : recv;

		//---

		u32 tl1 = f1.sub[0][0].template cast_reg<U>().val;
		u32 tr1 = f1.sub[0][1].template cast_reg<U>().val;
		u32 bl1 = f1.sub[1][0].template cast_reg<U>().val;
		u32 br1 = f1.sub[1][1].template cast_reg<U>().val;

		recv = shuffle_xor_sync(bit2 ? tl1 : tr1, 4);
		tl1 = bit2 ? recv : tl1;
		tr1 = bit2 ? tr1 : recv;

		recv = shuffle_xor_sync(bit2 ? bl1 : br1, 4);
		bl1 = bit2 ? recv : bl1;
		br1 = bit2 ? br1 : recv;

		//---

		recv = shuffle_xor_sync(bit3 ? tl0 : tl1, 8);
		tl0 = bit3 ? recv : tl0;
		tl1 = bit3 ? tl1 : recv;

		recv = shuffle_xor_sync(bit3 ? tr0 : tr1, 8);
		tr0 = bit3 ? recv : tr0;
		tr1 = bit3 ? tr1 : recv;

		recv = shuffle_xor_sync(bit3 ? bl0 : bl1, 8);
		bl0 = bit3 ? recv : bl0;
		bl1 = bit3 ? bl1 : recv;

		recv = shuffle_xor_sync(bit3 ? br0 : br1, 8);
		br0 = bit3 ? recv : br0;
		br1 = bit3 ? br1 : recv;

		//---

		u8 *top_base = reinterpret_cast<u8 *>(dst.template tile_m<16>(m_idx / 16)._ptr);
		usize stride = dst.stride_bytes();
		usize col_off = n_idx * usize(sizeof(U)) + (tid % 16) * 4;
		usize row_base = (tid / 16) * 4 * stride;

		*reinterpret_cast<u32 *>(top_base + row_base + 0*stride + col_off) = tl0;
		*reinterpret_cast<u32 *>(top_base + row_base + 1*stride + col_off) = tr0;
		*reinterpret_cast<u32 *>(top_base + row_base + 2*stride + col_off) = tl1;
		*reinterpret_cast<u32 *>(top_base + row_base + 3*stride + col_off) = tr1;

		u8 *bot_base = top_base + 8 * stride;
		*reinterpret_cast<u32 *>(bot_base + row_base + 0*stride + col_off) = bl0;
		*reinterpret_cast<u32 *>(bot_base + row_base + 1*stride + col_off) = br0;
		*reinterpret_cast<u32 *>(bot_base + row_base + 2*stride + col_off) = bl1;
		*reinterpret_cast<u32 *>(bot_base + row_base + 3*stride + col_off) = br1;
	}

	//--------------------------------------------------------------------------------------------------

	template<
		typename T,
		const usize M,
		const usize N
	>
	requires(
		sizeof(T) == 2
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

		template<const usize TILE_M>
		requires(TILE_M > 0 && TILE_M % 8 == 0 && M % TILE_M == 0)
		X17_DEVICE constexpr SMatrix<T, TILE_M, N> tile_m(usize tile_idx) const {
			return SMatrix<T, TILE_M, N>{
				_ptr + (tile_idx * TILE_M * ROW_BYTES)
			};
		}
	};

	template<
		usize THREADS_PER_BLOCK,
		usize HEIGHT, usize WIDTH,
		typename T, usize M, usize N,
		usize GN
	>
	requires(
		WIDTH <= GN && WIDTH <= N
		&& HEIGHT <= M
	)
	X17_DEVICE void async_load(
		usize tid,
		GMatrixDynSize<T, GN> src, usize src_row, usize src_col,
		SMatrix<T, M, N> dst, usize dst_row, usize dst_col
	) {
		// `b8` SMatrix uses the same swizzle, so we can use its loader
		GMatrixDynSize<u8, 2*GN> b8_src(reinterpret_cast<u8 *>(src._ptr), src._stride_bytes);
		b8::SMatrix<u8, M, 2*N> b8_dst(dst._ptr);
		b8::async_load<THREADS_PER_BLOCK, HEIGHT, WIDTH>(
			tid,
			b8_src, src_row, src_col,
			b8_dst, dst_row, 2*dst_col
		);
	}

	/// Both `m_idx` and `n_idx` must be multiples of 16.
	template<typename T, usize M, usize N>
	X17_DEVICE void load_fragment(
		SMatrix<T, M, N> src, usize m_idx, usize n_idx,
		Fragment_16x16<T> &dst
	) {
		using Src = SMatrix<T, M, N>;
		if constexpr (N > 0) {
			usize tid = threadIdx.x;
			usize row = m_idx + (tid & 15);
			usize swizzle = ((threadIdx.x & 7) << 4) ^ (threadIdx.x & 16);
			usize byte_col = n_idx * sizeof(T);
			u32 addr = src._ptr + (row * Src::ROW_BYTES) + (byte_col ^ swizzle);

			sm80::ldmatrix_8x8xu16_x4(
				addr,
				dst.sub[0][0].val, dst.sub[1][0].val, dst.sub[0][1].val, dst.sub[1][1].val
			);
		}
	}

	/// `m_idx` must be a multiple of 16 and `n_idx` must be a multiple of 8.
	template<typename T, usize M, usize N>
	X17_DEVICE void load_fragment(
		SMatrix<T, M, N> src, usize m_idx, usize n_idx,
		Fragment_16x8<T> &dst
	) {
		using Src = SMatrix<T, M, N>;
		if constexpr (N > 0) {
			usize tid = threadIdx.x;
			usize row = m_idx + (tid & 15);
			usize swizzle = ((threadIdx.x & 7) << 4) ^ (threadIdx.x & 16);
			usize byte_col = n_idx * sizeof(T);
			u32 addr = src._ptr + (row * Src::ROW_BYTES) + (byte_col ^ swizzle);

			sm80::ldmatrix_8x8xu16_x2(
				addr,
				dst.sub[0].val, dst.sub[1].val
			);
		}
	}

	/// Loads a 16x16 tile from SMEM, transposed, into MMA registers.
	/// Uses ldmatrix.trans to transpose each 8x8 sub-tile during load,
	/// plus swaps off-diagonal destinations for a full 16x16 transpose.
	template<typename T, usize M, usize N>
	X17_DEVICE void load_fragment_trans(
		SMatrix<T, M, N> src, usize m_idx, usize n_idx,
		Fragment_16x16<T> &dst
	) {
		using Src = SMatrix<T, M, N>;
		if constexpr (N > 0) {
			usize tid = threadIdx.x;
			usize row = m_idx + (tid & 15);
			usize swizzle = ((threadIdx.x & 7) << 4) ^ (threadIdx.x & 16);
			usize byte_col = n_idx * sizeof(T);
			u32 addr = src._ptr + (row * Src::ROW_BYTES) + (byte_col ^ swizzle);

			sm80::ldmatrix_t_8x8xu16_x4(
				addr,
				dst.sub[0][0].val, dst.sub[0][1].val, dst.sub[1][0].val, dst.sub[1][1].val
			);
		}
	}

	template<typename T, usize M, usize N>
	X17_DEVICE void load_fragment(
		SMatrix<T, M, N> src, usize m_idx, usize n_idx,
		Fragment_32x32<T> &dst
	) {
		load_fragment(src, m_idx,      n_idx,      dst.v16x32[0].h16x16[0]);
		load_fragment(src, m_idx,      n_idx + 16, dst.v16x32[0].h16x16[1]);
		load_fragment(src, m_idx + 16, n_idx,      dst.v16x32[1].h16x16[0]);
		load_fragment(src, m_idx + 16, n_idx + 16, dst.v16x32[1].h16x16[1]);
	}

	template<typename T, usize M, usize N>
	X17_DEVICE void load_fragment_trans(
		SMatrix<T, M, N> src, usize m_idx, usize n_idx,
		Fragment_32x32<T> &dst
	) {
		load_fragment_trans(src, m_idx,      n_idx,      dst.v16x32[0].h16x16[0]);
		load_fragment_trans(src, m_idx + 16, n_idx,      dst.v16x32[0].h16x16[1]);
		load_fragment_trans(src, m_idx,      n_idx + 16, dst.v16x32[1].h16x16[0]);
		load_fragment_trans(src, m_idx + 16, n_idx + 16, dst.v16x32[1].h16x16[1]);
	}
}
