#pragma once

#include "utils_b8.cuh"

#pragma nv_diag_suppress 186

namespace b8 {
	template<
		typename T,
		const usize _GN,
		const usize _M, const usize _N,
		const usize _GMEM_PRELOAD = 2
	>
	requires(sizeof(T) == 1)
	struct MatrixLoaderEvenOdd {
		using Elem = T;

		static constexpr usize GN = _GN;
		static constexpr usize M = _M;
		static constexpr usize N = _N;
		static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

		static_assert(GN % N == 0);
		static_assert(M % 32 == 0);
		static_assert(N % 32 == 0);

		static constexpr usize SMEM_BYTES = M * N * GMEM_PRELOAD * sizeof(T);

		using GInput = GMatrixDynSize<T, GN>;
		using SPreload = SMatrixEvenOdd<T, M * GMEM_PRELOAD, N>;

		GInput gInput;
		SPreload sPreload;

		X17_DEVICE usize m_rows() const { return gInput.m_rows(); }
		X17_DEVICE usize n_cols() const { return gInput.n_cols(); }

		X17_DEVICE MatrixLoaderEvenOdd(T *gmem_addr, usize m_rows):
			gInput(gmem_addr, m_rows),
			sPreload()
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			sPreload._ptr = smem_alloc.alloc(SMEM_BYTES);
		}

		/// `cp_async` a tile with size [M, N] at position [M*m, N*n] into SMEM.
		/// `step` may be a global K-step; the shared-memory ring slot is selected
		/// modulo `GMEM_PRELOAD`.
		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize step, usize m, usize n) {
			GMatrix<T, M, N> src = gInput.template tile_m<M>(m).slice_n<N>(N*n);
			sPreload.template cp_async_from<THREADS_PER_BLOCK, M, N>(
				threadIdx.x,
				src,
				M * (step % GMEM_PRELOAD), 0,
				0, 0
			);
		}

		/// Load a 32x32 fragment at tile coordinates [m, n] from the SMEM ring buffer.
		X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_32x32<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			sPreload.tile_to_fragment(first_row + 32*m, 32*n, frag);
		}

		/// Load a transposed 32x32 fragment at tile coordinates [m, n] from the SMEM ring buffer.
		X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_32x32<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			sPreload.tile_to_fragment_trans(first_row + 32*m, 32*n, frag);
		}
	};

	template<typename Loader>
	struct MatrixTransLoader {
		using Elem = typename Loader::Elem;

		static constexpr usize M = Loader::N;
		static constexpr usize N = Loader::M;
		static constexpr usize GMEM_PRELOAD = Loader::GMEM_PRELOAD;
		static constexpr usize SMEM_BYTES = Loader::SMEM_BYTES;

		Loader loader;

		X17_DEVICE usize m_rows() const { return loader.n_cols(); }
		X17_DEVICE usize n_cols() const { return loader.m_rows(); }

		template<typename... Args>
		X17_DEVICE MatrixTransLoader(Args... args):
			loader(args...)
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			loader.alloc_smem(smem_alloc);
		}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize step, usize m, usize n) {
			loader.template cp_async<THREADS_PER_BLOCK>(step, n, m);
		}

		X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_32x32<Elem> &frag) {
			loader.load_fragment_trans(step, n, m, frag);
		}

		X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_32x32<Elem> &frag) {
			loader.load_fragment(step, n, m, frag);
		}
	};

	template<typename T, const usize GN>
	requires(sizeof(T) == 1)
	struct MatrixWriterEvenOdd {
		T *gC;
		usize c_stride;

		X17_DEVICE MatrixWriterEvenOdd(T *gC):
			gC(gC),
			c_stride(GN)
		{}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			Fragment_32x32<T> (&acc)[M_TILES][N_TILES]
		) {
			GMatrix<T, 32 * M_TILES, 32 * N_TILES> C(gC, c_stride);
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				store_even_odd(acc[mi], C, row + 32 * mi, col);
			}
		}
	};

	template<const usize GN, const f64 SCALE>
	struct FixedI8MatrixWriterEvenOdd: MatrixWriterEvenOdd<FixedI8, GN> {

		X17_DEVICE FixedI8MatrixWriterEvenOdd(FixedI8 *gC):
			MatrixWriterEvenOdd<FixedI8, GN>(gC)
		{}

		static X17_DEVICE i8 conv_one(i32 inp) {
			f32 val_f = f32(inp) * f32(SCALE / FIXED_I8_SCALE);
			i32 val_i = __float2int_rn(val_f);
			i8 val = val_i < -127 || val_i > +127 ? -128 : val_i; // TODO - inefficient
			return val;
		}

		X17_DEVICE void conv(b32::Fragment_16x16<i32> const &inp, Fragment_16x16<i8> &out) {
			union {
				u32 value;
				struct {
					i8 even_0, even_1, odd_0, odd_1;
				} packed;
			} left, right;

			left.packed.even_0 = conv_one(inp.v8x16[0].h8x8[0].val0);
			left.packed.even_1 = conv_one(inp.v8x16[0].h8x8[0].val1);
			left.packed.odd_0 = conv_one(inp.v8x16[1].h8x8[0].val0);
			left.packed.odd_1 = conv_one(inp.v8x16[1].h8x8[0].val1);

			right.packed.even_0 = conv_one(inp.v8x16[0].h8x8[1].val0);
			right.packed.even_1 = conv_one(inp.v8x16[0].h8x8[1].val1);
			right.packed.odd_0 = conv_one(inp.v8x16[1].h8x8[1].val0);
			right.packed.odd_1 = conv_one(inp.v8x16[1].h8x8[1].val1);

			usize tid = threadIdx.x;
			u32 u0 = left.value;
			u32 u1 = right.value;

			u0 = shuffle_xor_sync(u0, 2);

			u32 v0 = (tid & 2) == 0 ? u1 : u0;
			u32 v1 = (tid & 2) == 0 ? u0 : u1;

			v0 = shuffle_xor_sync(v0, 3);

			u32 w0 = (tid & 1) == 0 ? v1 : v0;
			u32 w1 = (tid & 1) == 0 ? v0 : v1;

			w0 = shuffle_xor_sync(w0, 1);

			out.v8x16[0].val = __byte_perm(w0, w1, 0x5410);
			out.v8x16[1].val = __byte_perm(w0, w1, 0x7632);
		}

		X17_DEVICE void conv(b32::Fragment_32x32<i32> const &inp, Fragment_32x32<i8> &out) {
			X17_UNROLL for (usize j = 0; j < 2; ++j) {
				X17_UNROLL for (usize i = 0; i < 2; ++i) {
					conv(inp.v16x32[j].h16x16[i], out.v16x32[j].h16x16[i]);
				}
			}
		}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
		) {
			Fragment_32x32<FixedI8> t[M_TILES][N_TILES];

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					conv(acc[mi][ni], t[mi][ni]);
				}
			}

			MatrixWriterEvenOdd<FixedI8, GN>::write(row, col, t);
		}
	};
}
