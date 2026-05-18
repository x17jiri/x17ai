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
	struct MatrixLoader {
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
		using SPreload = SMatrix<T, M * GMEM_PRELOAD, N>;

		GInput gInput;
		SPreload sPreload;

		X17_DEVICE usize m_rows() const { return gInput.m_rows(); }
		X17_DEVICE usize n_cols() const { return gInput.n_cols(); }

		X17_DEVICE MatrixLoader(T *gmem_addr, usize m_rows):
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
			auto slot = sPreload.template tile_m<M>(step % GMEM_PRELOAD);
			GMatrix<T, M, N> src = gInput.template tile_m<M>(m).slice_n<N>(N*n);
			slot.template cp_async_from<THREADS_PER_BLOCK>(
				threadIdx.x,
				src,
				0, 0, 0, 0
			);
		}

		/// Load a 32x32 fragment at tile coordinates [m, n] from the SMEM ring buffer.
		X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_32x32<T> &frag) {
			auto slot = sPreload.template tile_m<M>(step % GMEM_PRELOAD);
			slot.tile_to_fragment(32*m, 32*n, frag);
		}

		/// Load a transposed 32x32 fragment at tile coordinates [m, n] from the SMEM ring buffer.
		X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_32x32<T> &frag) {
			auto slot = sPreload.template tile_m<M>(step % GMEM_PRELOAD);
			slot.tile_to_fragment_trans(32*m, 32*n, frag);
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
	struct MatrixWriter {
		T *gC;
		usize c_stride;

		X17_DEVICE MatrixWriter(T *gC):
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
				store(acc[mi], C, row + 32 * mi, col);
			}
		}
	};

	template<const usize GN, const f64 SCALE>
	struct FixedI8MatrixWriter: MatrixWriter<FixedI8, GN> {

		X17_DEVICE FixedI8MatrixWriter(FixedI8 *gC):
			MatrixWriter<FixedI8, GN>(gC)
		{}

		i8 conv_one(i32 inp) {
			f32 val_f = f32(inp) * f32(SCALE / FIXED_I8_SCALE);
			i32 val_i = __float2int_rn(val_f);
			i8 val = val_i < -127 || val_i > +127 ? -128 : val; // TODO - inefficient
			return val;
		}

		void conv(b32::Fragment_16x16<i32> const &inp, Fragment_16x16<i8> &out) {
			union {
				u32 value;
				struct {
					i8 top_0, top_1, bot_0, bot_1;
				} packed;
			} left, right;

			left.top_0 = conv_one(inp.v8x16[0].h8x8[0].val0);
			left.top_1 = conv_one(inp.v8x16[0].h8x8[0].val1);
			left.bot_0 = conv_one(inp.v8x16[1].h8x8[0].val0);
			left.bot_1 = conv_one(inp.v8x16[1].h8x8[0].val1);

			right.top_0 = conv_one(inp.v8x16[0].h8x8[1].val0);
			right.top_1 = conv_one(inp.v8x16[0].h8x8[1].val1);
			right.bot_0 = conv_one(inp.v8x16[1].h8x8[1].val0);
			right.bot_1 = conv_one(inp.v8x16[1].h8x8[1].val1);

			// TODO - shuffle
		}

		void conv(b32::Fragment_32x32<i32> const &inp, Fragment_32x32<i8> &out) {
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

			MatrixWriter<FixedI8, GN>::write(row, col, t);
		}
	};
}
