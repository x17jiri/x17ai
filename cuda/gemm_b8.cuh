#pragma once

#include "utils_b8.cuh"

#pragma nv_diag_suppress 186
#pragma nv_diag_suppress 179

namespace b8 {
	template<
		typename T,
		const usize _GN, // number of columns of the input matrix in GMEM
		const usize _M, const usize _K, // size of preload tile
		const usize _GMEM_PRELOAD = 2 // number of preload tiles
	>
	requires(sizeof(T) == 1)
	struct MatrixLoader {
		using Elem = T;

		static constexpr usize GN = _GN;
		static constexpr usize M = _M;
		static constexpr usize K = _K;
		static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

		static_assert(GN % K == 0);
		static_assert(M % 32 == 0);
		static_assert(K % 32 == 0);

		static constexpr usize SMEM_BYTES = M * K * GMEM_PRELOAD * sizeof(T);

		using GInput = GMatrixDynSize<T, GN>;
		using SPreload = SMatrix<T, M * GMEM_PRELOAD, K>;

		usize _m_rows;
		GInput gInput;
		SPreload sPreload;

		X17_DEVICE usize m_rows() const { return _m_rows; }
		X17_DEVICE usize n_cols() const { return GN; }

		X17_DEVICE MatrixLoader(T *gmem_addr, usize m_rows):
			_m_rows(m_rows),
			gInput(gmem_addr),
			sPreload()
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			sPreload._ptr = smem_alloc.alloc(SMEM_BYTES);
		}

		/// `async_load` a tile with size [M, K] at position [m, k] into SMEM.
		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void async_load(usize step, usize m, usize k) {
			b8::async_load<THREADS_PER_BLOCK, M, K>(
				threadIdx.x,
				gInput, m, k,
				sPreload, M * (step % GMEM_PRELOAD), 0
			);
		}

		/// Load a fragment at tile coordinates [m, k] from the SMEM ring buffer.
		X17_DEVICE void load_fragment(usize step, usize m, usize k, Fragment_32x32<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			b8::load_fragment(sPreload, first_row + 32*m, 32*k, frag);
		}
	};

	template<
		const usize GN,
		const usize M, const usize K,
		const usize GMEM_PRELOAD = 2
	>
	using FixedI8MatrixLoader = MatrixLoader<FixedI8, GN, M, K, GMEM_PRELOAD>;

	template<
		typename T,
		const usize _GN,
		const usize _M, const usize _K,
		const usize _GMEM_PRELOAD = 2
	>
	requires(sizeof(T) == 1)
	struct MatrixLoaderEvenOdd {
		using Elem = T;

		static constexpr usize GN = _GN;
		static constexpr usize M = _M;
		static constexpr usize K = _K;
		static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

		static_assert(GN % K == 0);
		static_assert(M % 32 == 0);
		static_assert(K % 32 == 0);

		static constexpr usize SMEM_BYTES = M * K * GMEM_PRELOAD * sizeof(T);

		using GInput = GMatrixDynSize<T, GN>;
		using SPreload = SMatrixEvenOdd<T, M * GMEM_PRELOAD, K>;

		usize _m_rows;
		GInput gInput;
		SPreload sPreload;

		X17_DEVICE usize m_rows() const { return _m_rows; }
		X17_DEVICE usize n_cols() const { return GN; }

		X17_DEVICE MatrixLoaderEvenOdd(T *gmem_addr, usize m_rows):
			_m_rows(m_rows),
			gInput(gmem_addr),
			sPreload()
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			sPreload._ptr = smem_alloc.alloc(SMEM_BYTES);
		}

		/// `async_load` a tile with size [M, K] at position [M*m, K*k] into SMEM.
		/// `step` may be a global K-step; the shared-memory ring slot is selected
		/// modulo `GMEM_PRELOAD`.
		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void async_load(usize step, usize m, usize k) {
			async_load<THREADS_PER_BLOCK, M, K>(
				threadIdx.x,
				gInput, M * m, K * k,
				sPreload, M * (step % GMEM_PRELOAD), 0
			);
		}

		/// Load a 32x32 fragment at tile coordinates [m, k] from the SMEM ring buffer.
		X17_DEVICE void load_fragment(usize step, usize m, usize k, Fragment_32x32<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			b8::load_fragment(sPreload, first_row + 32*m, 32*k, frag);
		}

		/// Load a transposed 32x32 fragment at tile coordinates [m, k] from the SMEM ring buffer.
		X17_DEVICE void load_fragment_trans(usize step, usize m, usize k, Fragment_32x32<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			b8::load_fragment_pretrans(sPreload, first_row + 32*m, 32*k, frag);
			frag.finish_trans_load_();
		}
	};

	template<typename Loader>
	struct MatrixTransLoader {
		using Elem = typename Loader::Elem;

		static constexpr usize M = Loader::K;
		static constexpr usize K = Loader::M;
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
		X17_DEVICE void async_load(usize step, usize m, usize k) {
			loader.template async_load<THREADS_PER_BLOCK>(step, k, m);
		}

		X17_DEVICE void load_fragment(usize step, usize m, usize k, Fragment_32x32<Elem> &frag) {
			loader.load_fragment_trans(step, k, m, frag);
		}

		X17_DEVICE void load_fragment_trans(usize step, usize m, usize k, Fragment_32x32<Elem> &frag) {
			loader.load_fragment(step, k, m, frag);
		}
	};

	template<typename Derived, typename Elem_>
	struct MatrixStore {
		using Elem = Elem_;
		static_assert(sizeof(Elem) == 1);

		Elem *gC;
		usize c_stride;

		X17_DEVICE MatrixStore(Elem *gC, usize c_stride):
			gC(gC),
			c_stride(c_stride)
		{}

		template<typename T>
		X17_DEVICE void conv(b32::Fragment_16x16<T> const &inp, Fragment_16x16<Elem> &out) {
			union {
				u32 value;
				struct {
					FixedI8 top_0, top_1, bot_0, bot_1;
				} packed;
			} left, right;

			left.packed.top_0 = Derived::convert(inp.v8x16[0].h8x8[0].get0());
			left.packed.top_1 = Derived::convert(inp.v8x16[0].h8x8[0].get1());
			left.packed.bot_0 = Derived::convert(inp.v8x16[1].h8x8[0].get0());
			left.packed.bot_1 = Derived::convert(inp.v8x16[1].h8x8[0].get1());

			right.packed.top_0 = Derived::convert(inp.v8x16[0].h8x8[1].get0());
			right.packed.top_1 = Derived::convert(inp.v8x16[0].h8x8[1].get1());
			right.packed.bot_0 = Derived::convert(inp.v8x16[1].h8x8[1].get0());
			right.packed.bot_1 = Derived::convert(inp.v8x16[1].h8x8[1].get1());

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

			out.v8x16[0].data = __byte_perm(w0, w1, 0x5410);
			out.v8x16[1].data = __byte_perm(w0, w1, 0x7632);
		}

		template<typename T>
		X17_DEVICE void conv(b32::Fragment_32x32<T> const &inp, Fragment_32x32<Elem> &out) {
			X17_UNROLL for (usize j = 0; j < 2; ++j) {
				X17_UNROLL for (usize i = 0; i < 2; ++i) {
					conv(inp.v16x32[j].h16x16[i], out.v16x32[j].h16x16[i]);
				}
			}
		}

		template<typename T, usize N_TILES>
		requires(sizeof(T) == 4)
		X17_DEVICE void conv_store(
			usize row, usize col,
			b32::Fragment_32x32<T> (&acc)[N_TILES]
		) {
			Fragment_32x32<Elem> t[N_TILES];
			X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
				conv(acc[ni], t[ni]);
			}
			direct_store(row, col, t);
		}

		template<usize N_TILES>
		X17_DEVICE void direct_store(
			usize row, usize col,
			Fragment_32x32<Elem> (&acc)[N_TILES]
		) {
			GMatrix<Elem, 32, 32*N_TILES> C(gC, c_stride);
			b8::store(acc, C, row, col);
		}
	};

	struct Int8Store: MatrixStore<Int8Store, FixedI8> {
		using Base = MatrixStore<Int8Store, FixedI8>;

		using Base::Elem;
		using Base::Base;
		using Base::conv_store;
		using Base::direct_store;

		X17_DEVICE static FixedI8 convert(f32 value) {
			return f32_to_fixedi8(value);
		}
	};

	struct E4m3Store: MatrixStore<E4m3Store, E4m3> {
		using Base = MatrixStore<E4m3Store, E4m3>;

		using Base::Elem;
		using Base::Base;
		using Base::conv_store;
		using Base::direct_store;

		X17_DEVICE static E4m3 convert(f32 value) {
			return f32_to_e4m3(value);
		}
	};

	template<typename Store, usize _M_PER_BLOCK, usize _N_PER_BLOCK, f64 SCALE>
	struct ScaledMatrixWriter: Store {
		static constexpr usize M_PER_BLOCK = _M_PER_BLOCK;
		static constexpr usize N_PER_BLOCK = _N_PER_BLOCK;
		static constexpr usize SMEM_BYTES = 0;

		X17_DEVICE ScaledMatrixWriter(typename Store::Elem *gC, usize c_stride):
			Store(gC, c_stride)
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void async_load(usize row, usize col) {}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<f32> (&acc)[M_TILES][N_TILES]
		) {
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				if constexpr (SCALE != 1.0) {
					Store::conv_store(row + 32*mi, col, acc);
				} else {
					b32::Fragment_32x32<f32> t[N_TILES];
					X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
						t[ni] = acc[mi][ni];
						b32::scale_(t[mi][ni], f32(SCALE));
					}
					Store::conv_store(row + 32*mi, col, t);
				}
			}
		}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
		) {
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				b32::Fragment_32x32<f32> t[N_TILES];
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					b32::cast<SCALE / FIXED_I8_SCALE>(acc[mi][ni], t[ni]);
				}
				Store::conv_store(row + 32*mi, col, t);
			}
		}
	};

	template<typename Store, usize _M_PER_BLOCK, usize _N_PER_BLOCK, f64 INP_SCALE, f64 OUT_SCALE>
	struct GeGluMatrixWriter: Store {
		static constexpr usize M_PER_BLOCK = _M_PER_BLOCK;
		static constexpr usize N_PER_BLOCK = _N_PER_BLOCK;
		static constexpr usize SMEM_BYTES = 0;

		X17_DEVICE GeGluMatrixWriter(typename Store::Elem *gC, usize c_stride):
			Store(gC, c_stride)
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void async_load(usize row, usize col) {}

		X17_DEVICE typename Store::Elem geglu(b32::Fragment_8x8<i32> frag) {
			// `frag.val0` and `frag.val1` are raw accumulators of `sum(FixedI8 * FixedI8)`.
			// Each `FixedI8` is scaled by FIXED_I8_SCALE, so `FixedI8 * FixedI8` is scaled
			// by `FIXED_I8_SCALE^2`. We need to divide the inputs by this value and
			// apply the configured real-value scales.
			//
			// `math::fast::gelu` expects squared scale factors, hence the `_2` suffix:
			// - `GEGLU_INP_SCALE_2` is the squared scale of the gate input
			// - `GEGLU_OUT_SCALE_2` is logically the squared scale of the `lin` input, but we
			//   apply it to the GELU output instead. That lets GELU fold the multiply into its
			//   constants, so the scale is effectively free at runtime.
			constexpr f64 FIXED_I8_SCALE_2 = FIXED_I8_SCALE * FIXED_I8_SCALE;
			constexpr f64 GEGLU_INP_SCALE_2 = (INP_SCALE * INP_SCALE) / (FIXED_I8_SCALE_2 * FIXED_I8_SCALE_2);
			constexpr f64 GEGLU_OUT_SCALE_2 = (OUT_SCALE * OUT_SCALE) / (FIXED_I8_SCALE_2 * FIXED_I8_SCALE_2);
			f32 gate = math::fast::gelu<GEGLU_INP_SCALE_2, GEGLU_OUT_SCALE_2, 1.0>(f32(frag.get0())).val;
			f32 lin = f32(frag.get1());
			return Store::convert(gate * lin);
		}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
		) {
			static_assert(N_TILES % 2 == 0);

			union {
				struct {
					typename Store::Elem a, b, c, d;
				} packed;
				u32 value;
			} u1, u2, u3, u4, v1, v2, v3, v4;

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				Fragment_32x32<typename Store::Elem> t[N_TILES/2];
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					u1.packed.a = geglu(acc[mi][ni].v16x32[0].h16x16[0].v8x16[0].h8x8[0]);
					u1.packed.b = geglu(acc[mi][ni].v16x32[0].h16x16[0].v8x16[1].h8x8[0]);
					u1.packed.c = geglu(acc[mi][ni].v16x32[1].h16x16[0].v8x16[0].h8x8[0]);
					u1.packed.d = geglu(acc[mi][ni].v16x32[1].h16x16[0].v8x16[1].h8x8[0]);

					u2.packed.a = geglu(acc[mi][ni].v16x32[0].h16x16[0].v8x16[0].h8x8[1]);
					u2.packed.b = geglu(acc[mi][ni].v16x32[0].h16x16[0].v8x16[1].h8x8[1]);
					u2.packed.c = geglu(acc[mi][ni].v16x32[1].h16x16[0].v8x16[0].h8x8[1]);
					u2.packed.d = geglu(acc[mi][ni].v16x32[1].h16x16[0].v8x16[1].h8x8[1]);

					u3.packed.a = geglu(acc[mi][ni].v16x32[0].h16x16[1].v8x16[0].h8x8[0]);
					u3.packed.b = geglu(acc[mi][ni].v16x32[0].h16x16[1].v8x16[1].h8x8[0]);
					u3.packed.c = geglu(acc[mi][ni].v16x32[1].h16x16[1].v8x16[0].h8x8[0]);
					u3.packed.d = geglu(acc[mi][ni].v16x32[1].h16x16[1].v8x16[1].h8x8[0]);

					u4.packed.a = geglu(acc[mi][ni].v16x32[0].h16x16[1].v8x16[0].h8x8[1]);
					u4.packed.b = geglu(acc[mi][ni].v16x32[0].h16x16[1].v8x16[1].h8x8[1]);
					u4.packed.c = geglu(acc[mi][ni].v16x32[1].h16x16[1].v8x16[0].h8x8[1]);
					u4.packed.d = geglu(acc[mi][ni].v16x32[1].h16x16[1].v8x16[1].h8x8[1]);

					shuffle_4x4(u1.value, u2.value, u3.value, u4.value);

					v1.packed.a = u1.packed.a;
					v1.packed.b = u2.packed.a;
					v1.packed.c = u3.packed.a;
					v1.packed.d = u4.packed.a;

					v2.packed.a = u1.packed.b;
					v2.packed.b = u2.packed.b;
					v2.packed.c = u3.packed.b;
					v2.packed.d = u4.packed.b;

					v3.packed.a = u1.packed.c;
					v3.packed.b = u2.packed.c;
					v3.packed.c = u3.packed.c;
					v3.packed.d = u4.packed.c;

					v4.packed.a = u1.packed.d;
					v4.packed.b = u2.packed.d;
					v4.packed.c = u3.packed.d;
					v4.packed.d = u4.packed.d;

					t[ni/2].v16x32[0].h16x16[ni%2].v8x16[0].data = v1.value;
					t[ni/2].v16x32[0].h16x16[ni%2].v8x16[1].data = v2.value;
					t[ni/2].v16x32[1].h16x16[ni%2].v8x16[0].data = v3.value;
					t[ni/2].v16x32[1].h16x16[ni%2].v8x16[1].data = v4.value;
				}
				Store::direct_store(row + 32*mi, col / 2, t);
			}
		}
	};

	template<typename Store, usize _M_PER_BLOCK, usize _N_PER_BLOCK, f64 SCALE>
	struct ResidualMatrixWriter: Store {
		using GResidual = GMatrixDynSize<FixedI8, _N_PER_BLOCK / 2>;
		using SResidual = SMatrix<FixedI8, _M_PER_BLOCK, _N_PER_BLOCK>;

		static constexpr usize M_PER_BLOCK = _M_PER_BLOCK;
		static constexpr usize N_PER_BLOCK = _N_PER_BLOCK;
		static constexpr usize SMEM_BYTES = _M_PER_BLOCK * _N_PER_BLOCK * sizeof(FixedI8);

		static_assert(_N_PER_BLOCK % 2 == 0);

		GResidual gResidual;
		SResidual sResidual;
		usize residualRow0;
		usize residualCol0;

		X17_DEVICE ResidualMatrixWriter(typename Store::Elem *gC, FixedI8 *gResidual, usize c_stride):
			Store(gC, c_stride),
			gResidual(gResidual, c_stride),
			sResidual(),
			residualRow0(0),
			residualCol0(0)
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			sResidual._ptr = smem_alloc.alloc(SMEM_BYTES);
		}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void async_load(usize row, usize col) {
			residualRow0 = (row / M_PER_BLOCK) * M_PER_BLOCK;
			residualCol0 = (col / N_PER_BLOCK) * (N_PER_BLOCK / 2);

			b8::async_load<THREADS_PER_BLOCK, M_PER_BLOCK, N_PER_BLOCK / 2>(
				threadIdx.x,
				gResidual, residualRow0, residualCol0,
				sResidual, 0, 0
			);
		}

		template<typename A>
		X17_DEVICE typename Store::Elem residual_gate(b32::Fragment_8x8<A> frag, FixedI8 residual) {
			f32 gate = f32(frag.get0());

			f32 old_weight =
				math::fast::sigmoid_base4<
					-SCALE
				>(gate);

			f32 new_weight =
				math::fast::imprecise_softplus_base4<
					SCALE,
					SCALE * FIXED_I8_SCALE
				>(gate);

			f32 residual_f = f32(residual);
			f32 output_f = f32(frag.get1());

			return Store::convert(math::fma(new_weight, output_f, old_weight * residual_f));
		}

		template<const usize M_TILES, const usize N_TILES, typename A>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<A> (&acc)[M_TILES][N_TILES]
		) {
			static_assert(N_TILES % 2 == 0);

			union PackedOut4 {
				struct {
					typename Store::Elem a, b, c, d;
				} packed;
				u32 data;
			} u1, u2, u3, u4, v1, v2, v3, v4;

			union PackedResidual4 {
				struct {
					FixedI8 a, b, c, d;
				} packed;
				u32 data;
			} r1, r2, r3, r4, s1, s2, s3, s4;

			usize local_row = row - residualRow0;
			usize local_col = (col / 2) - residualCol0;

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				Fragment_32x32<FixedI8> residual[N_TILES/2];
				X17_UNROLL for (usize ni = 0; ni < N_TILES / 2; ++ni) {
					b8::load_fragment(sResidual, local_row + 32 * mi, local_col + 32 * ni, residual[ni]);
				}

				Fragment_32x32<typename Store::Elem> t[N_TILES/2];
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					auto &inp32x32 = acc[mi][ni];
					auto &res32x32 = residual[ni / 2];

					// Residual fragments are loaded in the final stored layout.
					// Reconstruct the pre-store 8x8 packing so each residual byte lines up
					// with the corresponding gate/output accumulator pair.
					r1.data = res32x32.v16x32[0].h16x16[ni % 2].v8x16[0].data;
					r2.data = res32x32.v16x32[0].h16x16[ni % 2].v8x16[1].data;
					r3.data = res32x32.v16x32[1].h16x16[ni % 2].v8x16[0].data;
					r4.data = res32x32.v16x32[1].h16x16[ni % 2].v8x16[1].data;

					s1.packed.a = r1.packed.a;
					s1.packed.b = r2.packed.a;
					s1.packed.c = r3.packed.a;
					s1.packed.d = r4.packed.a;

					s2.packed.a = r1.packed.b;
					s2.packed.b = r2.packed.b;
					s2.packed.c = r3.packed.b;
					s2.packed.d = r4.packed.b;

					s3.packed.a = r1.packed.c;
					s3.packed.b = r2.packed.c;
					s3.packed.c = r3.packed.c;
					s3.packed.d = r4.packed.c;

					s4.packed.a = r1.packed.d;
					s4.packed.b = r2.packed.d;
					s4.packed.c = r3.packed.d;
					s4.packed.d = r4.packed.d;

					shuffle_4x4(s1.data, s2.data, s3.data, s4.data);

					u1.packed.a = residual_gate(inp32x32.v16x32[0].h16x16[0].v8x16[0].h8x8[0], s1.packed.a);
					u1.packed.b = residual_gate(inp32x32.v16x32[0].h16x16[0].v8x16[1].h8x8[0], s1.packed.b);
					u1.packed.c = residual_gate(inp32x32.v16x32[1].h16x16[0].v8x16[0].h8x8[0], s1.packed.c);
					u1.packed.d = residual_gate(inp32x32.v16x32[1].h16x16[0].v8x16[1].h8x8[0], s1.packed.d);

					u2.packed.a = residual_gate(inp32x32.v16x32[0].h16x16[0].v8x16[0].h8x8[1], s2.packed.a);
					u2.packed.b = residual_gate(inp32x32.v16x32[0].h16x16[0].v8x16[1].h8x8[1], s2.packed.b);
					u2.packed.c = residual_gate(inp32x32.v16x32[1].h16x16[0].v8x16[0].h8x8[1], s2.packed.c);
					u2.packed.d = residual_gate(inp32x32.v16x32[1].h16x16[0].v8x16[1].h8x8[1], s2.packed.d);

					u3.packed.a = residual_gate(inp32x32.v16x32[0].h16x16[1].v8x16[0].h8x8[0], s3.packed.a);
					u3.packed.b = residual_gate(inp32x32.v16x32[0].h16x16[1].v8x16[1].h8x8[0], s3.packed.b);
					u3.packed.c = residual_gate(inp32x32.v16x32[1].h16x16[1].v8x16[0].h8x8[0], s3.packed.c);
					u3.packed.d = residual_gate(inp32x32.v16x32[1].h16x16[1].v8x16[1].h8x8[0], s3.packed.d);

					u4.packed.a = residual_gate(inp32x32.v16x32[0].h16x16[1].v8x16[0].h8x8[1], s4.packed.a);
					u4.packed.b = residual_gate(inp32x32.v16x32[0].h16x16[1].v8x16[1].h8x8[1], s4.packed.b);
					u4.packed.c = residual_gate(inp32x32.v16x32[1].h16x16[1].v8x16[0].h8x8[1], s4.packed.c);
					u4.packed.d = residual_gate(inp32x32.v16x32[1].h16x16[1].v8x16[1].h8x8[1], s4.packed.d);

					shuffle_4x4(u1.data, u2.data, u3.data, u4.data);

					v1.packed.a = u1.packed.a;
					v1.packed.b = u2.packed.a;
					v1.packed.c = u3.packed.a;
					v1.packed.d = u4.packed.a;

					v2.packed.a = u1.packed.b;
					v2.packed.b = u2.packed.b;
					v2.packed.c = u3.packed.b;
					v2.packed.d = u4.packed.b;

					v3.packed.a = u1.packed.c;
					v3.packed.b = u2.packed.c;
					v3.packed.c = u3.packed.c;
					v3.packed.d = u4.packed.c;

					v4.packed.a = u1.packed.d;
					v4.packed.b = u2.packed.d;
					v4.packed.c = u3.packed.d;
					v4.packed.d = u4.packed.d;

					t[ni/2].v16x32[0].h16x16[ni%2].v8x16[0].data = v1.data;
					t[ni/2].v16x32[0].h16x16[ni%2].v8x16[1].data = v2.data;
					t[ni/2].v16x32[1].h16x16[ni%2].v8x16[0].data = v3.data;
					t[ni/2].v16x32[1].h16x16[ni%2].v8x16[1].data = v4.data;
				}
				Store::direct_store(row + 32*mi, col / 2, t);
			}
		}
	};

	template<typename _ALoader, typename _BLoader, typename _Writer>
	struct Gemm {
		using ALoader = _ALoader;
		using BLoader = _BLoader;
		using Writer = _Writer;

		static constexpr usize M_PER_BLOCK = ALoader::M;
		static constexpr usize N_PER_BLOCK = BLoader::K;
		static constexpr usize K_STEP = ALoader::K;
		static_assert(BLoader::M == K_STEP);

		static constexpr usize M_WARPS = 2;
		static constexpr usize N_WARPS = 2;
		static constexpr usize WARPS_PER_BLOCK = M_WARPS * N_WARPS;
		static constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
		static constexpr usize M_PER_WARP = M_PER_BLOCK / M_WARPS;
		static constexpr usize N_PER_WARP = N_PER_BLOCK / N_WARPS;

		static constexpr usize M_TILES = M_PER_WARP / 32;
		static constexpr usize N_TILES = N_PER_WARP / 32;
		static constexpr usize K_TILES = K_STEP / 32;
		static_assert(Writer::M_PER_BLOCK == M_PER_BLOCK);
		static_assert(Writer::N_PER_BLOCK == N_PER_BLOCK);
		static_assert(M_TILES * M_WARPS * 32 == M_PER_BLOCK);
		static_assert(N_TILES * N_WARPS * 32 == N_PER_BLOCK);

		static constexpr usize SMEM_BYTES =
			std::max<usize>(
				ALoader::SMEM_BYTES + BLoader::SMEM_BYTES,
				Writer::SMEM_BYTES
			);
		static constexpr usize GMEM_PRELOAD = ALoader::GMEM_PRELOAD;
		static_assert(ALoader::GMEM_PRELOAD == BLoader::GMEM_PRELOAD);

		X17_HOST_DEVICE static bool has_full_output_tiles(usize output_rows, usize output_cols) {
			return output_rows % M_PER_BLOCK == 0 && output_cols % N_PER_BLOCK == 0;
		}

		X17_HOST_DEVICE static dim3 output_grid(usize output_rows, usize output_cols) {
			return dim3(output_rows / M_PER_BLOCK, output_cols / N_PER_BLOCK);
		}

		X17_DEVICE void run(
			ALoader &A,
			BLoader &B,
			Writer &C
		) {
			usize K_ITERS = std::min<usize>(A.n_cols(), B.m_rows()) / K_STEP;

			usize block_m = blockIdx.x;
			usize block_n = blockIdx.y;
			usize tid = threadIdx.x;
			usize warp_idx = tid / WARP_SIZE;
			usize warp_m = (warp_idx / N_WARPS);
			usize warp_n = (warp_idx % N_WARPS);

			SMemAllocator<SMEM_BYTES> smem_alloc;
			A.alloc_smem(smem_alloc);
			B.alloc_smem(smem_alloc);
			smem_alloc.finish();
			SMemAllocator<SMEM_BYTES> smem_alloc2;
			C.alloc_smem(smem_alloc2);
			smem_alloc.finish();

			X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
				if (p < K_ITERS) {
					A.template async_load<THREADS_PER_BLOCK>(p, M_PER_BLOCK*block_m, K_STEP*p);
					B.template async_load<THREADS_PER_BLOCK>(p, K_STEP*p, N_PER_BLOCK*block_n);
				}
				async_load_commit();
			}

			Fragment_32x32<FixedI8> rA[M_TILES][K_TILES];
			Fragment_32x32<FixedI8> rBT[K_TILES][N_TILES];
			b32::Fragment_32x32<i32> acc[M_TILES][N_TILES];
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					zero_(acc[mi][ni]);
				}
			}

			async_load_wait<GMEM_PRELOAD - 1>();
			sync_threads();

			X17_UNROLL for (usize ki = 0; ki < K_TILES; ++ki) {
				X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
					A.load_fragment(0, warp_m * M_TILES + mi, ki, rA[mi][ki]);
				}
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					B.load_fragment_trans(0, ki, warp_n * N_TILES + ni, rBT[ki][ni]);
				}
			}

			usize output_row = block_m * M_PER_BLOCK + warp_m * M_PER_WARP;
			usize output_col = block_n * N_PER_BLOCK + warp_n * N_PER_WARP;
			for (usize k_step = 0; k_step < K_ITERS; ++k_step) {
				{ // Get more data from GMEM
					async_load_wait<GMEM_PRELOAD - 2>();
					sync_threads();

					usize p = k_step + GMEM_PRELOAD;
					if (p < K_ITERS) {
						A.template async_load<THREADS_PER_BLOCK>(p, M_PER_BLOCK*block_m, K_STEP*p);
						B.template async_load<THREADS_PER_BLOCK>(p, K_STEP*p, N_PER_BLOCK*block_n);
					} else if (k_step + 1 >= K_ITERS) {
						if constexpr (Writer::SMEM_BYTES > 0) {
							C.template async_load<THREADS_PER_BLOCK>(output_row, output_col);
						}
					}
					async_load_commit();
				}

				X17_UNROLL for (usize ki = 0; ki < K_TILES; ++ki) {
					X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
						X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
							mma_a_bt(rA[mi][ki], rBT[ki][ni], acc[mi][ni]);
						}
						A.load_fragment(k_step + 1, warp_m * M_TILES + mi, ki, rA[mi][ki]);
					}
					X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
						B.load_fragment_trans(k_step + 1, ki, warp_n * N_TILES + ni, rBT[ki][ni]);
					}
				}
			}

			if constexpr (Writer::SMEM_BYTES > 0) {
				async_load_wait<0>();
				sync_threads();
			}
			C.write(output_row, output_col, acc);
		}
	};

	namespace kv_helpers {
		// Calculates L2 Norm of `acc[..][pos .. pos+W_TILES]`, multiplies by `SCALE`
		// and stores the result to `out[..][pos .. pos+W_TILES]`
		template<
			f64 SCALE, bool USE_DYN_SCALE, f64 EPS,
			bool STORE_RRMS,
			usize RRMS_COLS,
			usize W_TILES,
			usize M_TILES, usize N_TILES
		>
		X17_DEVICE void l2_norm(
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES],
			b32::Fragment_32x32<f32> (&out)[M_TILES][N_TILES],
			usize pos,
			u32 dyn_scales_ptr,
			f32 (&rrms)[4 * M_TILES][RRMS_COLS],
			usize rrms_col = 0
		) {
			bf16 dyn_scales0[4 * N_TILES];
			bf16 dyn_scales1[4 * N_TILES];
			usize tid = threadIdx.x;
			if constexpr (USE_DYN_SCALE) {
				dyn_scales_ptr += (tid % 4) * (2 * sizeof(bf16));
				X17_UNROLL for (usize ni = pos; ni < pos + W_TILES; ++ni) {
					X17_UNROLL for (usize i = 0; i < 4; ++i) {
						load_shared_2x16b(dyn_scales_ptr, dyn_scales0[4*ni + i], dyn_scales1[4*ni + i]);
						dyn_scales_ptr += 8 * sizeof(bf16);
					}
				}
			}
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				f32 sum[4] = {0.0, 0.0, 0.0, 0.0};
				X17_UNROLL for (usize ni = pos; ni < pos + W_TILES; ++ni) {
					auto &inp32x32 = acc[mi][ni];
					auto &out32x32 = out[mi][ni];
					X17_UNROLL for (usize j = 0; j < 4; ++j) {
						X17_UNROLL for (usize i = 0; i < 4; ++i) {
							auto &inp = inp32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
							auto &out = out32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
							out.set0(f32(inp.get0()));
							out.set1(f32(inp.get1()));
							sum[j] = math::fma(out.get0(), out.get0(), sum[j]);
							sum[j] = math::fma(out.get1(), out.get1(), sum[j]);
						}
					}
				}
				f32 scale[4] = {1.0, 1.0, 1.0, 1.0};
				X17_UNROLL for (usize j = 0; j < 4; ++j) {
					sum[j] += shuffle_xor_sync(sum[j], 1);
					sum[j] += shuffle_xor_sync(sum[j], 2);
					scale[j] = math::fast::rsqrt(sum[j] + f32(EPS)) * f32(SCALE);
					if constexpr (STORE_RRMS) {
						if ((tid % 4) == 0) {
							rrms[4 * mi + j][rrms_col] = scale[j];
						}
					}
				}
				X17_UNROLL for (usize ni = pos; ni < pos + W_TILES; ++ni) {
					auto &out32x32 = out[mi][ni];
					X17_UNROLL for (usize j = 0; j < 4; ++j) {
						X17_UNROLL for (usize i = 0; i < 4; ++i) {
							auto &out = out32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
							if constexpr (USE_DYN_SCALE) {
								out.set0(out.get0() * scale[j] * f32(dyn_scales0[4*ni + i]));
								out.set1(out.get1() * scale[j] * f32(dyn_scales1[4*ni + i]));
							} else {
								out.set0(out.get0() * scale[j]);
								out.set1(out.get1() * scale[j]);
							}
						}
					}
				}
			}
		}

		// Casts raw accumulators `acc[..][pos .. pos+W_TILES]`, multiplies by `SCALE`
		// and stores the result to `out[..][pos .. pos+W_TILES]`
		template<
			f64 SCALE,
			usize W_TILES,
			usize M_TILES, usize N_TILES
		>
		X17_DEVICE void raw_output(
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES],
			b32::Fragment_32x32<f32> (&out)[M_TILES][N_TILES],
			usize pos
		) {
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = pos; ni < pos + W_TILES; ++ni) {
					auto &inp32x32 = acc[mi][ni];
					auto &out32x32 = out[mi][ni];
					X17_UNROLL for (usize j = 0; j < 4; ++j) {
						X17_UNROLL for (usize i = 0; i < 4; ++i) {
							auto &inp = inp32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
							auto &out = out32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
							out.set0(f32(inp.get0()) * f32(SCALE));
							out.set1(f32(inp.get1()) * f32(SCALE));
						}
					}
				}
			}
		}
	}

	template<
		typename Store,
		usize HEAD_DIM,
		usize SEP_DIM,
		f64 EPS, f64 HEAD_SCALE,
		f64 SEP_SCALE,
		usize _M_PER_BLOCK,
		usize _N_PER_BLOCK
	>
	struct L2NormMatrixWriter: Store {
		static constexpr usize M_PER_BLOCK = _M_PER_BLOCK;
		static constexpr usize N_PER_BLOCK = _N_PER_BLOCK;
		static constexpr usize SMEM_BYTES = 0;

		static_assert(HEAD_DIM % 32 == 0);
		static_assert(SEP_DIM % 32 == 0);
		static_assert(HEAD_DIM == SEP_DIM);
		static_assert(_N_PER_BLOCK % (HEAD_DIM + SEP_DIM) == 0);

		f32 *g_rrms_ptr;
		usize rrms_cols;

		X17_DEVICE L2NormMatrixWriter(
			typename Store::Elem *gC,
			usize c_stride,
			f32 *g_rrms_ptr
		):
			Store(gC, c_stride),
			g_rrms_ptr(g_rrms_ptr),
			rrms_cols(c_stride / (HEAD_DIM + SEP_DIM))
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void async_load(usize row, usize col) {}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
		) {
			b32::Fragment_32x32<f32> t[M_TILES][N_TILES];

			constexpr usize K_TILES = HEAD_DIM / 32;
			constexpr usize V_TILES = SEP_DIM / 32;
			constexpr usize KV_TILES = K_TILES + V_TILES;
			constexpr usize HEADS = N_TILES / KV_TILES;
			static_assert(N_TILES % KV_TILES == 0);
			f32 rrms[4 * M_TILES][HEADS];

			// k
			constexpr f64 K_SCALE = HEAD_SCALE * f64(FIXED_I8_SCALE);
			X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
				kv_helpers::l2_norm<K_SCALE, false, EPS, true, HEADS, K_TILES>(
					acc,
					t,
					hi * KV_TILES,
					0,
					rrms,
					hi
				);
			}

			// v
			constexpr f64 V_SCALE = SEP_SCALE / f64(FIXED_I8_SCALE);
			X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
				kv_helpers::raw_output<V_SCALE, V_TILES>(
					acc,
					t,
					hi * KV_TILES + K_TILES
				);
			}

			usize tid = threadIdx.x % WARP_SIZE;
			if ((tid % 4) == 0) {
				usize head0 = col / (HEAD_DIM + SEP_DIM);
				X17_UNROLL for (usize ri = 0; ri < 4 * M_TILES; ++ri) {
					usize rrms_row = row + 8 * ri + tid / 4;
					store_gmem_Nx32b(g_rrms_ptr + head0 + (rrms_row * rrms_cols), rrms[ri]);
				}
			}

			for (usize mi = 0; mi < M_TILES; ++mi) {
				Store::conv_store(row + 32*mi, col, t[mi]);
			}
		}
	};
}
