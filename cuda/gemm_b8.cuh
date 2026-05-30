#pragma once

#include "utils_b8.cuh"

#pragma nv_diag_suppress 186
#pragma nv_diag_suppress 179

namespace b8 {
	template<
		typename T,
		const usize _GN,
		const usize _M, const usize _K,
		const usize _FAN_IN = _GN,
		const usize _STEP = 0,
		const usize _BLOCK = 0,
		const bool _MODULO = false,
		const usize _GMEM_PRELOAD = 2
	>
	requires(sizeof(T) == 1)
	struct MatrixLoader {
		using Elem = T;

		static constexpr usize GN = _GN;
		static constexpr usize M = _M;
		static constexpr usize K = _K;
		static constexpr usize FAN_IN = _FAN_IN;
		static constexpr usize STEP = _STEP;
		static constexpr usize BLOCK = _BLOCK;
		static constexpr bool MODULO = _MODULO;
		static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

		static_assert(FAN_IN <= GN);
		static_assert(GN % K == 0);
		static_assert(M % 32 == 0);
		static_assert(K % 32 == 0);
		static_assert(FAN_IN % 32 == 0);
		static_assert(STEP == 0 || (GN % STEP == 0));
		static_assert(STEP == 0 || (BLOCK > 0));
		static_assert(STEP == 0 || (BLOCK % M == 0));

		static constexpr usize SMEM_BYTES = M * K * GMEM_PRELOAD * sizeof(T);

		using GInput = GMatrixDynSize<T, GN>;
		using SPreload = SMatrix<T, M * GMEM_PRELOAD, K>;

		usize _m_rows;
		GInput gInput;
		SPreload sPreload;

		X17_DEVICE usize m_rows() const { return _m_rows; }
		X17_DEVICE usize n_cols() const { return FAN_IN; }

		X17_DEVICE MatrixLoader(T *gmem_addr, usize m_rows):
			_m_rows(m_rows),
			gInput(gmem_addr),
			sPreload()
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			sPreload._ptr = smem_alloc.alloc(SMEM_BYTES);
		}

		/// `cp_async` a tile with size [M, K] at position [m, k] into SMEM.
		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize step, usize m, usize k, usize other_n) {
			usize k_shift = 0;
			if constexpr (BLOCK > 0) {
				k_shift = (other_n / BLOCK) * STEP;
			}
			sPreload.template cp_async_from<THREADS_PER_BLOCK, M, K, GN, MODULO>(
				threadIdx.x,
				gInput,
				m, k + k_shift,
				M * (step % GMEM_PRELOAD), 0
			);
		}

		/// Load a 32x32 fragment at tile coordinates [m, k] from the SMEM ring buffer.
		X17_DEVICE void load_fragment(usize step, usize m, usize k, Fragment_32x32<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			sPreload.tile_to_fragment(first_row + 32*m, 32*k, frag);
		}
	};

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

		/// `cp_async` a tile with size [M, K] at position [M*m, K*k] into SMEM.
		/// `step` may be a global K-step; the shared-memory ring slot is selected
		/// modulo `GMEM_PRELOAD`.
		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize step, usize m, usize k, usize other_n) {
			sPreload.template cp_async_from<THREADS_PER_BLOCK, M, K>(
				threadIdx.x,
				gInput,
				M * m, K * k,
				M * (step % GMEM_PRELOAD), 0
			);
		}

		/// Load a 32x32 fragment at tile coordinates [m, k] from the SMEM ring buffer.
		X17_DEVICE void load_fragment(usize step, usize m, usize k, Fragment_32x32<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			sPreload.tile_to_fragment(first_row + 32*m, 32*k, frag);
		}

		/// Load a transposed 32x32 fragment at tile coordinates [m, k] from the SMEM ring buffer.
		X17_DEVICE void load_fragment_trans(usize step, usize m, usize k, Fragment_32x32<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			sPreload.tile_to_fragment_trans(first_row + 32*m, 32*k, frag);
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
		X17_DEVICE void cp_async(usize step, usize m, usize k, usize other_n) {
			loader.template cp_async<THREADS_PER_BLOCK>(step, k, m, other_n);
		}

		X17_DEVICE void load_fragment(usize step, usize m, usize k, Fragment_32x32<Elem> &frag) {
			loader.load_fragment_trans(step, k, m, frag);
		}

		X17_DEVICE void load_fragment_trans(usize step, usize m, usize k, Fragment_32x32<Elem> &frag) {
			loader.load_fragment(step, k, m, frag);
		}
	};

	template<
		typename T,
		const usize _GN,
		const usize _M_PER_BLOCK = 0,
		const usize _N_PER_BLOCK = 0
	>
	requires(sizeof(T) == 1)
	struct MatrixWriter {
		static constexpr usize GN = _GN;
		static constexpr usize M_PER_BLOCK = _M_PER_BLOCK;
		static constexpr usize N_PER_BLOCK = _N_PER_BLOCK;
		static constexpr usize SMEM_BYTES = 0;

		T *gC;
		usize c_stride;

		X17_DEVICE MatrixWriter(T *gC):
			gC(gC),
			c_stride(GN)
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize row, usize col) {}

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

	template<const usize GN, const usize M_PER_BLOCK, const usize N_PER_BLOCK, const f64 SCALE>
	struct FixedI8MatrixWriter: MatrixWriter<FixedI8, GN, M_PER_BLOCK, N_PER_BLOCK> {
		using Base = MatrixWriter<FixedI8, GN, M_PER_BLOCK, N_PER_BLOCK>;

		X17_DEVICE FixedI8MatrixWriter(FixedI8 *gC):
			Base(gC)
		{}

		X17_DEVICE void conv(b32::Fragment_16x16<f32> const &inp, Fragment_16x16<FixedI8> &out) {
			union {
				u32 value;
				struct {
					FixedI8 top_0, top_1, bot_0, bot_1;
				} packed;
			} left, right;

			left.packed.top_0 = to_fixedi8(inp.v8x16[0].h8x8[0].val0);
			left.packed.top_1 = to_fixedi8(inp.v8x16[0].h8x8[0].val1);
			left.packed.bot_0 = to_fixedi8(inp.v8x16[1].h8x8[0].val0);
			left.packed.bot_1 = to_fixedi8(inp.v8x16[1].h8x8[0].val1);

			right.packed.top_0 = to_fixedi8(inp.v8x16[0].h8x8[1].val0);
			right.packed.top_1 = to_fixedi8(inp.v8x16[0].h8x8[1].val1);
			right.packed.bot_0 = to_fixedi8(inp.v8x16[1].h8x8[1].val0);
			right.packed.bot_1 = to_fixedi8(inp.v8x16[1].h8x8[1].val1);

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

		X17_DEVICE void conv(b32::Fragment_32x32<f32> const &inp, Fragment_32x32<FixedI8> &out) {
			X17_UNROLL for (usize j = 0; j < 2; ++j) {
				X17_UNROLL for (usize i = 0; i < 2; ++i) {
					conv(inp.v16x32[j].h16x16[i], out.v16x32[j].h16x16[i]);
				}
			}
		}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<f32> (&acc)[M_TILES][N_TILES]
		) {
			Fragment_32x32<FixedI8> t[M_TILES][N_TILES];

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					conv(acc[mi][ni], t[mi][ni]);
				}
			}

			Base::write(row, col, t);
		}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
		) {
			Fragment_32x32<FixedI8> t[M_TILES][N_TILES];

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					b32::Fragment_32x32<f32> f;
					cast<SCALE / FIXED_I8_SCALE>(acc[mi][ni], f);
					conv(f, t[mi][ni]);
				}
			}

			Base::write(row, col, t);
		}
	};

	template<const usize GN, const usize M_PER_BLOCK, const usize N_PER_BLOCK, const usize FAN_IN>
	struct FixedI8MatrixGeGluWriter: MatrixWriter<FixedI8, GN, M_PER_BLOCK, N_PER_BLOCK> {
		using Base = MatrixWriter<FixedI8, GN, M_PER_BLOCK, N_PER_BLOCK>;

		X17_DEVICE FixedI8MatrixGeGluWriter(FixedI8 *gC):
			Base(gC)
		{}

		X17_DEVICE FixedI8 geglu(b32::Fragment_8x8<i32> frag) {
			// `frag.val0` and `frag.val1` are raw accumulators of `sum(FixedI8 * FixedI8)`.
			// Each `FixedI8` is scaled by FIXED_I8_SCALE, so `FixedI8 * FixedI8` is scaled
			// by `FIXED_I8_SCALE^2`. We need to divide the inputs by this value.
			// We also need to divide by `sqrt(FAN_IN)` so the inputs have unit variance.
			//
			// `math::fast::gelu` expects squared scale factors, hence the `_2` suffix:
			// - `INP_SCALE_2` => gate scale `1 / (FIXED_I8_SCALE^2 * sqrt(FAN_IN))`
			//                 = `1 / sqrt(FAN_IN * FIXED_I8_SCALE^4)`
			// - `OUT_SCALE_2` is logically the scale of the `lin` input, but we apply it to the
			//   GELU output instead. That lets GELU fold the multiply into its constants, so the
			//   scale is effectively free at runtime. This scale factor
			//   is the same as `INP_SCALE_2` followed by a multiplication by `FIXED_I8_SCALE`
			//   which converts the final value into `FixedI8` again.
			constexpr f64 FIXED_I8_SCALE_2 = FIXED_I8_SCALE * FIXED_I8_SCALE;
			constexpr f64 INP_SCALE_2 = 1.0 / (FAN_IN * FIXED_I8_SCALE_2 * FIXED_I8_SCALE_2);
			constexpr f64 OUT_SCALE_2 = 1.0 / (FAN_IN * FIXED_I8_SCALE_2);
			f32 gate = math::fast::gelu<INP_SCALE_2, OUT_SCALE_2, 1.0>(f32(frag.val0)).val;
			f32 lin = f32(frag.val1);
			return to_fixedi8(gate * lin);
		}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
		) {
			static_assert(N_TILES % 2 == 0);

			union {
				struct {
					FixedI8 a, b, c, d;
				} packed;
				u32 value;
			} u1, u2, u3, u4, v1, v2, v3, v4;

			Fragment_32x32<FixedI8> t[M_TILES][N_TILES/2];
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
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

					t[mi][ni/2].v16x32[0].h16x16[ni%2].v8x16[0].val = v1.value;
					t[mi][ni/2].v16x32[0].h16x16[ni%2].v8x16[1].val = v2.value;
					t[mi][ni/2].v16x32[1].h16x16[ni%2].v8x16[0].val = v3.value;
					t[mi][ni/2].v16x32[1].h16x16[ni%2].v8x16[1].val = v4.value;
				}
			}

			Base::write(row, col / 2, t);
		}
	};

	template<const usize GN, const usize M_PER_BLOCK, const usize N_PER_BLOCK, const usize FAN_IN>
	struct FixedI8MatrixResidualWriter: MatrixWriter<FixedI8, GN, M_PER_BLOCK, N_PER_BLOCK> {
		using Base = MatrixWriter<FixedI8, GN, M_PER_BLOCK, N_PER_BLOCK>;
		using GResidual = GMatrixDynSize<FixedI8, GN>;
		using SResidual = SMatrix<FixedI8, M_PER_BLOCK, N_PER_BLOCK>;

		static constexpr usize SMEM_BYTES = M_PER_BLOCK * N_PER_BLOCK * sizeof(FixedI8);

		static_assert(N_PER_BLOCK % 2 == 0);

		GResidual gResidual;
		SResidual sResidual;
		usize residualRow0;
		usize residualCol0;

		X17_DEVICE FixedI8MatrixResidualWriter(FixedI8 *gC, FixedI8 *gResidual):
			Base(gC),
			gResidual(gResidual),
			sResidual(),
			residualRow0(0),
			residualCol0(0)
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			sResidual._ptr = smem_alloc.alloc(SMEM_BYTES);
		}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize row, usize col) {
			residualRow0 = (row / M_PER_BLOCK) * M_PER_BLOCK;
			residualCol0 = (col / N_PER_BLOCK) * (N_PER_BLOCK / 2);

			sResidual.template cp_async_from<THREADS_PER_BLOCK, M_PER_BLOCK, N_PER_BLOCK / 2, GN>(
				threadIdx.x,
				gResidual,
				residualRow0,
				residualCol0,
				0,
				0
			);
		}

		X17_DEVICE FixedI8 residual_gate(b32::Fragment_8x8<i32> frag, FixedI8 residual) {
			constexpr f64 SCALE = math::constexpr_sqrt(math::fast::GELU_VAR_FIX_2 / f64(FAN_IN));
			constexpr f64 RAW_TO_REAL = SCALE / (f64(FIXED_I8_SCALE) * f64(FIXED_I8_SCALE));
			constexpr f64 RAW_TO_FIXED = SCALE / f64(FIXED_I8_SCALE);

			f32 gate = math::fast::sigmoid(f32(frag.val0) * f32(RAW_TO_REAL));
			f32 residual_f = f32(residual);
			f32 output_f = f32(frag.val1) * f32(RAW_TO_FIXED);
			f32 val_f = math::fma(gate, output_f - residual_f, residual_f);
			return to_fixedi8(val_f);
		}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
		) {
			static_assert(N_TILES % 2 == 0);

			union Packed4 {
				struct {
					FixedI8 a, b, c, d;
				} packed;
				u32 value;
			} u1, u2, u3, u4, v1, v2, v3, v4, r1, r2, r3, r4, s1, s2, s3, s4;

			Fragment_32x32<FixedI8> residual[M_TILES][N_TILES/2];
			Fragment_32x32<FixedI8> t[M_TILES][N_TILES/2];
			usize local_row = row - residualRow0;
			usize local_col = (col / 2) - residualCol0;

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize no = 0; no < N_TILES / 2; ++no) {
					sResidual.tile_to_fragment(local_row + 32 * mi, local_col + 32 * no, residual[mi][no]);
				}
			}

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					auto &inp32x32 = acc[mi][ni];
					auto &res32x32 = residual[mi][ni / 2];

					// Residual fragments are loaded in the final stored layout.
					// Reconstruct the pre-store 8x8 packing so each residual byte lines up
					// with the corresponding gate/output accumulator pair.
					r1.value = res32x32.v16x32[0].h16x16[ni % 2].v8x16[0].val;
					r2.value = res32x32.v16x32[0].h16x16[ni % 2].v8x16[1].val;
					r3.value = res32x32.v16x32[1].h16x16[ni % 2].v8x16[0].val;
					r4.value = res32x32.v16x32[1].h16x16[ni % 2].v8x16[1].val;

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

					shuffle_4x4(s1.value, s2.value, s3.value, s4.value);

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

					t[mi][ni/2].v16x32[0].h16x16[ni%2].v8x16[0].val = v1.value;
					t[mi][ni/2].v16x32[0].h16x16[ni%2].v8x16[1].val = v2.value;
					t[mi][ni/2].v16x32[1].h16x16[ni%2].v8x16[0].val = v3.value;
					t[mi][ni/2].v16x32[1].h16x16[ni%2].v8x16[1].val = v4.value;
				}
			}

			Base::write(row, col / 2, t);
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
					A.template cp_async<THREADS_PER_BLOCK>(p, M_PER_BLOCK*block_m, K_STEP*p, N_PER_BLOCK*block_n);
					B.template cp_async<THREADS_PER_BLOCK>(p, K_STEP*p, N_PER_BLOCK*block_n, M_PER_BLOCK*block_m);
				}
				cp_async_commit();
			}

			Fragment_32x32<FixedI8> rA[M_TILES][K_TILES];
			Fragment_32x32<FixedI8> rBT[K_TILES][N_TILES];
			b32::Fragment_32x32<i32> acc[M_TILES][N_TILES];
			zero_(acc);

			cp_async_wait<GMEM_PRELOAD - 1>();
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
					cp_async_wait<GMEM_PRELOAD - 2>();
					sync_threads();

					usize p = k_step + GMEM_PRELOAD;
					if (p < K_ITERS) {
						A.template cp_async<THREADS_PER_BLOCK>(p, M_PER_BLOCK*block_m, K_STEP*p, N_PER_BLOCK*block_n);
						B.template cp_async<THREADS_PER_BLOCK>(p, K_STEP*p, N_PER_BLOCK*block_n, M_PER_BLOCK*block_m);
					} else if (k_step + 1 >= K_ITERS) {
						if constexpr (Writer::SMEM_BYTES > 0) {
							C.template cp_async<THREADS_PER_BLOCK>(output_row, output_col);
						}
					}
					cp_async_commit();
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
				cp_async_wait<0>();
				sync_threads();
			}
			C.write(output_row, output_col, acc);
		}
	};
}
