#pragma once

#include "utils_b8.cuh"

#pragma nv_diag_suppress 186

namespace b8 {
	template<
		typename T,
		const usize _GN,
		const usize _M, const usize _K,
		const usize _FAN_IN = _GN,
		const usize _D_OUT = 0,
		const usize _GMEM_PRELOAD = 2
	>
	requires(sizeof(T) == 1)
	struct MatrixLoader {
		using Elem = T;

		static constexpr usize GN = _GN;
		static constexpr usize M = _M;
		static constexpr usize K = _K;
		static constexpr usize FAN_IN = _FAN_IN;
		static constexpr usize STEP = FAN_IN < GN ? FAN_IN / 2 : 0;
		static constexpr usize STEPS = FAN_IN < GN ? GN / STEP : 1;
		static constexpr usize CNT_PER_STEP = _D_OUT / STEPS;
		static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

		static_assert(GN % K == 0);
		static_assert(M % 32 == 0);
		static_assert(K % 32 == 0);
		static_assert(FAN_IN % 32 == 0);

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
		/// `step` may be a global K-step; the shared-memory ring slot is selected
		/// modulo `GMEM_PRELOAD`.
		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize step, usize m, usize k, usize other_n) {
			constexpr bool GMEM_COL_MODULO = STEPS > 1;
			usize k_shift = 0;
			if constexpr (GMEM_COL_MODULO) {
				static_assert(CNT_PER_STEP * STEPS == _D_OUT);
				static_assert(CNT_PER_STEP % M == 0);
				k_shift = other_n / CNT_PER_STEP * STEP;
			}
			sPreload.template cp_async_from<THREADS_PER_BLOCK, M, K, GN, GMEM_COL_MODULO>(
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

		static X17_DEVICE i8 conv_one_f32(f32 inp) {
			inp = fmaxf(-127.0f, fminf(+127.0f, inp));
			return __float2int_rn(inp);
		}

		static X17_DEVICE i8 conv_one(i32 inp) {
			f32 val_f = f32(inp) * f32(SCALE / FIXED_I8_SCALE);
			val_f = fmaxf(-127.0f, fminf(+127.0f, val_f));
			return __float2int_rn(val_f);
		}

		X17_DEVICE void conv(b32::Fragment_16x16<f32> const &inp, Fragment_16x16<i8> &out) {
			union {
				u32 value;
				struct {
					i8 top_0, top_1, bot_0, bot_1;
				} packed;
			} left, right;

			left.packed.top_0 = conv_one_f32(inp.v8x16[0].h8x8[0].val0);
			left.packed.top_1 = conv_one_f32(inp.v8x16[0].h8x8[0].val1);
			left.packed.bot_0 = conv_one_f32(inp.v8x16[1].h8x8[0].val0);
			left.packed.bot_1 = conv_one_f32(inp.v8x16[1].h8x8[0].val1);

			right.packed.top_0 = conv_one_f32(inp.v8x16[0].h8x8[1].val0);
			right.packed.top_1 = conv_one_f32(inp.v8x16[0].h8x8[1].val1);
			right.packed.bot_0 = conv_one_f32(inp.v8x16[1].h8x8[1].val0);
			right.packed.bot_1 = conv_one_f32(inp.v8x16[1].h8x8[1].val1);

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

		X17_DEVICE void conv(b32::Fragment_16x16<i32> const &inp, Fragment_16x16<i8> &out) {
			union {
				u32 value;
				struct {
					i8 top_0, top_1, bot_0, bot_1;
				} packed;
			} left, right;

			left.packed.top_0 = conv_one(inp.v8x16[0].h8x8[0].val0);
			left.packed.top_1 = conv_one(inp.v8x16[0].h8x8[0].val1);
			left.packed.bot_0 = conv_one(inp.v8x16[1].h8x8[0].val0);
			left.packed.bot_1 = conv_one(inp.v8x16[1].h8x8[0].val1);

			right.packed.top_0 = conv_one(inp.v8x16[0].h8x8[1].val0);
			right.packed.top_1 = conv_one(inp.v8x16[0].h8x8[1].val1);
			right.packed.bot_0 = conv_one(inp.v8x16[1].h8x8[1].val0);
			right.packed.bot_1 = conv_one(inp.v8x16[1].h8x8[1].val1);

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

		X17_DEVICE void conv(b32::Fragment_32x32<f32> const &inp, Fragment_32x32<i8> &out) {
			X17_UNROLL for (usize j = 0; j < 2; ++j) {
				X17_UNROLL for (usize i = 0; i < 2; ++i) {
					conv(inp.v16x32[j].h16x16[i], out.v16x32[j].h16x16[i]);
				}
			}
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
			b32::Fragment_32x32<f32> (&acc)[M_TILES][N_TILES]
		) {
			Fragment_32x32<FixedI8> t[M_TILES][N_TILES];

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					conv(acc[mi][ni], t[mi][ni]);
				}
			}

			MatrixWriter<FixedI8, GN>::write(row, col, t);
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

	template<const usize GN, const usize FAN_IN>
	struct FixedI8MatrixGeGluWriter: MatrixWriter<FixedI8, GN> {

		X17_DEVICE FixedI8MatrixGeGluWriter(FixedI8 *gC):
			MatrixWriter<FixedI8, GN>(gC)
		{}

		X17_DEVICE i8 geglu(b32::Fragment_8x8<i32> frag) {
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
			f32 val_f = fmaxf(-127.0f, fminf(+127.0f, gate * lin));
			return __float2int_rn(val_f);
		}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
		) {
			static_assert(N_TILES % 2 == 0);

			union {
				struct {
					i8 a, b, c, d;
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

			MatrixWriter<FixedI8, GN>::write(row, col / 2, t);
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
		static_assert(M_TILES * M_WARPS * 32 == M_PER_BLOCK);
		static_assert(N_TILES * N_WARPS * 32 == N_PER_BLOCK);

		static constexpr usize SMEM_BYTES = ALoader::SMEM_BYTES + BLoader::SMEM_BYTES;
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

			X17_UNROLL
			for (usize p = 0; p < GMEM_PRELOAD; ++p) {
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

			for (usize k_step = 0; k_step < K_ITERS; ++k_step) {
				{ // Get more data from GMEM
					cp_async_wait<GMEM_PRELOAD - 2>();
					sync_threads();

					usize p = k_step + GMEM_PRELOAD;
					if (p < K_ITERS) {
						A.template cp_async<THREADS_PER_BLOCK>(p, M_PER_BLOCK*block_m, K_STEP*p, N_PER_BLOCK*block_n);
						B.template cp_async<THREADS_PER_BLOCK>(p, K_STEP*p, N_PER_BLOCK*block_n, M_PER_BLOCK*block_m);
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

			C.write(
				block_m * M_PER_BLOCK + warp_m * M_PER_WARP,
				block_n * N_PER_BLOCK + warp_n * N_PER_WARP,
				acc
			);
		}
	};
}
