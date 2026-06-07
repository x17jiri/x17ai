#pragma once

#include "utils.cuh"

#pragma nv_diag_suppress 186

namespace b16 {
	template<
		typename T,
		const usize _GN, // number of columns of the input matrix in GMEM
		const usize _M, const usize _K, // size of preload tile
		const usize _GMEM_PRELOAD = 2 // number of preload tiles
	>
	requires(sizeof(T) == 2)
	struct MatrixLoader {
		using Elem = T;

		static constexpr usize GN = _GN;
		static constexpr usize M = _M;
		static constexpr usize K = _K;
		static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

		static_assert(GN % K == 0);
		static_assert(M % 16 == 0);
		static_assert(K % 16 == 0);

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
			b16::async_load<THREADS_PER_BLOCK, M, K>(
				threadIdx.x,
				gInput, m, k,
				sPreload, M * (step % GMEM_PRELOAD), 0
			);
		}

		/// Load a fragment at tile coordinates [m, k] from the SMEM ring buffer.
		X17_DEVICE void load_fragment(usize step, usize m, usize k, Fragment_16x16<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			b16::load_fragment(sPreload, first_row + 16*m, 16*k, frag);
		}

		X17_DEVICE void load_fragment_trans(usize step, usize m, usize k, Fragment_16x16<T> &frag) {
			usize first_row = M * (step % GMEM_PRELOAD);
			b16::load_fragment_trans(sPreload, first_row + 16*m, 16*k, frag);
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

		X17_DEVICE void load_fragment(usize step, usize m, usize k, Fragment_16x16<Elem> &frag) {
			loader.load_fragment_trans(step, k, m, frag);
		}

		X17_DEVICE void load_fragment_trans(usize step, usize m, usize k, Fragment_16x16<Elem> &frag) {
			loader.load_fragment(step, k, m, frag);
		}
	};

	template<
		typename T,
		const usize _GN,
		const usize _M_PER_BLOCK = 0,
		const usize _N_PER_BLOCK = 0
	>
	requires(sizeof(T) == 2)
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
		X17_DEVICE void async_load(usize row, usize col) {}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			Fragment_16x16<T> (&acc)[M_TILES][N_TILES]
		) {
			GMatrix<bf16, 16*M_TILES, 16*N_TILES> C(gC, c_stride);
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				store(acc[mi], C, row + 16*mi, col);
			}
		}
	};

	template<const usize GN, const usize M_PER_BLOCK, const usize N_PER_BLOCK>
	struct Bf16MatrixWriter: MatrixWriter<bf16, GN, M_PER_BLOCK, N_PER_BLOCK> {
		using Base = MatrixWriter<bf16, GN, M_PER_BLOCK, N_PER_BLOCK>;

		X17_DEVICE Bf16MatrixWriter(bf16 *gC):
			Base(gC)
		{}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			b32::Fragment_16x16<f32> (&acc)[M_TILES][N_TILES]
		) {
			Fragment_16x16<bf16> t[M_TILES][N_TILES];

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					cast(acc[mi][ni], t[mi][ni]);
				}
			}

			Base::write(row, col, t);
		}
	};
/*
	template<const usize GN, const usize D_IN, const usize FAN_IN = D_IN>
	struct MatrixGeGluWriter {
		bf16 *gC;
		bf16 *gGrad;
		usize c_stride;
		usize g_stride;

		static constexpr f64 SPARSE_SCALE_2 = f64(D_IN) / f64(FAN_IN);
		static constexpr f64 OUT_SCALE_2 = 1.0 / f64(GN);

		X17_DEVICE MatrixGeGluWriter(bf16 *gC, bf16 *gGrad):
			gC(gC),
			gGrad(gGrad),
			c_stride(GN),
			g_stride(2*GN)
		{}

		template<const usize M_TILES, const usize N_TILES>
		X17_DEVICE void write(
			usize row, usize col,
			Fragment_16x16<f32> (&acc)[M_TILES][N_TILES]
		) {
			GMatrix<bf16, 16*M_TILES, 8*N_TILES> C(gC, c_stride);
			static_assert(N_TILES % 2 == 0);
			Fragment_16x16<bf16> out[M_TILES][N_TILES / 2];

			if (gGrad != nullptr) {
				GMatrix<bf16, 16 * M_TILES, 32> G(gGrad, g_stride);
				X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
					X17_UNROLL for (usize ni = 0; ni < N_TILES/2; ++ni) {
						geglu_and_backvec_<SPARSE_SCALE_2, OUT_SCALE_2>(
							acc[mi][2*ni+0],
							acc[mi][2*ni+1],
							out[mi][ni]
						);
					}
					store(acc[mi], G, row + 16*mi, col);
					store(out[mi], C, row + 16*mi, col/2);
				}
			} else {
				X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
					X17_UNROLL for (usize ni = 0; ni < N_TILES/2; ++ni) {
						geglu_and_backvec_<SPARSE_SCALE_2, OUT_SCALE_2>(
							acc[mi][2*ni+0],
							acc[mi][2*ni+1],
							out[mi][ni]
						);
					}
					store(out[mi], C, row + 16*mi, col/2);
				}
			}
		}
	};

	template<
		const usize _GN,
		const usize _M,
		const usize _N,
		const usize _GMEM_PRELOAD = 2
	>
	struct GeGluBackwardLoader {
		static constexpr usize GN = _GN;
		static constexpr usize M = _M;
		static constexpr usize N = _N;
		static constexpr usize DF_GN = GN / 2;
		static constexpr usize DF_TILE_N = N / 2;
		static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

		static_assert(GN % N == 0);
		static_assert(DF_GN % DF_TILE_N == 0);
		static_assert(M % 16 == 0);
		static_assert(N % 32 == 0);

		static constexpr usize BACKVEC_SMEM_BYTES = N * M * GMEM_PRELOAD * sizeof(bf16);
		static constexpr usize DF_SMEM_BYTES = M * DF_TILE_N * GMEM_PRELOAD * sizeof(bf16);
		static constexpr usize SMEM_BYTES = BACKVEC_SMEM_BYTES + DF_SMEM_BYTES;

		using GBackvec = GMatrixDynSize<bf16, GN>;
		using GDf = GMatrixDynSize<bf16, DF_GN>;
		using SBackvec = SMatrix<bf16, M * GMEM_PRELOAD, N>;
		using SDf = SMatrix<bf16, M, DF_TILE_N * GMEM_PRELOAD>;

		GBackvec gBackvec;
		GDf gDf;
		SBackvec sBackvec;
		SDf sDf;

		X17_DEVICE usize m_rows() const { return gBackvec.m_rows(); }
		X17_DEVICE usize n_cols() const { return gBackvec.n_cols(); }

		X17_DEVICE GeGluBackwardLoader(bf16 *d_f, bf16 *backvec, usize m_rows):
			gBackvec(backvec, m_rows),
			gDf(d_f, m_rows),
			sBackvec(),
			sDf()
		{}

		template<const u32 CAP>
		X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
			sBackvec._ptr = smem_alloc.alloc(BACKVEC_SMEM_BYTES);
			sDf._ptr = smem_alloc.alloc(DF_SMEM_BYTES);
		}

		template<const usize THREADS_PER_BLOCK>
		X17_DEVICE void cp_async(usize step, usize m, usize n) {
			cp_async_gmem_to_smem<THREADS_PER_BLOCK, M, N>(
				threadIdx.x,
				gBackvec.template tile_m<M>(m).template slice_n<N>(n * N),
				sBackvec.template tile_m<M>(step % GMEM_PRELOAD),
				0, 0, 0, 0
			);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK, M, DF_TILE_N>(
				threadIdx.x,
				gDf.template tile_m<M>(m).template slice_n<DF_TILE_N>(n * DF_TILE_N),
				sDf,
				0, 0, 0, (step % GMEM_PRELOAD) * DF_TILE_N
			);
		}

		X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
			auto backvec_tile = sBackvec.template tile_m<M>(step % GMEM_PRELOAD);
			smem_tile_to_fragment(backvec_tile, m * 16, n * 16, frag);

			Fragment_16x8<bf16> d_f_tile;
			usize slot_col = (step % GMEM_PRELOAD) * DF_TILE_N;
			usize d_f_tile_col = slot_col + n * 8;
			smem_tile_to_fragment(sDf, m * 16, d_f_tile_col, d_f_tile);

			Fragment_8x8<bf16> d_f_top = d_f_tile.sub[0];
			Fragment_8x8<bf16> d_f_bot = d_f_tile.sub[1];
			geglu_backward_(d_f_top, frag.sub[0][0], frag.sub[0][1]);
			geglu_backward_(d_f_bot, frag.sub[1][0], frag.sub[1][1]);
		}

		X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
			b16::load_fragment(step, m, n, frag);
			frag.transpose_();
		}
	};
*/
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

		static constexpr usize M_TILES = M_PER_WARP / 16;
		static constexpr usize N_TILES = N_PER_WARP / 16;
		static constexpr usize K_TILES = K_STEP / 16;
		static_assert(Writer::M_PER_BLOCK == M_PER_BLOCK);
		static_assert(Writer::N_PER_BLOCK == N_PER_BLOCK);
		static_assert(M_TILES * M_WARPS * 16 == M_PER_BLOCK);
		static_assert(N_TILES * N_WARPS * 16 == N_PER_BLOCK);

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

			Fragment_16x16<bf16> rA[M_TILES][K_TILES];
			Fragment_16x16<bf16> rBT[K_TILES][N_TILES];
			b32::Fragment_16x16<f32> acc[M_TILES][N_TILES];
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
}
