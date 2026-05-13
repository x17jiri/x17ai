#pragma once

#include "utils.cuh"

#pragma nv_diag_suppress 186

template<const usize CAP>
struct SMemAllocator {
	u32 _ptr;

	X17_DEVICE SMemAllocator(): _ptr(0) {}

	X17_DEVICE u32 alloc(usize size) {
		u32 result = _ptr;
		_ptr += size;
		return result;
	}

	X17_DEVICE void finish() {
		// TODO: assert _ptr == CAP
	}
};

template<
	const usize _GN, // number of columns of the input matrix in GMEM
	const usize _M, const usize _N, // size of preload tile
	const usize _GMEM_PRELOAD = 2 // number of preload tiles
>
struct MatrixLoader {
	static constexpr usize M = _M;
	static constexpr usize N = _N;
	static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

	static_assert(_GN % N == 0);
	static_assert(M % 16 == 0);
	static_assert(N % 16 == 0);

	static constexpr usize SMEM_BYTES = M * N * GMEM_PRELOAD * sizeof(bf16);

	using GInput = GMatrixDynSize<bf16, _GN>;
	using SPreload = SMatrix<bf16, M * GMEM_PRELOAD, N>;

	GInput gInput;
	SPreload sPreload;

	X17_DEVICE usize m_rows() const { return gInput.m_rows(); }
	X17_DEVICE usize n_cols() const { return gInput.n_cols(); }

	X17_DEVICE MatrixLoader(bf16 *gmem_addr, usize m_rows):
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
		cp_async_gmem_to_smem<THREADS_PER_BLOCK, M, N>(
			threadIdx.x,
			gInput.template tile_m<M>(m).slice_n<N>(N*n),
			sPreload.template tile_m<M>(step % GMEM_PRELOAD),
			0, 0, 0, 0
		);
	}

	/// Load fragment at tile coordinates [m, n] from the SMEM ring buffer.
	X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
		auto s = sPreload.tile_m<M>(step % GMEM_PRELOAD);
		smem_tile_to_fragment(s, m*16, n*16, frag);
	}

	X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
		auto s = sPreload.tile_m<M>(step % GMEM_PRELOAD);
		smem_tile_to_fragment_trans(s, m*16, n*16, frag);
	}
};

template<
	const usize _GN, // number of columns in the decoded matrix
	const usize _M, const usize _N, // size of decoded preload tile
	const usize _GMEM_PRELOAD = 2 // number of preload tiles
>
struct MatrixF8Loader {
	static constexpr usize GN = _GN;
	static constexpr usize M = _M;
	static constexpr usize N = _N;
	static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;
	static constexpr usize PACKED_GN = GN / 2;
	static constexpr usize PACKED_N = N / 2;

	static_assert(_GN % N == 0);
	static_assert(M % 16 == 0);
	static_assert(N % 16 == 0);
	static_assert(GN % 2 == 0);
	static_assert(N % 2 == 0);

	static constexpr usize SMEM_BYTES = M * PACKED_N * GMEM_PRELOAD * sizeof(bf16);

	using GInput = GMatrixDynSize<bf16, PACKED_GN>;
	using SPreload = SMatrix<bf16, M * GMEM_PRELOAD, PACKED_N>;

	GInput gInput;
	SPreload sPreload;

	X17_DEVICE usize m_rows() const { return gInput.m_rows(); }
	X17_DEVICE usize n_cols() const { return GN; }

	X17_DEVICE MatrixF8Loader(bf16 *gmem_addr, usize m_rows):
		gInput(gmem_addr, m_rows),
		sPreload()
	{}

	template<const u32 CAP>
	X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
		sPreload._ptr = smem_alloc.alloc(SMEM_BYTES);
	}

	/// `cp_async` a packed tile with decoded logical size [M, N] at position [M*m, N*n].
	template<const usize THREADS_PER_BLOCK>
	X17_DEVICE void cp_async(usize step, usize m, usize n) {
		cp_async_gmem_to_smem<THREADS_PER_BLOCK, M, PACKED_N>(
			threadIdx.x,
			gInput.template tile_m<M>(m).slice_n<PACKED_N>(PACKED_N * n),
			sPreload.template tile_m<M>(step % GMEM_PRELOAD),
			0, 0, 0, 0
		);
	}

	/// Load and decode fragment at tile coordinates [m, n] from the SMEM ring buffer.
	X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
		auto s = sPreload.tile_m<M>(step % GMEM_PRELOAD);
		Fragment_16x8<bf16> packed;
		smem_tile_to_fragment(s, m*16, n*8, packed);
		unpack_f8_fragment(packed.sub[0], frag.sub[0][0], frag.sub[0][1]);
		unpack_f8_fragment(packed.sub[1], frag.sub[1][0], frag.sub[1][1]);
	}

	X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
		load_fragment(step, m, n, frag);
		frag.transpose_();
	}
};

template<
	const usize _GN, // number of columns of the input matrix in GMEM
	const usize FAN_IN,
	const usize CYCLE,
	const usize _M, const usize _N, // size of preload tile
	const usize _GMEM_PRELOAD = 2 // number of preload tiles
>
struct SparseMatrixLoader {
	static constexpr usize GN = _GN;
	static constexpr usize M = _M;
	static constexpr usize N = _N;
	static constexpr usize GMEM_PRELOAD = _GMEM_PRELOAD;

	static_assert(_GN % N == 0);
	static_assert(M % 16 == 0);
	static_assert(N % 16 == 0);

	static constexpr usize INPUT_STEP = _GN / CYCLE;
	static constexpr usize GROUP_TILE_CNT = CYCLE / 16;
	static constexpr usize GROUP_CNT = M / CYCLE;
	static_assert(FAN_IN < _GN);
	static_assert(INPUT_STEP % 16 == 0);
	static_assert(_GN % CYCLE == 0);
	static_assert(M % CYCLE == 0);
	static_assert(CYCLE <= N);
	static_assert(CYCLE % 16 == 0);

	static constexpr usize SMEM_BYTES = M * N * GMEM_PRELOAD * sizeof(bf16);

	using GInput = GMatrixDynSize<bf16, FAN_IN>;
	using SPreload = SMatrix<bf16, M * GMEM_PRELOAD, N>;

	GInput gInput;
	SPreload sPreload;

	X17_DEVICE usize m_rows() const { return gInput.m_rows(); }
	X17_DEVICE usize n_cols() const { return GN; }

	X17_DEVICE SparseMatrixLoader(bf16 *gmem_addr, usize m_rows):
		gInput(gmem_addr, m_rows),
		sPreload()
	{}

	template<const u32 CAP>
	X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
		sPreload._ptr = smem_alloc.alloc(SMEM_BYTES);
	}

	template<const usize THREADS_PER_BLOCK>
	X17_DEVICE void cp_async(usize p, usize m, usize n) {
		static_assert(N % CYCLE == 0, "current implementation assumes we don't start in the middle of a cycle");

		using T = bf16;
		GMatrix<T, M, FAN_IN> src = gInput.template tile_m<M>(m);
		SMatrix<T, M, N> dst = sPreload.template tile_m<M>(p % GMEM_PRELOAD);

		usize SRC_ROW_BYTES = src.stride_bytes();
		constexpr usize DST_ROW_BYTES = N*sizeof(T);
		constexpr usize CP_BYTES = 16;
		constexpr usize CP_PER_ROW = DST_ROW_BYTES / CP_BYTES;
		constexpr usize ROWS_PER_STEP = THREADS_PER_BLOCK / CP_PER_ROW;
		constexpr usize STEPS = M / ROWS_PER_STEP;

		static_assert(THREADS_PER_BLOCK % CP_PER_ROW == 0);

		// Thread's position within a step is fixed
		usize tid = threadIdx.x;
		usize row_in_step = tid / CP_PER_ROW;
		usize off_in_row = (tid % CP_PER_ROW) * CP_BYTES;

		constexpr usize REPEAT_AFTER = least_common_multiple(8, ROWS_PER_STEP) / ROWS_PER_STEP;
		usize swizzle[REPEAT_AFTER];
		X17_UNROLL for (usize i = 0; i < REPEAT_AFTER; ++i) {
			usize row = i * ROWS_PER_STEP + row_in_step;
			swizzle[i] = off_in_row ^ ((row & 7) << 4);
		}

		// off = column offset in the expanded matrix; off < GN*sizeof(T)
		// We stay at this offset the whole time, but the data_off, i.e., the offset where
		// data actually starts is different for each row.
		usize off = n * (N*sizeof(T)) + off_in_row;
		usize data_off = usize(row_in_step * (INPUT_STEP*sizeof(T))) % usize(GN*sizeof(T));
		constexpr usize DATA_SIZE = FAN_IN*sizeof(T);

		u8 const *src_row_ptr = reinterpret_cast<u8 *>(src._ptr) + row_in_step * SRC_ROW_BYTES;
		u32 dst_row_ptr = dst._ptr + row_in_step * DST_ROW_BYTES;

		// for step in 0 ..< STEPS:
		//     if off in (data_off ..< data_off + DATA_SIZE):
		//         read the non-expanded matrix at: off - data_off
		//     else if off + GN*sizeof(T) in (data_off ..< data_off + DATA_SIZE):
		//         read the non-expanded matrix at: off + GN*sizeof(T) - data_off
		//     else:
		//         zero
		//     data_off = (data_off + ROWS_PER_STEP * INPUT_STEP*sizeof(T)) % (GN*sizeof(T))
		if constexpr (STEPS > 0) {
			X17_UNROLL for (usize step = 0; step < STEPS; ++step) {
				u32 dst_ptr = dst_row_ptr + swizzle[step % REPEAT_AFTER];
				usize t1 = off - data_off;
				usize t2 = t1 + GN*sizeof(T);
				if (t1 < DATA_SIZE) {
					sm80::cp_async(src_row_ptr + t1, dst_ptr);
				} else if (t2 < DATA_SIZE) {
					sm80::cp_async(src_row_ptr + t2, dst_ptr);
				} else {
					store_shared_4x32b(dst_ptr, 0.0f, 0.0f, 0.0f, 0.0f);
				}
				src_row_ptr += ROWS_PER_STEP * SRC_ROW_BYTES;
				dst_row_ptr += ROWS_PER_STEP * DST_ROW_BYTES;
				data_off = (data_off + usize(ROWS_PER_STEP * INPUT_STEP*sizeof(T))) % usize(GN*sizeof(T));
			}
		}
		if constexpr (M % ROWS_PER_STEP != 0) {
			usize step = STEPS;
			if (tid < (M % ROWS_PER_STEP) * CP_PER_ROW) {
				u32 dst_ptr = dst_row_ptr + swizzle[step % REPEAT_AFTER];
				usize t1 = off - data_off;
				usize t2 = t1 + GN*sizeof(T);
				if (t1 < DATA_SIZE) {
					sm80::cp_async(src_row_ptr + t1, dst_ptr);
				} else if (t2 < DATA_SIZE) {
					sm80::cp_async(src_row_ptr + t2, dst_ptr);
				} else {
					store_shared_4x32b(dst_ptr, 0.0f, 0.0f, 0.0f, 0.0f);
				}
			}
		}
	}

	/// Load fragment at tile coordinates [m, n] from the SMEM ring buffer.
	X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
		auto s = sPreload.tile_m<M>(step % GMEM_PRELOAD);
		smem_tile_to_fragment(s, m*16, n*16, frag);
	}

	X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
		auto s = sPreload.tile_m<M>(step % GMEM_PRELOAD);
		smem_tile_to_fragment_trans(s, m*16, n*16, frag);
	}
};

template<typename MatrixLoader>
struct MatrixTransLoader {
	static constexpr usize M = MatrixLoader::N;
	static constexpr usize N = MatrixLoader::M;
	static constexpr usize GMEM_PRELOAD = MatrixLoader::GMEM_PRELOAD;
	static constexpr usize SMEM_BYTES = MatrixLoader::SMEM_BYTES;

	MatrixLoader loader;

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

	X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
		loader.load_fragment_trans(step, n, m, frag);
	}

	X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_16x16<bf16> &frag) {
		loader.load_fragment(step, n, m, frag);
	}
};

template<const usize GN>
struct MatrixWriter {
	bf16 *gC;
	usize c_stride;

	X17_DEVICE MatrixWriter(bf16 *gC):
		gC(gC),
		c_stride(GN)
	{}

	template<const usize M_TILES, const usize N_TILES>
	X17_DEVICE void write(
		usize row, usize col,
		Fragment_16x16<f32> (&acc)[M_TILES][N_TILES]
	) {
		GMatrix<bf16, 16*M_TILES, 16*N_TILES> C(gC, c_stride);
		X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
			store(acc[mi], C, row + 16*mi, col);
		}
	}
};

/// This is used for implementing backward pass for sparse weight matrices.
/// It writes only the weights that belong to the sparse matrix
template<const usize GN, const usize _FAN_IN, const usize CYCLE>
struct SparseMatrixWriter {
	bf16 *gC;

	static constexpr usize INPUT_STEP = GN / CYCLE;

	static_assert(GN % 16 == 0);
	static_assert(_FAN_IN % 16 == 0);
	static_assert(INPUT_STEP % 16 == 0);

	X17_DEVICE SparseMatrixWriter(bf16 *gC):
		gC(gC)
	{}

	X17_DEVICE void write_8x8(
		usize proj_row_base,
		usize dense_col_base,
		Fragment_8x8<f32> const &frag
	) {
		usize lane = threadIdx.x % WARP_SIZE;
		usize local_row = lane / 4;
		usize local_col_pair = lane % 4;

		usize proj_row = proj_row_base + local_row;
		usize dense_col = dense_col_base + 2 * local_col_pair;
		usize dense_col_start = (proj_row * INPUT_STEP) % GN;
		usize compact_col = (dense_col + GN - dense_col_start) % GN;
		if (compact_col >= _FAN_IN) {
			return;
		}

		usize out_idx = proj_row * _FAN_IN + compact_col;
		FragmentReg<bf16> packed;
		packed.set(
			round_cast<bf16>(frag.first()),
			round_cast<bf16>(frag.second())
		);
		reinterpret_cast<u32 *>(gC)[out_idx / 2] = packed.val;
	}

	template<const usize ROW_TILES, const usize COL_TILES>
	X17_DEVICE void write(
		usize row, usize col,
		Fragment_16x16<f32> (&acc)[ROW_TILES][COL_TILES]
	) {
		X17_UNROLL for (usize row_tile = 0; row_tile < ROW_TILES; ++row_tile) {
			usize proj_row = row + 16 * row_tile;
			X17_UNROLL for (usize col_tile = 0; col_tile < COL_TILES; ++col_tile) {
				usize dense_col = col + 16 * col_tile;
				auto const &tile = acc[row_tile][col_tile];

				write_8x8(proj_row + 0, dense_col + 0, tile.sub[0][0]);
				write_8x8(proj_row + 0, dense_col + 8, tile.sub[0][1]);
				write_8x8(proj_row + 8, dense_col + 0, tile.sub[1][0]);
				write_8x8(proj_row + 8, dense_col + 8, tile.sub[1][1]);
			}
		}
	}
};

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
		load_fragment(step, m, n, frag);
		frag.transpose_();
	}
};

template<typename _ALoader, typename _BLoader, typename _Writer>
struct Gemm {
	using ALoader = _ALoader;
	using BLoader = _BLoader;
	using Writer = _Writer;

	static constexpr usize M_PER_BLOCK = ALoader::M;
	static constexpr usize N_PER_BLOCK = BLoader::N;
	static constexpr usize K_STEP = ALoader::N;
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
	static_assert(M_TILES * M_WARPS * 16 == M_PER_BLOCK);
	static_assert(N_TILES * N_WARPS * 16 == N_PER_BLOCK);

	static constexpr usize SMEM_BYTES = ALoader::SMEM_BYTES + BLoader::SMEM_BYTES;
	static constexpr usize GMEM_PRELOAD = ALoader::GMEM_PRELOAD;
	static_assert(ALoader::GMEM_PRELOAD == BLoader::GMEM_PRELOAD);

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

		X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
			if (p < K_ITERS) {
				A.template cp_async<THREADS_PER_BLOCK>(p, block_m, p);
				B.template cp_async<THREADS_PER_BLOCK>(p, p, block_n);
			}
			cp_async_commit();
		}

		Fragment_16x16<bf16> rA[M_TILES][K_TILES];
		Fragment_16x16<bf16> rBT[K_TILES][N_TILES];
		Fragment_16x16<f32> acc[N_TILES][M_TILES];
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

		X17_UNROLL for (usize k_step = 0; k_step < K_ITERS; ++k_step) {
			{ // Get more data from GMEM
				cp_async_wait<GMEM_PRELOAD - 2>();
				sync_threads();

				usize p = k_step + GMEM_PRELOAD;
				if (p < K_ITERS) {
					A.template cp_async<THREADS_PER_BLOCK>(p, block_m, p);
					B.template cp_async<THREADS_PER_BLOCK>(p, p, block_n);
				}
				cp_async_commit();
			}

			X17_UNROLL for (usize ki = 0; ki < K_TILES; ++ki) {
				X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
					X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
						mma_a_bt(rBT[ki][ni], rA[mi][ki], acc[ni][mi]);
					}
					A.load_fragment(k_step + 1, warp_m * M_TILES + mi, ki, rA[mi][ki]);
				}
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					B.load_fragment_trans(k_step + 1, ki, warp_n * N_TILES + ni, rBT[ki][ni]);
				}
			}
		}

		C.write(
			block_n * N_PER_BLOCK + warp_n * N_PER_WARP,
			block_m * M_PER_BLOCK + warp_m * M_PER_WARP,
			acc
		);
	}
};
