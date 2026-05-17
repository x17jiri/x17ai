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
			gInput.template tile_m<M>(m).slice_n<N>(N*n),
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

		X17_DEVICE void load_fragment(usize step, usize m, usize n, Fragment_32x32<T> &frag) {
			loader.load_fragment_trans(step, n, m, frag);
		}

		X17_DEVICE void load_fragment_trans(usize step, usize m, usize n, Fragment_32x32<T> &frag) {
			loader.load_fragment(step, n, m, frag);
		}
	};
}
