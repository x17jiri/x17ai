#pragma once

#include "utils_b8.cuh"
#include "gemm_b8.cuh"

namespace qkvg_helpers {
	// Calculates L2 Norm of `acc[..][pos .. pos+W_TILES]`, multiplies by `SCALE`
	// and stores the result to `out[..][pos .. pos+W_TILES]`
	template<
		const f64 SCALE, const bool USE_DYN_SCALE,
		const usize W_TILES,
		const usize M_TILES, const usize N_TILES
	>
	X17_DEVICE void l2_norm(
		b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES],
		b32::Fragment_32x32<f32> (&out)[M_TILES][N_TILES],
		usize pos,
		u32 dyn_scales_ptr
	) {
		bf16 dyn_scales0[4 * N_TILES];
		bf16 dyn_scales1[4 * N_TILES];
		if constexpr (USE_DYN_SCALE) {
			usize tid = threadIdx.x % 4;
			dyn_scales_ptr += tid * (2 * sizeof(bf16));
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
						out.val0 = f32(inp.val0);
						out.val1 = f32(inp.val1);
						sum[j] = math::fma(out.val0, out.val0, sum[j]);
						sum[j] = math::fma(out.val1, out.val1, sum[j]);
					}
				}
			}
			f32 scale[4] = {1.0, 1.0, 1.0, 1.0};
			X17_UNROLL for (usize j = 0; j < 4; ++j) {
				sum[j] += shuffle_xor_sync(sum[j], 1);
				sum[j] += shuffle_xor_sync(sum[j], 2);
				scale[j] = math::fast::rsqrt(sum[j]) * f32(SCALE);
			}
			X17_UNROLL for (usize ni = pos; ni < pos + W_TILES; ++ni) {
				auto &out32x32 = out[mi][ni];
				X17_UNROLL for (usize j = 0; j < 4; ++j) {
					X17_UNROLL for (usize i = 0; i < 4; ++i) {
						auto &out = out32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
						if constexpr (USE_DYN_SCALE) {
							out.val0 *= scale[j] * f32(dyn_scales0[4*ni + i]);
							out.val1 *= scale[j] * f32(dyn_scales1[4*ni + i]);
						} else {
							out.val0 *= scale[j];
							out.val1 *= scale[j];
						}
					}
				}
			}
		}
	}

	// Casts raw accumulators `acc[..][pos .. pos+W_TILES]`, multiplies by `SCALE`
	// and stores the result to `out[..][pos .. pos+W_TILES]`
	template<
		const f64 SCALE,
		const usize W_TILES,
		const usize M_TILES, const usize N_TILES
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
						out.val0 = f32(inp.val0) * f32(SCALE);
						out.val1 = f32(inp.val1) * f32(SCALE);
					}
				}
			}
		}
	}

	// Applies GELU to raw accumulators `acc[..][pos .. pos+W_TILES]` using the
	// provided squared input and output scales, then stores the result to
	// `out[..][pos .. pos+W_TILES]`.
	template<
		const f64 INP_SCALE_2,
		const f64 OUT_SCALE_2,
		const usize W_TILES,
		const usize M_TILES, const usize N_TILES
	>
	X17_DEVICE void gelu_output(
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
						out.val0 = math::fast::gelu<INP_SCALE_2, OUT_SCALE_2, 1.0>(f32(inp.val0)).val;
						out.val1 = math::fast::gelu<INP_SCALE_2, OUT_SCALE_2, 1.0>(f32(inp.val1)).val;
					}
				}
			}
		}
	}
}

template<const usize PROJ_OUTPUTS, const usize SPARSE_FAN_IN>
using QKVGBaseMatrixWriter =
	b8::FixedI8MatrixWriter<
		PROJ_OUTPUTS,
		math::constexpr_inv_sqrt(SPARSE_FAN_IN)
	>;

template<
	const usize HEAD_DIM,
	const usize PROJ_OUTPUTS,
	const usize SPARSE_FAN_IN
>
struct QGMatrixWriter: QKVGBaseMatrixWriter<PROJ_OUTPUTS, SPARSE_FAN_IN> {
	using Base = QKVGBaseMatrixWriter<PROJ_OUTPUTS, SPARSE_FAN_IN>;

	static_assert(HEAD_DIM % 32 == 0);
	static_assert(PROJ_OUTPUTS % HEAD_DIM == 0);

	bf16 const *g_norm_scales;
	u32 s_norm_scales;
	usize norm_scale_col0;

	X17_DEVICE QGMatrixWriter(
		b8::FixedI8 *gC,
		bf16 const *norm_scales
	):
		Base(gC),
		g_norm_scales(norm_scales),
		s_norm_scales(0),
		norm_scale_col0(0)
	{}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK>
	X17_HOST_DEVICE static constexpr usize smem_cols() {
		return N_PER_BLOCK / 2;
	}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK>
	X17_HOST_DEVICE static constexpr usize smem_bytes() {
		return smem_cols<M_PER_BLOCK, N_PER_BLOCK>() * sizeof(bf16);
	}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK, const u32 CAP>
	X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
		s_norm_scales = smem_alloc.alloc(smem_bytes<M_PER_BLOCK, N_PER_BLOCK>());
	}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK, const usize THREADS_PER_BLOCK>
	X17_DEVICE void cp_async(usize row, usize col) {
		norm_scale_col0 = (col / N_PER_BLOCK) * N_PER_BLOCK / 2;
		constexpr usize COLS = smem_cols<M_PER_BLOCK, N_PER_BLOCK>();
		SVector<bf16, COLS> smem(s_norm_scales);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, g_norm_scales + norm_scale_col0, smem);
	}

	template<const usize M_TILES, const usize N_TILES>
	X17_DEVICE void write(
		usize row, usize col,
		b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
	) {
		b32::Fragment_32x32<f32> t[M_TILES][N_TILES];

		constexpr usize K_TILES = HEAD_DIM / 32;
		constexpr usize V_TILES = HEAD_DIM / 32;
		constexpr usize KV_TILES = K_TILES + V_TILES;
		constexpr usize HEADS = N_TILES / KV_TILES;
		static_assert(N_TILES % KV_TILES == 0);

		// q
		constexpr f64 QK_SCALE = math::constexpr_sqrt(f64(HEAD_DIM)) * f64(b8::FIXED_I8_SCALE);
		X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
			u32 norm_scales_col = ((col / 2) - norm_scale_col0) + (hi * K_TILES * 32);
			qkvg_helpers::l2_norm<QK_SCALE, true, K_TILES>(
				acc,
				t,
				hi * KV_TILES,
				s_norm_scales + (norm_scales_col * sizeof(bf16))
			);
		}

		// g
		constexpr f64 FIXED_I8_SCALE_2 = f64(b8::FIXED_I8_SCALE) * f64(b8::FIXED_I8_SCALE);
		constexpr f64 G_INP_SCALE_2 = 1.0 / (f64(SPARSE_FAN_IN) * FIXED_I8_SCALE_2 * FIXED_I8_SCALE_2);
		constexpr f64 G_OUT_SCALE_2 = FIXED_I8_SCALE_2;
		X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
			qkvg_helpers::gelu_output<G_INP_SCALE_2, G_OUT_SCALE_2, V_TILES>(
				acc,
				t,
				hi * KV_TILES + K_TILES
			);
		}

		Base::write(row, col, t);
	}
};

template<
	const usize HEAD_DIM,
	const usize PROJ_OUTPUTS,
	const usize SPARSE_FAN_IN
>
struct KVMatrixWriter: QKVGBaseMatrixWriter<PROJ_OUTPUTS, SPARSE_FAN_IN> {
	using Base = QKVGBaseMatrixWriter<PROJ_OUTPUTS, SPARSE_FAN_IN>;

	static_assert(HEAD_DIM % 32 == 0);
	static_assert(PROJ_OUTPUTS % HEAD_DIM == 0);

	X17_DEVICE KVMatrixWriter(b8::FixedI8 *gC):
		Base(gC)
	{}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK>
	X17_HOST_DEVICE static constexpr usize smem_cols() {
		return 0;
	}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK>
	X17_HOST_DEVICE static constexpr usize smem_bytes() {
		return 0;
	}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK, const u32 CAP>
	X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK, const usize THREADS_PER_BLOCK>
	X17_DEVICE void cp_async(usize row, usize col) {}

	template<const usize M_TILES, const usize N_TILES>
	X17_DEVICE void write(
		usize row, usize col,
		b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
	) {
		b32::Fragment_32x32<f32> t[M_TILES][N_TILES];

		constexpr usize K_TILES = HEAD_DIM / 32;
		constexpr usize V_TILES = HEAD_DIM / 32;
		constexpr usize KV_TILES = K_TILES + V_TILES;
		constexpr usize HEADS = N_TILES / KV_TILES;
		static_assert(N_TILES % KV_TILES == 0);

		// k
		constexpr f64 QK_SCALE = math::constexpr_sqrt(f64(HEAD_DIM)) * f64(b8::FIXED_I8_SCALE);
		X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
			qkvg_helpers::l2_norm<QK_SCALE, false, K_TILES>(
				acc,
				t,
				hi * KV_TILES,
				0
			);
		}

		// v
		constexpr f64 VG_SCALE = math::constexpr_inv_sqrt(SPARSE_FAN_IN) / f64(b8::FIXED_I8_SCALE);
		X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
			qkvg_helpers::raw_output<VG_SCALE, V_TILES>(
				acc,
				t,
				hi * KV_TILES + K_TILES
			);
		}

		Base::write(row, col, t);
	}
};
