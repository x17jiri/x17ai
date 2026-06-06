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
				scale[j] = math::fast::rsqrt(sum[j]) * f32(SCALE);
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
						out.set0(f32(inp.get0()) * f32(SCALE));
						out.set1(f32(inp.get1()) * f32(SCALE));
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
						out.set0(math::fast::gelu<INP_SCALE_2, OUT_SCALE_2, 1.0>(f32(inp.get0())).val);
						out.set1(math::fast::gelu<INP_SCALE_2, OUT_SCALE_2, 1.0>(f32(inp.get1())).val);
					}
				}
			}
		}
	}
}

template<
	const usize HEAD_DIM,
	const usize PROJ_OUTPUTS,
	const usize M_PER_BLOCK,
	const usize N_PER_BLOCK
>
struct QMatrixWriter: b8::FixedI8MatrixWriter<
	PROJ_OUTPUTS,
	M_PER_BLOCK,
	N_PER_BLOCK,
	math::constexpr_inv_sqrt(f64(config::MODEL_DIM))
> {
	using Base = b8::FixedI8MatrixWriter<
		PROJ_OUTPUTS,
		M_PER_BLOCK,
		N_PER_BLOCK,
		math::constexpr_inv_sqrt(f64(config::MODEL_DIM))
	>;
	using SNormScales = SVector<bf16, N_PER_BLOCK>;

	static_assert(HEAD_DIM % 32 == 0);
	static_assert(PROJ_OUTPUTS % HEAD_DIM == 0);
	static_assert(N_PER_BLOCK % HEAD_DIM == 0);
	static constexpr usize SMEM_BYTES = N_PER_BLOCK * sizeof(bf16);

	bf16 const *g_norm_scales;
	SNormScales s_norm_scales;
	usize norm_scale_col0;

	X17_DEVICE QMatrixWriter(
		b8::FixedI8 *gC,
		bf16 const *norm_scales
	):
		Base(gC),
		g_norm_scales(norm_scales),
		s_norm_scales(),
		norm_scale_col0(0)
	{}

	template<const u32 CAP>
	X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
		s_norm_scales._ptr = smem_alloc.alloc(SMEM_BYTES);
	}

	template<const usize THREADS_PER_BLOCK>
	X17_DEVICE void async_load(usize row, usize col) {
		norm_scale_col0 = (col / N_PER_BLOCK) * N_PER_BLOCK;
		s_norm_scales.template cp_async_from<THREADS_PER_BLOCK>(threadIdx.x, g_norm_scales + norm_scale_col0);
	}

	template<const usize M_TILES, const usize N_TILES>
	X17_DEVICE void write(
		usize row, usize col,
		b32::Fragment_32x32<i32> (&acc)[M_TILES][N_TILES]
	) {
		b32::Fragment_32x32<f32> t[M_TILES][N_TILES];

		constexpr usize Q_TILES = HEAD_DIM / 32;
		constexpr usize HEADS = N_TILES / Q_TILES;
		static_assert(N_TILES % Q_TILES == 0);

		constexpr f64 QK_SCALE = math::constexpr_sqrt(f64(HEAD_DIM)) * f64(b8::FIXED_I8_SCALE);
		X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
			u32 norm_scales_col = (col - norm_scale_col0) + (hi * Q_TILES * 32);
			qkvg_helpers::l2_norm<QK_SCALE, true, Q_TILES>(
				acc,
				t,
				hi * Q_TILES,
				s_norm_scales._ptr + (norm_scales_col * sizeof(bf16))
			);
		}

		Base::write(row, col, t);
	}
};

template<
	const usize HEAD_DIM,
	const usize PROJ_OUTPUTS,
	const usize M_PER_BLOCK,
	const usize N_PER_BLOCK
>
struct KVMatrixWriter: b8::FixedI8MatrixWriter<
	PROJ_OUTPUTS,
	M_PER_BLOCK,
	N_PER_BLOCK,
	math::constexpr_inv_sqrt(f64(config::MODEL_DIM))
> {
	using Base = b8::FixedI8MatrixWriter<
		PROJ_OUTPUTS,
		M_PER_BLOCK,
		N_PER_BLOCK,
		math::constexpr_inv_sqrt(f64(config::MODEL_DIM))
	>;
	using SNormScales = SVector<bf16, N_PER_BLOCK / 2>;

	static_assert(HEAD_DIM % 32 == 0);
	static_assert(PROJ_OUTPUTS % HEAD_DIM == 0);
	static_assert(N_PER_BLOCK % (2 * HEAD_DIM) == 0);
	static constexpr usize SMEM_BYTES = (N_PER_BLOCK / 2) * sizeof(bf16);

	bf16 const *g_k_norm_scales;
	SNormScales s_k_norm_scales;
	usize k_norm_scale_col0;

	X17_DEVICE KVMatrixWriter(
		b8::FixedI8 *gC,
		bf16 const *k_norm_scales
	):
		Base(gC),
		g_k_norm_scales(k_norm_scales),
		s_k_norm_scales(),
		k_norm_scale_col0(0)
	{}

	template<const u32 CAP>
	X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
		s_k_norm_scales._ptr = smem_alloc.alloc(SMEM_BYTES);
	}

	template<const usize THREADS_PER_BLOCK>
	X17_DEVICE void async_load(usize row, usize col) {
		k_norm_scale_col0 = ((col / N_PER_BLOCK) * N_PER_BLOCK) / 2;
		s_k_norm_scales.template cp_async_from<THREADS_PER_BLOCK>(threadIdx.x, g_k_norm_scales + k_norm_scale_col0);
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

		// k
		constexpr f64 QK_SCALE = math::constexpr_sqrt(f64(HEAD_DIM)) * f64(b8::FIXED_I8_SCALE);
		X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
			usize head_col = col + hi * KV_TILES * 32;
			usize k_norm_scale_col = head_col / 2 - k_norm_scale_col0;
			qkvg_helpers::l2_norm<QK_SCALE, true, K_TILES>(
				acc,
				t,
				hi * KV_TILES,
				s_k_norm_scales._ptr + (k_norm_scale_col * sizeof(bf16))
			);
		}

		// v
		constexpr f64 VG_SCALE = math::constexpr_inv_sqrt(f64(config::MODEL_DIM)) / f64(b8::FIXED_I8_SCALE);
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
