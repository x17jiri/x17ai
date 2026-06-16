#pragma once

#include "utils_b8.cuh"
namespace qkvg_helpers {
	// Calculates L2 Norm of `acc[..][pos .. pos+W_TILES]`, multiplies by `SCALE`
	// and stores the result to `out[..][pos .. pos+W_TILES]`
	template<
		const f64 SCALE, const bool USE_DYN_SCALE,
		const bool STORE_RRMS,
		const usize RRMS_COLS,
		const usize W_TILES,
		const usize M_TILES, const usize N_TILES
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
				if constexpr (STORE_RRMS) {
					constexpr f64 FIXED_I8_SCALE_2 = f64(b8::FIXED_I8_SCALE) * f64(b8::FIXED_I8_SCALE);
					constexpr f64 RAW_SUM_TO_REAL_MEAN =
						1.0 / (f64(W_TILES * 32) * f64(config::MODEL_DIM) * FIXED_I8_SCALE_2 * FIXED_I8_SCALE_2);
					usize lane = threadIdx.x % WARP_SIZE;
					if ((lane % 4) == 0) {
						rrms[4 * mi + j][rrms_col] =
							math::fast::rsqrt(sum[j] * f32(RAW_SUM_TO_REAL_MEAN) + f32(config::L2_NORM_EPS));
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
}


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
	static_assert(HEAD_DIM % 32 == 0);
	static_assert(PROJ_OUTPUTS % HEAD_DIM == 0);
	static_assert(N_PER_BLOCK % (2 * HEAD_DIM) == 0);

	f32 *g_rrms_ptr;

	X17_DEVICE KVMatrixWriter(
		b8::FixedI8 *gC,
		f32 *g_rrms_ptr
	):
		Base(gC),
		g_rrms_ptr(g_rrms_ptr)
	{}

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
		f32 rrms[4 * M_TILES][HEADS];

		// k
		constexpr f64 QK_SCALE = math::constexpr_sqrt(f64(HEAD_DIM)) * f64(b8::FIXED_I8_SCALE);
		X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
			qkvg_helpers::l2_norm<QK_SCALE, false, true, HEADS, K_TILES>(
				acc,
				t,
				hi * KV_TILES,
				0,
				rrms,
				hi
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

		usize lane = threadIdx.x % WARP_SIZE;
		if ((lane % 4) == 0) {
			usize head0 = col / (2 * HEAD_DIM);
			X17_UNROLL for (usize ri = 0; ri < 4 * M_TILES; ++ri) {
				usize rrms_row = row + 8 * ri + lane / 4;
				store_gmem_Nx32b(g_rrms_ptr + head0 + (rrms_row * config::N_HEADS), rrms[ri]);
			}
		}

		Base::write(row, col, t);
	}
};

