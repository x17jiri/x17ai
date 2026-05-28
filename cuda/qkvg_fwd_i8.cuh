#pragma once

#include "utils_b8.cuh"
#include "gemm_b8.cuh"

template<const usize PROJ_OUTPUTS, const usize SPARSE_FAN_IN>
using QKVGBaseMatrixWriter =
	b8::FixedI8MatrixWriter<
		PROJ_OUTPUTS,
		math::constexpr_inv_sqrt(SPARSE_FAN_IN)
	>;

template<
	const usize HEAD_DIM,
	const usize PROJ_OUTPUTS,
	const usize SPARSE_FAN_IN,
	const bool HAS_NORM_SCALE = false
>
struct QKVGMatrixWriter: QKVGBaseMatrixWriter<PROJ_OUTPUTS, SPARSE_FAN_IN> {
	using Base = QKVGBaseMatrixWriter<PROJ_OUTPUTS, SPARSE_FAN_IN>;

	static_assert(HEAD_DIM % 32 == 0);
	static_assert(PROJ_OUTPUTS % HEAD_DIM == 0);

	bf16 const *g_norm_scales;
	u32 s_norm_scales;
	usize norm_scale_col0;

	X17_DEVICE QKVGMatrixWriter(
		b8::FixedI8 *gC,
		bf16 const *norm_scales = nullptr
	):
		Base(gC),
		g_norm_scales(norm_scales),
		s_norm_scales(0),
		norm_scale_col0(0)
	{}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK>
	X17_HOST_DEVICE static constexpr usize smem_cols() {
		if constexpr (HAS_NORM_SCALE) {
			return N_PER_BLOCK / 2;
		} else {
			return 0;
		}
	}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK>
	X17_HOST_DEVICE static constexpr usize smem_bytes() {
		return smem_cols<M_PER_BLOCK, N_PER_BLOCK>() * sizeof(bf16);
	}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK, const u32 CAP>
	X17_DEVICE void alloc_smem(SMemAllocator<CAP> &smem_alloc) {
		if constexpr (HAS_NORM_SCALE) {
			s_norm_scales = smem_alloc.alloc(smem_bytes<M_PER_BLOCK, N_PER_BLOCK>());
		}
	}

	template<const usize M_PER_BLOCK, const usize N_PER_BLOCK, const usize THREADS_PER_BLOCK>
	X17_DEVICE void cp_async(usize row, usize col) {
		if constexpr (HAS_NORM_SCALE) {
			norm_scale_col0 = (col / N_PER_BLOCK) * N_PER_BLOCK;
			constexpr usize COLS = smem_cols<M_PER_BLOCK, N_PER_BLOCK>();
			SVector<bf16, COLS> smem(s_norm_scales);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, g_norm_scales + norm_scale_col0, smem);
		}
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

		// q, k
		X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
			X17_UNROLL for (usize ni = 0; ni < K_TILES; ++ni) {
				// TODO TODO TODO - load scales for current hi,ni - exactly 8 values, so 16 bytes
				X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
					auto &inp32x32 = acc[mi][hi*KV_TILES + ni];
					auto &out32x32 = t[mi][hi*KV_TILES + ni];
					f32 sum[4] = {0.0, 0.0, 0.0, 0.0};
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
					X17_UNROLL for (usize j = 0; j < 4; ++j) {
						sum[j] += shuffle_xor_sync(sum[j], 1);
						sum[j] += shuffle_xor_sync(sum[j], 2);
						// TODO TODO TODO - apply scale
						f32 k_scale =
							math::fast::rsqrt(sum[j])
							* f32(b8::FIXED_I8_SCALE * math::constexpr_sqrt(HEAD_DIM));
						X17_UNROLL for (usize i = 0; i < 4; ++i) {
							auto &out = out32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
							out.val0 *= k_scale;
							out.val1 *= k_scale;
						}
					}
				}
			}
		}

		// v, g
		constexpr f32 v_scale = f32(math::constexpr_inv_sqrt(SPARSE_FAN_IN) / b8::FIXED_I8_SCALE);
		X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
			X17_UNROLL for (usize hi = 0; hi < HEADS; ++hi) {
				X17_UNROLL for (usize ni = 0; ni < V_TILES; ++ni) {
					auto &inp32x32 = acc[mi][hi*KV_TILES + K_TILES + ni];
					auto &out32x32 = t[mi][hi*KV_TILES + K_TILES + ni];
					X17_UNROLL for (usize j = 0; j < 4; ++j) {
						X17_UNROLL for (usize i = 0; i < 4; ++i) {
							auto &inp = inp32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
							auto &out = out32x32.v16x32[j/2].h16x16[i/2].v8x16[j%2].h8x8[i%2];
							out.val0 = f32(inp.val0) * v_scale;
							out.val1 = f32(inp.val1) * v_scale;
						}
					}
				}
			}
		}

		Base::write(row, col, t);
	}
};
