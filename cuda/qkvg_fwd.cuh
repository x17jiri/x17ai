#pragma once

#include "utils.cuh"

#pragma nv_diag_suppress 186

template<
	typename _MatMul,
	const usize N_HEAD,
	const usize D_HEAD,
	const f64 L2_NORM_EPS,
	const usize ROPE_DIM,
	const f64 ROPE_BASE
>
struct QKVGFwd {
	using MatMul = _MatMul;

	static constexpr usize M = MatMul::M;
	static constexpr usize N = MatMul::N;
	static constexpr usize M_WARPS = MatMul::M_WARPS;
	static constexpr usize N_WARPS = MatMul::N_WARPS;
	static constexpr usize M_PER_WARP = MatMul::M_PER_WARP;
	static constexpr usize N_PER_WARP = MatMul::N_PER_WARP;
	static constexpr usize M_PER_BLOCK = MatMul::M_PER_BLOCK;
	static constexpr usize N_PER_BLOCK = MatMul::N_PER_BLOCK;
	static constexpr usize M_TILES = MatMul::M_TILES;
	static constexpr usize N_TILES = MatMul::N_TILES;

	static constexpr usize THREADS_PER_BLOCK = MatMul::THREADS_PER_BLOCK;
	static constexpr usize SMEM_BYTES = MatMul::SMEM_BYTES;

	static constexpr usize SEGMENT_SIZE = N_HEAD * D_HEAD;
	static constexpr usize GROUP_CNT = M_PER_WARP / D_HEAD;
	static constexpr usize GROUP_TILE_CNT = D_HEAD / 16;

	static_assert(ROPE_DIM <= D_HEAD);
	static_assert(ROPE_DIM % 16 == 0);

	X17_DEVICE bool is_q_or_k(usize block_m, usize warp_m) {
		usize warp_col = block_m + warp_m;
		return warp_col < 2 * SEGMENT_SIZE;
	}

	X17_DEVICE bool is_q(usize block_m, usize warp_m) {
		usize warp_col = block_m + warp_m;
		return warp_col < SEGMENT_SIZE;
	}

	template<usize N_TILE_CNT>
	X17_DEVICE void l2_norm(Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES]) {
		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
				f32 top_sum_sq = 0.0f;
				f32 bot_sum_sq = 0.0f;

				X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
					Fragment_16x16<f32> &frag = acc[ni][group * GROUP_TILE_CNT + tile];
					top_sum_sq = math::fma(frag.sub[0][0].val0, frag.sub[0][0].val0, top_sum_sq);
					top_sum_sq = math::fma(frag.sub[0][0].val1, frag.sub[0][0].val1, top_sum_sq);
					top_sum_sq = math::fma(frag.sub[0][1].val0, frag.sub[0][1].val0, top_sum_sq);
					top_sum_sq = math::fma(frag.sub[0][1].val1, frag.sub[0][1].val1, top_sum_sq);

					bot_sum_sq = math::fma(frag.sub[1][0].val0, frag.sub[1][0].val0, bot_sum_sq);
					bot_sum_sq = math::fma(frag.sub[1][0].val1, frag.sub[1][0].val1, bot_sum_sq);
					bot_sum_sq = math::fma(frag.sub[1][1].val0, frag.sub[1][1].val0, bot_sum_sq);
					bot_sum_sq = math::fma(frag.sub[1][1].val1, frag.sub[1][1].val1, bot_sum_sq);
				}

				top_sum_sq += shuffle_xor_sync(top_sum_sq, 1);
				top_sum_sq += shuffle_xor_sync(top_sum_sq, 2);
				bot_sum_sq += shuffle_xor_sync(bot_sum_sq, 1);
				bot_sum_sq += shuffle_xor_sync(bot_sum_sq, 2);

				f32 top_inv_norm = math::fast::rsqrt(top_sum_sq + L2_NORM_EPS);
				f32 bot_inv_norm = math::fast::rsqrt(bot_sum_sq + L2_NORM_EPS);

				X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
					Fragment_16x16<f32> &frag = acc[ni][group * GROUP_TILE_CNT + tile];
					frag.scale_top_(top_inv_norm);
					frag.scale_bottom_(bot_inv_norm);
				}
			}
		}
	}

	template<usize N_TILE_CNT>
	X17_DEVICE void apply_q_norm_scales(
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES],
		bf16 const *gQKNormScale_ptr,
		usize block_m,
		usize warp_m
	) {
		usize pair_in_quad = threadIdx.x % 4;

		X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
			usize col_base = block_m + warp_m + mi * 16;
			usize left_col = col_base + pair_in_quad * 2;
			usize right_col = left_col + 8;

			union {
				u32 packed;
				bf16 values[2];
			} left_scale, right_scale;

			left_scale.packed = __float_as_uint(load_gmem_1x32b(reinterpret_cast<f32 const *>(gQKNormScale_ptr + left_col)));
			right_scale.packed = __float_as_uint(load_gmem_1x32b(reinterpret_cast<f32 const *>(gQKNormScale_ptr + right_col)));

			f32 left0 = f32(left_scale.values[0]);
			f32 left1 = f32(left_scale.values[1]);
			f32 right0 = f32(right_scale.values[0]);
			f32 right1 = f32(right_scale.values[1]);

			X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
				Fragment_16x16<f32> &frag = acc[ni][mi];
				frag.sub[0][0].set(frag.sub[0][0].first() * left0, frag.sub[0][0].second() * left1);
				frag.sub[1][0].set(frag.sub[1][0].first() * left0, frag.sub[1][0].second() * left1);
				frag.sub[0][1].set(frag.sub[0][1].first() * right0, frag.sub[0][1].second() * right1);
				frag.sub[1][1].set(frag.sub[1][1].first() * right0, frag.sub[1][1].second() * right1);
			}
		}
	}

	template<usize N_TILE_CNT>
	X17_DEVICE void store_sink_scores(
		Fragment_16x16<f32> const (&acc)[N_TILE_CNT][M_TILES],
		bf16 const *gSinkK_ptr,
		f32 *gSinkScore_ptr,
		usize seq_len,
		usize block_m,
		usize block_n,
		usize warp_m,
		usize warp_n
	) {
		usize tid = threadIdx.x % WARP_SIZE;
		usize pair_in_quad = tid % 4;
		usize row_in_half = tid / 4;
		usize head_base = (block_m + warp_m) / D_HEAD;

		X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
			usize head_idx = head_base + group;
			bf16 const *sink_ptr = gSinkK_ptr + head_idx * D_HEAD;
			f32 top[N_TILE_CNT];
			f32 bot[N_TILE_CNT];
			X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
				top[ni] = 0.0f;
				bot[ni] = 0.0f;
			}

			X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
				usize col_base = group * GROUP_TILE_CNT * 16 + tile * 16;
				usize left_col = col_base + pair_in_quad * 2;
				usize right_col = left_col + 8;

				union {
					u32 packed;
					bf16 values[2];
				} left_sink, right_sink;

				left_sink.packed = __float_as_uint(load_gmem_1x32b(reinterpret_cast<f32 const *>(sink_ptr + left_col)));
				right_sink.packed = __float_as_uint(load_gmem_1x32b(reinterpret_cast<f32 const *>(sink_ptr + right_col)));

				f32 left0 = f32(left_sink.values[0]);
				f32 left1 = f32(left_sink.values[1]);
				f32 right0 = f32(right_sink.values[0]);
				f32 right1 = f32(right_sink.values[1]);

				X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
					Fragment_16x16<f32> const &frag = acc[ni][group * GROUP_TILE_CNT + tile];
					top[ni] = math::fma(frag.sub[0][0].first(), left0, top[ni]);
					top[ni] = math::fma(frag.sub[0][0].second(), left1, top[ni]);
					top[ni] = math::fma(frag.sub[0][1].first(), right0, top[ni]);
					top[ni] = math::fma(frag.sub[0][1].second(), right1, top[ni]);

					bot[ni] = math::fma(frag.sub[1][0].first(), left0, bot[ni]);
					bot[ni] = math::fma(frag.sub[1][0].second(), left1, bot[ni]);
					bot[ni] = math::fma(frag.sub[1][1].first(), right0, bot[ni]);
					bot[ni] = math::fma(frag.sub[1][1].second(), right1, bot[ni]);
				}
			}

			X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
				top[ni] += shuffle_xor_sync(top[ni], 1);
				top[ni] += shuffle_xor_sync(top[ni], 2);
				bot[ni] += shuffle_xor_sync(bot[ni], 1);
				bot[ni] += shuffle_xor_sync(bot[ni], 2);
			}

			if (pair_in_quad == 0) {
				X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
					usize top_row = block_n + warp_n + ni * 16 + row_in_half;
					usize bot_row = top_row + 8;
					gSinkScore_ptr[head_idx * seq_len + top_row] = top[ni];
					gSinkScore_ptr[head_idx * seq_len + bot_row] = bot[ni];
				}
			}
		}
	}

	template<usize N_TILE_CNT>
	X17_DEVICE void apply_rope(
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILES],
		usize block_n,
		usize warp_n
	) {
		constexpr usize ROPE_TILE_CNT = ROPE_DIM / 16;

		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			X17_UNROLL for (usize tile = 0; tile < ROPE_TILE_CNT; ++tile) {
				Fragment_16x16<f32> coefs;
				rope_coefs<ROPE_DIM, ROPE_BASE>(coefs, block_n + warp_n + ni * 16, tile * 16);
				X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
					Fragment_16x16<f32> &frag = acc[ni][group * GROUP_TILE_CNT + tile];
					apply_rope_(frag, coefs);
				}
			}
		}
	}

	X17_DEVICE void run_matmul(
		bf16 *A,
		bf16 *B,
		Fragment_16x16<f32> (&acc_t)[N_TILES][M_TILES]
	) {
		MatMul matmul = MatMul();
		matmul.run(A, B, acc_t);
	}

	X17_DEVICE void run_epilogue(
		Fragment_16x16<f32> (&acc_t)[MatMul::N_TILES][MatMul::M_TILES],
		bf16 *C,

		usize seq_len,
		bf16 const *gQKNormScale_ptr,
		bf16 const *gSinkK_ptr,
		f32 *gSinkScore_ptr
	) {
		usize tid = threadIdx.x;
		usize warp_idx = tid / WARP_SIZE;
		usize block_m = blockIdx.x * M_PER_BLOCK;
		usize block_n = blockIdx.y * N_PER_BLOCK;
		usize warp_m = (warp_idx / N_WARPS) * M_PER_WARP;
		usize warp_n = (warp_idx % N_WARPS) * N_PER_WARP;

		if (is_q_or_k(block_m, warp_m)) {
			l2_norm(acc_t);
			if (is_q(block_m, warp_m)) {
				apply_q_norm_scales(acc_t, gQKNormScale_ptr, block_m, warp_m);
				store_sink_scores(acc_t, gSinkK_ptr, gSinkScore_ptr, seq_len, block_m, block_n, warp_m, warp_n);
			}
			apply_rope(acc_t, block_n, warp_n);
		}

		bf16 *c_ptr = C + blockIdx.y * N_PER_BLOCK * M + blockIdx.x * M_PER_BLOCK;
		GMatrix<bf16, N_PER_BLOCK, M_PER_BLOCK> gC_block{c_ptr, M};
		X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
			Fragment_16x16<bf16> acc_bf16[M_TILES];
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				acc_bf16[mi].sub[0][0].set(
					__float2bfloat16_rn(acc_t[ni][mi].sub[0][0].val0),
					__float2bfloat16_rn(acc_t[ni][mi].sub[0][0].val1)
				);
				acc_bf16[mi].sub[0][1].set(
					__float2bfloat16_rn(acc_t[ni][mi].sub[0][1].val0),
					__float2bfloat16_rn(acc_t[ni][mi].sub[0][1].val1)
				);
				acc_bf16[mi].sub[1][0].set(
					__float2bfloat16_rn(acc_t[ni][mi].sub[1][0].val0),
					__float2bfloat16_rn(acc_t[ni][mi].sub[1][0].val1)
				);
				acc_bf16[mi].sub[1][1].set(
					__float2bfloat16_rn(acc_t[ni][mi].sub[1][1].val0),
					__float2bfloat16_rn(acc_t[ni][mi].sub[1][1].val1)
				);
			}

			// TODO - this is matmul epilogue
			store(acc_bf16, gC_block, warp_n + ni * 16, warp_m);
		}
	}
};

template<typename QKVGFwd>
__global__ __launch_bounds__(QKVGFwd::THREADS_PER_BLOCK) void qkvg_fwd(
	bf16 *A,
	bf16 *B,
	bf16 *C,
	usize seq_len,
	bf16 const *gQKNormScale_ptr,
	bf16 const *gSinkK_ptr,
	f32 *gSinkScore_ptr
) {
	QKVGFwd qkvg_fwd = QKVGFwd();
	Fragment_16x16<f32> acc_t[QKVGFwd::N_TILES][QKVGFwd::M_TILES];
	qkvg_fwd.run_matmul(A, B, acc_t);
	qkvg_fwd.run_epilogue(
		acc_t, C,
		seq_len, gQKNormScale_ptr, gSinkK_ptr, gSinkScore_ptr
	);
}
