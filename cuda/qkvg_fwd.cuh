#pragma once

#include "utils.cuh"

#pragma nv_diag_suppress 186

template<
	const usize GN,
	const usize D_IN,
	const usize FAN_IN,
	const usize N_HEAD,
	const usize D_HEAD,
	const f64 L2_NORM_EPS,
	const usize ROPE_DIM,
	const f64 ROPE_BASE
>
struct MatrixQKVGWriter {
	bf16 *gC;
	usize c_stride;
	usize seq_len;
	bf16 const *gQKNormScale_ptr;
	bf16 const *gSinkK_ptr;
	f32 *gSinkScore_ptr;

	static constexpr usize SEGMENT_SIZE = N_HEAD * D_HEAD;
	static constexpr f64 SPARSE_SCALE_2 = f64(D_IN) / f64(FAN_IN);
	static constexpr f64 G_OUT_SCALE_2 = 1.0 / f64(SEGMENT_SIZE);

	static_assert(GN == 4 * SEGMENT_SIZE);
	static_assert(ROPE_DIM <= D_HEAD);
	static_assert(ROPE_DIM % 16 == 0);

	X17_DEVICE MatrixQKVGWriter(
		bf16 *gC,
		usize seq_len,
		bf16 const *gQKNormScale_ptr,
		bf16 const *gSinkK_ptr,
		f32 *gSinkScore_ptr
	):
		gC(gC),
		c_stride(GN),
		seq_len(seq_len),
		gQKNormScale_ptr(gQKNormScale_ptr),
		gSinkK_ptr(gSinkK_ptr),
		gSinkScore_ptr(gSinkScore_ptr)
	{}

	X17_DEVICE static bool is_q_or_k(usize col) {
		return col < 2 * SEGMENT_SIZE;
	}

	X17_DEVICE static bool is_q(usize col) {
		return col < SEGMENT_SIZE;
	}

	X17_DEVICE static bool is_g(usize col) {
		return col >= 3 * SEGMENT_SIZE;
	}

	static X17_DEVICE void prepare_g_output(Fragment_8x8<f32> &g) {
		g.set(
			math::fast::gelu<SPARSE_SCALE_2, G_OUT_SCALE_2>(g.first()).val,
			g.second()
		);
	}

	template<const usize N_TILE_CNT, const usize M_TILE_CNT>
	X17_DEVICE static void prepare_g_output(Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILE_CNT]) {
		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			X17_UNROLL for (usize mi = 0; mi < M_TILE_CNT; ++mi) {
				Fragment_16x16<f32> &g = acc[ni][mi];
				prepare_g_output(g.sub[0][0]);
				prepare_g_output(g.sub[0][1]);
				prepare_g_output(g.sub[1][0]);
				prepare_g_output(g.sub[1][1]);
			}
		}
	}

	template<const usize N_TILE_CNT, const usize M_TILE_CNT>
	X17_DEVICE static void l2_norm(Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILE_CNT]) {
		static constexpr usize GROUP_TILE_CNT = D_HEAD / 16;
		static constexpr usize GROUP_CNT = (M_TILE_CNT * 16) / D_HEAD;
		static_assert((M_TILE_CNT * 16) % D_HEAD == 0);

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

	template<const usize N_TILE_CNT, const usize M_TILE_CNT>
	X17_DEVICE static void apply_q_norm_scales(
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILE_CNT],
		bf16 const *gQKNormScale_ptr,
		usize col
	) {
		usize pair_in_quad = threadIdx.x % 4;

		X17_UNROLL for (usize mi = 0; mi < M_TILE_CNT; ++mi) {
			usize col_base = col + mi * 16;
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

	template<const usize N_TILE_CNT, const usize M_TILE_CNT>
	X17_DEVICE static void store_sink_scores(
		Fragment_16x16<f32> const (&acc)[N_TILE_CNT][M_TILE_CNT],
		bf16 const *gSinkK_ptr,
		f32 *gSinkScore_ptr,
		usize seq_len,
		usize row,
		usize col
	) {
		static constexpr usize GROUP_TILE_CNT = D_HEAD / 16;
		static constexpr usize GROUP_CNT = (M_TILE_CNT * 16) / D_HEAD;
		static_assert((M_TILE_CNT * 16) % D_HEAD == 0);

		usize tid = threadIdx.x % WARP_SIZE;
		usize pair_in_quad = tid % 4;
		usize row_in_half = tid / 4;
		usize head_base = col / D_HEAD;

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
					usize top_row = row + ni * 16 + row_in_half;
					usize bot_row = top_row + 8;
					gSinkScore_ptr[head_idx * seq_len + top_row] = top[ni];
					gSinkScore_ptr[head_idx * seq_len + bot_row] = bot[ni];
				}
			}
		}
	}

	template<const usize N_TILE_CNT, const usize M_TILE_CNT>
	X17_DEVICE static void apply_rope(
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILE_CNT],
		usize row
	) {
		static constexpr usize GROUP_TILE_CNT = D_HEAD / 16;
		static constexpr usize GROUP_CNT = (M_TILE_CNT * 16) / D_HEAD;
		static_assert((M_TILE_CNT * 16) % D_HEAD == 0);
		constexpr usize ROPE_TILE_CNT = ROPE_DIM / 16;

		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			X17_UNROLL for (usize tile = 0; tile < ROPE_TILE_CNT; ++tile) {
				Fragment_16x16<f32> coefs;
				rope_coefs<ROPE_DIM, ROPE_BASE>(coefs, row + ni * 16, tile * 16);
				X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
					Fragment_16x16<f32> &frag = acc[ni][group * GROUP_TILE_CNT + tile];
					apply_rope_(frag, coefs);
				}
			}
		}
	}

	template<const usize N_TILE_CNT, const usize M_TILE_CNT>
	X17_DEVICE void write(
		usize row,
		usize col,
		Fragment_16x16<f32> (&acc)[N_TILE_CNT][M_TILE_CNT]
	) {
		static_assert((M_TILE_CNT * 16) % D_HEAD == 0);
		static_assert(SEGMENT_SIZE % (M_TILE_CNT * 16) == 0);

		if (is_q_or_k(col)) {
			l2_norm(acc);
			if (is_q(col)) {
				apply_q_norm_scales(acc, gQKNormScale_ptr, col);
				store_sink_scores(acc, gSinkK_ptr, gSinkScore_ptr, seq_len, row, col);
			}
			apply_rope(acc, row);
		} else if (is_g(col)) {
			prepare_g_output(acc);
		}

		GMatrix<bf16, 16 * N_TILE_CNT, 16 * M_TILE_CNT> gC_block{gC, c_stride};
		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			Fragment_16x16<bf16> acc_bf16[M_TILE_CNT];
			X17_UNROLL for (usize mi = 0; mi < M_TILE_CNT; ++mi) {
				acc_bf16[mi].sub[0][0].set(
					__float2bfloat16_rn(acc[ni][mi].sub[0][0].val0),
					__float2bfloat16_rn(acc[ni][mi].sub[0][0].val1)
				);
				acc_bf16[mi].sub[0][1].set(
					__float2bfloat16_rn(acc[ni][mi].sub[0][1].val0),
					__float2bfloat16_rn(acc[ni][mi].sub[0][1].val1)
				);
				acc_bf16[mi].sub[1][0].set(
					__float2bfloat16_rn(acc[ni][mi].sub[1][0].val0),
					__float2bfloat16_rn(acc[ni][mi].sub[1][0].val1)
				);
				acc_bf16[mi].sub[1][1].set(
					__float2bfloat16_rn(acc[ni][mi].sub[1][1].val0),
					__float2bfloat16_rn(acc[ni][mi].sub[1][1].val1)
				);
			}

			store(acc_bf16, gC_block, row + ni * 16, col);
		}
	}
};
