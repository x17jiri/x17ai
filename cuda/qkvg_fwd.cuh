#pragma once

#include "utils.cuh"

#pragma nv_diag_suppress 186

template<
	const usize GN,
	const usize D_IN,
	const usize FAN_IN,
	const usize N_HEADS,
	const usize QK_DIM,
	const usize VG_DIM,
	const f64 L2_NORM_EPS,
	const f64 V_SCALE_FIX
>
struct MatrixQKVGWriter: MatrixWriter<GN> {
	bf16 const *gQKNormScale_ptr;

	static constexpr usize QK_SEGMENT_SIZE = N_HEADS * QK_DIM;
	static constexpr usize VG_SEGMENT_SIZE = N_HEADS * VG_DIM;
	static constexpr f64 SPARSE_SCALE_2 = f64(D_IN) / f64(FAN_IN);
	static constexpr f64 G_OUT_SCALE_2 = 1.0 / f64(VG_SEGMENT_SIZE);

	X17_DEVICE MatrixQKVGWriter(
		bf16 *gC,
		bf16 const *gQKNormScale_ptr
	):
		MatrixWriter<GN>(gC),
		gQKNormScale_ptr(gQKNormScale_ptr)
	{}

	X17_DEVICE static bool is_q_or_k(usize col) {
		return col < (2 * QK_SEGMENT_SIZE);
	}

	X17_DEVICE static bool is_q(usize col) {
		return col < QK_SEGMENT_SIZE;
	}

	X17_DEVICE static bool is_g(usize col) {
		return col >= (2*QK_SEGMENT_SIZE + VG_SEGMENT_SIZE);
	}

	static X17_DEVICE void prepare_g_output_(Fragment_8x8<f32> &g) {
		f32 gelu =
			math::fast::gelu<
				SPARSE_SCALE_2,
				G_OUT_SCALE_2 * SPARSE_SCALE_2 * (V_SCALE_FIX * V_SCALE_FIX)
			>(g.first()).val;
		f32 raw = g.second();
		g.set(gelu, raw);
	}

	template<const usize M_TILES, const usize N_TILES>
	X17_DEVICE static void prepare_g_output_(Fragment_16x16<f32> (&acc)[M_TILES][N_TILES]) {
		X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
			X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
				Fragment_16x16<f32> &g = acc[mi][ni];
				prepare_g_output_(g.sub[0][0]);
				prepare_g_output_(g.sub[0][1]);
				prepare_g_output_(g.sub[1][0]);
				prepare_g_output_(g.sub[1][1]);
			}
		}
	}

	template<const usize D_HEAD, const usize M_TILES, const usize N_TILES>
	X17_DEVICE static void l2_norm_(Fragment_16x16<f32> (&acc)[M_TILES][N_TILES]) {
		static constexpr usize GROUP_TILE_CNT = D_HEAD / 16;
		static constexpr usize GROUP_CNT = (N_TILES * 16) / D_HEAD;
		static_assert((N_TILES * 16) % D_HEAD == 0);

		X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
			X17_UNROLL for (usize group = 0; group < GROUP_CNT; ++group) {
				f32 top_sum_sq = 0.0f;
				f32 bot_sum_sq = 0.0f;

				X17_UNROLL for (usize tile = 0; tile < GROUP_TILE_CNT; ++tile) {
					Fragment_16x16<f32> &frag = acc[mi][group * GROUP_TILE_CNT + tile];
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
					Fragment_16x16<f32> &frag = acc[mi][group * GROUP_TILE_CNT + tile];
					frag.scale_top_(top_inv_norm);
					frag.scale_bottom_(bot_inv_norm);
				}
			}
		}
	}

	template<const usize M_TILES, const usize N_TILES>
	X17_DEVICE static void apply_q_norm_scales_(
		Fragment_16x16<f32> (&acc)[M_TILES][N_TILES],
		bf16 const *gQKNormScale_ptr,
		usize col
	) {
		usize pair_in_quad = threadIdx.x % 4;

		X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
			usize col_base = col + ni * 16;
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

			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				Fragment_16x16<f32> &frag = acc[mi][ni];
				frag.sub[0][0].set(frag.sub[0][0].first() * left0, frag.sub[0][0].second() * left1);
				frag.sub[1][0].set(frag.sub[1][0].first() * left0, frag.sub[1][0].second() * left1);
				frag.sub[0][1].set(frag.sub[0][1].first() * right0, frag.sub[0][1].second() * right1);
				frag.sub[1][1].set(frag.sub[1][1].first() * right0, frag.sub[1][1].second() * right1);
			}
		}
	}

	template<const usize M_TILES, const usize N_TILES>
	X17_DEVICE void write(
		usize row, usize col,
		Fragment_16x16<f32> (&acc)[M_TILES][N_TILES]
	) {
		static_assert((N_TILES * 16) % QK_DIM == 0);
		static_assert((N_TILES * 16) % VG_DIM == 0);
		static_assert(QK_SEGMENT_SIZE % (N_TILES * 16) == 0);
		static_assert(VG_SEGMENT_SIZE % (N_TILES * 16) == 0);

		if (is_q_or_k(col)) {
			l2_norm_<QK_DIM>(acc);
			if (is_q(col)) {
				apply_q_norm_scales_(acc, gQKNormScale_ptr, col);
			}
		} else if (is_g(col)) {
			prepare_g_output_(acc);
		}

		MatrixWriter<GN>::write(row, col, acc);
	}
};
