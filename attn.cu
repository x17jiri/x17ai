#include "utils2.cuh"
#include <vector>
#include <fstream>
#include <array>
#include <algorithm>

// TODO - allow K == V

constexpr usize Q_WARPS = 4;
constexpr usize KV_WARPS = 1;
constexpr usize WARPS_PER_BLOCK = Q_WARPS * KV_WARPS;
constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

constexpr usize Q_PER_WARP = 16;
constexpr usize Q_PER_BLOCK = Q_PER_WARP * Q_WARPS;
constexpr usize KV_PER_WARP = 16;
constexpr usize KV_PER_STEP = KV_PER_WARP * KV_WARPS;

constexpr usize GMEM_PRELOAD = 2;
constexpr bool LOAD_Q_DIRECTLY = false;//KV_WARPS == 1;
constexpr bool SMEM_OVERLAP_Q_WITH_KV = !LOAD_Q_DIRECTLY && Q_PER_BLOCK <= KV_PER_STEP;

X17_DEVICE void causal_mask_diagonal(Fragment_16x16<f32> &rScores) {
	usize tid = threadIdx.x % WARP_SIZE;
	usize q = tid / 4;           // 0..7
	usize k = 2 * (tid % 4);    // 0,2,4,6
	constexpr f32 NEG_INF = -INFINITY;

	rScores.sub[0][1].val0 = NEG_INF;
	rScores.sub[0][1].val1 = NEG_INF;

	rScores.sub[0][0].val0 = k <= q ? rScores.sub[0][0].val0 : NEG_INF;
	rScores.sub[1][1].val0 = k <= q ? rScores.sub[1][1].val0 : NEG_INF;

	rScores.sub[0][0].val1 = k + 1 <= q ? rScores.sub[0][0].val1 : NEG_INF;
	rScores.sub[1][1].val1 = k + 1 <= q ? rScores.sub[1][1].val1 : NEG_INF;
}

template<const usize K>
X17_DEVICE void online_softmax(
	bool first_step,
	SoftmaxStats &top,
	SoftmaxStats &bot,
	Fragment_16x16<f32> &rScores,
	Fragment_16x16<f32> (&rOut)[K]
) {
	// The `max` in `top` and `bot` is for the entire owned rows.
	// The `sum` is just the elements owned by the current thread.
	// Complete sum is calculated in combine_and_store().

	// Step 1: `max` of the owned values
	f32 new_top_max = math::max(
		math::max(rScores.sub[0][0].val0, rScores.sub[0][0].val1),
		math::max(rScores.sub[0][1].val0, rScores.sub[0][1].val1)
	);
	f32 new_bot_max = math::max(
		math::max(rScores.sub[1][0].val0, rScores.sub[1][0].val1),
		math::max(rScores.sub[1][1].val0, rScores.sub[1][1].val1)
	);

	// Step 2: Rescale outputs if needed
	f32 top_rescale = 1.0f;
	f32 bot_rescale = 1.0f;
	constexpr f32 THRESHOLD = 5.0 / math::fast::logb_2;
	bool needs_rescale = math::max(new_top_max - top.max, new_bot_max - bot.max) > THRESHOLD;
	if (any_sync(needs_rescale)) {
		new_top_max = math::max(new_top_max, shuffle_xor_sync(new_top_max, 1));
		new_top_max = math::max(new_top_max, shuffle_xor_sync(new_top_max, 2));

		new_bot_max = math::max(new_bot_max, shuffle_xor_sync(new_bot_max, 1));
		new_bot_max = math::max(new_bot_max, shuffle_xor_sync(new_bot_max, 2));

		new_top_max =
			new_top_max - top.max > THRESHOLD
				? new_top_max + THRESHOLD
				: top.max;

		new_bot_max =
			new_bot_max - bot.max > THRESHOLD
				? new_bot_max + THRESHOLD
				: bot.max;

		if (!first_step) {
			top_rescale = math::fast::expb(top.max - new_top_max);
			scale_top_(rOut, top_rescale);

			bot_rescale = math::fast::expb(bot.max - new_bot_max);
			scale_bottom_(rOut, bot_rescale);
		}

		top.max = new_top_max;
		bot.max = new_bot_max;
	}

	// Step 3: Replace scores with expb(score - max)
	rScores.sub[0][0].val0 = math::fast::expb(rScores.sub[0][0].val0 - top.max);
	rScores.sub[0][0].val1 = math::fast::expb(rScores.sub[0][0].val1 - top.max);
	rScores.sub[0][1].val0 = math::fast::expb(rScores.sub[0][1].val0 - top.max);
	rScores.sub[0][1].val1 = math::fast::expb(rScores.sub[0][1].val1 - top.max);

	rScores.sub[1][0].val0 = math::fast::expb(rScores.sub[1][0].val0 - bot.max);
	rScores.sub[1][0].val1 = math::fast::expb(rScores.sub[1][0].val1 - bot.max);
	rScores.sub[1][1].val0 = math::fast::expb(rScores.sub[1][1].val0 - bot.max);
	rScores.sub[1][1].val1 = math::fast::expb(rScores.sub[1][1].val1 - bot.max);

	// Step 4: `sum` of the owned values
	f32 top_add = (
		(rScores.sub[0][0].val0 + rScores.sub[0][0].val1)
		+ (rScores.sub[0][1].val0 + rScores.sub[0][1].val1)
	);
	top.sum = math::fma(top.sum, top_rescale, top_add);

	f32 bot_add = (
		(rScores.sub[1][0].val0 + rScores.sub[1][0].val1)
		+ (rScores.sub[1][1].val0 + rScores.sub[1][1].val1)
	);
	bot.sum = math::fma(bot.sum, bot_rescale, bot_add);
}

template<const usize K, const usize OUT_DIM>
requires(OUT_DIM == K * 16)
X17_DEVICE void combine_and_store(
	Fragment_16x16<f32> (&rOut)[K],
	SoftmaxStats top,
	SoftmaxStats bot,
	f32 sink_score,
	f32 top_score_scale,
	f32 bot_score_scale,
	f32 gate,
	u32 smem,
	usize q_warp_idx,
	usize kv_warp_idx,
	GMatrix<bf16, Q_PER_BLOCK, OUT_DIM> gOut_block
) {
	// Complete the row-wise sum reduction within each warp
	top.sum += shuffle_xor_sync(top.sum, 1);
	top.sum += shuffle_xor_sync(top.sum, 2);

	bot.sum += shuffle_xor_sync(bot.sum, 1);
	bot.sum += shuffle_xor_sync(bot.sum, 2);

	// Reduce across KV_WARPS warps that share the same Q rows via SMEM.
	// Each Q-warp group operates independently.
	// SMEM layout per Q-warp group:
	//   sReduce: (KV_WARPS-1) * Q_PER_WARP rows × K*16 cols (f32)
	//   stats:   KV_WARPS * WARP_SIZE * 4 * f32
	constexpr usize REDUCE_BYTES = sizeof(f32) * (KV_WARPS - 1) * Q_PER_WARP * K * 16;
	constexpr usize STATS_BYTES = KV_WARPS * WARP_SIZE * 4 * sizeof(f32);
	constexpr usize GROUP_BYTES = REDUCE_BYTES + STATS_BYTES;
	u32 group_smem = smem + q_warp_idx * GROUP_BYTES;
	SMatrix<f32, (KV_WARPS - 1) * Q_PER_WARP, K * 16> sReduce{group_smem};
	u32 stats_smem = sReduce._ptr + sReduce.bytes();
	usize tid = threadIdx.x % WARP_SIZE;

	// Step 1: All warps store their stats to smem (each warp gets its own slot)
	if constexpr (KV_WARPS > 1) {
		cp_async_wait<0>(); // make sure no remaining cp_async lands in our scratch
		sync_threads(); // make sure no threads are reading or writing to SMEM

		store_shared_4(
			stats_smem + (kv_warp_idx * WARP_SIZE + tid) * (4 * sizeof(f32)),
			top.sum, top.max, bot.sum, bot.max
		);

		bar_sync<KV_WARPS * WARP_SIZE>(q_warp_idx + 1);
	}

	f32 top_sink_scaled = sink_score * top_score_scale;
	f32 bot_sink_scaled = sink_score * bot_score_scale;

	f32 global_top_max, global_top_sum;
	f32 global_bot_max, global_bot_sum;

	// Step 2: Every thread reads all KV_WARPS stats, computes global max/sum
	if constexpr (KV_WARPS > 1) {
		global_top_max = top_sink_scaled;
		global_bot_max = bot_sink_scaled;

		f32 w_top_sum[KV_WARPS], w_top_max[KV_WARPS];
		f32 w_bot_sum[KV_WARPS], w_bot_max[KV_WARPS];
		X17_UNROLL for (usize w = 0; w < KV_WARPS; w++) {
			load_shared_4(
				stats_smem + (w * WARP_SIZE + tid) * (4 * sizeof(f32)),
				w_top_sum[w], w_top_max[w], w_bot_sum[w], w_bot_max[w]
			);
			global_top_max = math::max(global_top_max, w_top_max[w]);
			global_bot_max = math::max(global_bot_max, w_bot_max[w]);
		}

		global_top_sum = math::fast::expb(top_sink_scaled - global_top_max);
		global_bot_sum = math::fast::expb(bot_sink_scaled - global_bot_max);

		X17_UNROLL for (usize w = 0; w < KV_WARPS; w++) {
			global_top_sum = math::fma(
				w_top_sum[w],
				math::fast::expb(w_top_max[w] - global_top_max),
				global_top_sum
			);
			global_bot_sum = math::fma(
				w_bot_sum[w],
				math::fast::expb(w_bot_max[w] - global_bot_max),
				global_bot_sum
			);
		}
	} else {
		global_top_max = math::max(top.max, top_sink_scaled);
		global_bot_max = math::max(bot.max, bot_sink_scaled);

		global_top_sum = math::fma(
			top.sum,
			math::fast::expb(top.max - global_top_max),
			math::fast::expb(top_sink_scaled - global_top_max)
		);
		global_bot_sum = math::fma(
			bot.sum,
			math::fast::expb(bot.max - global_bot_max),
			math::fast::expb(bot_sink_scaled - global_bot_max)
		);
	}

	// Step 3: Rescale, folding in normalization and gate
	f32 top_L = math::fast::logb(global_top_sum) + global_top_max;
	f32 bot_L = math::fast::logb(global_bot_sum) + global_bot_max;

	f32 top_rescale = math::fast::expb(top.max - top_L) * gate;
	f32 bot_rescale = math::fast::expb(bot.max - bot_L) * gate;

	scale_top_(rOut, top_rescale);
	scale_bottom_(rOut, bot_rescale);

	// Step 4: accumulate results
	if constexpr (KV_WARPS > 1) {
		if (kv_warp_idx != 0) {
			// KV warps 1..N store their rescaled+normalized values to smem
			SMatrix<f32, Q_PER_WARP, K * 16> slot = sReduce.template tile_m<Q_PER_WARP>(kv_warp_idx - 1);
			fragments_to_smem(rOut, slot);

			bar_sync<KV_WARPS * WARP_SIZE>(q_warp_idx + 1);

			return;
		} else {
			// KV warp 0 accumulates and stores to gmem
			bar_sync<KV_WARPS * WARP_SIZE>(q_warp_idx + 1);

			X17_UNROLL for (usize w = 0; w < KV_WARPS - 1; w++) {
				SMatrix<f32, Q_PER_WARP, K * 16> slot = sReduce.template tile_m<Q_PER_WARP>(w);
				Fragment_16x16<f32> temp[K];
				smem_to_fragments(temp, slot);
				acc_(rOut, temp);
			}
		}
	}

	store(rOut, gOut_block, q_warp_idx * Q_PER_WARP, 0);

	// TODO - store L
}

template<usize NONROPE_DIM, usize ROPE_DIM, usize V_DIM, usize PRELOAD_DIM>
X17_DEVICE void cp_async_kv(
	GMatrixDynSize<bf16, NONROPE_DIM> gKc,
	GMatrixDynSize<bf16, ROPE_DIM> gKr,
	GMatrixDynSize<bf16, V_DIM> gV,
	SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload,
	usize p, usize kv_warp_idx, usize kv_steps
) {
	constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	if (p < kv_steps) {
		auto preload_tile = preload.template tile_m<KV_PER_STEP>(p % GMEM_PRELOAD).template tile_m<KV_PER_WARP>(kv_warp_idx);
		cp_async_gmem_to_smem<Q_WARPS * WARP_SIZE>(
			threadIdx.x % (Q_WARPS * WARP_SIZE),
			gKc.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(kv_warp_idx),
			preload_tile, 0, 0
		);
		cp_async_gmem_to_smem<Q_WARPS * WARP_SIZE>(
			threadIdx.x % (Q_WARPS * WARP_SIZE),
			gKr.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(kv_warp_idx),
			preload_tile, 0, NONROPE_DIM
		);
		cp_async_gmem_to_smem<Q_WARPS * WARP_SIZE>(
			threadIdx.x % (Q_WARPS * WARP_SIZE),
			gV.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(kv_warp_idx),
			preload_tile, 0, QK_DIM
		);
	}
}

template<usize V_DIM, usize NONROPE_DIM, usize ROPE_DIM, usize HEAD_CNT>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void
attn_forward(
	usize q_cnt, bf16 *gQ_ptr,
	usize kv_cnt, bf16 *gKc_ptr, bf16 *gKr_ptr, bf16 *gV_ptr,
	bf16 *gOut_ptr,
	f32 *sink
) {
	static_assert(V_DIM <= NONROPE_DIM, "V_DIM must be <= NONROPE_DIM for interleaved K/V register sharing");
	constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	constexpr usize NONROPE_TILES = NONROPE_DIM / 16;
	constexpr usize ROPE_TILES = ROPE_DIM / 16;
	constexpr usize QK_TILES = NONROPE_TILES + ROPE_TILES;
	constexpr usize V_TILES = V_DIM / 16;
	constexpr usize PRELOAD_DIM = QK_DIM + V_DIM;
	constexpr usize PRELOAD_TILES = QK_TILES + V_TILES;

	constexpr usize Q_STRIDE = QK_DIM * HEAD_CNT;
	constexpr usize KC_STRIDE = NONROPE_DIM * HEAD_CNT;
	constexpr usize KR_STRIDE = ROPE_DIM * HEAD_CNT;
	constexpr usize V_STRIDE = V_DIM * HEAD_CNT;

	u32 smem = 0;
	usize q_warp_idx = (threadIdx.x / WARP_SIZE) % Q_WARPS;
	usize kv_warp_idx = (threadIdx.x / WARP_SIZE) / Q_WARPS;
	SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, PRELOAD_DIM> preload{smem};
	SMatrix<bf16, Q_PER_BLOCK, QK_DIM> sQ{
		SMEM_OVERLAP_Q_WITH_KV
			? preload.template tile_m<KV_PER_STEP>(GMEM_PRELOAD - 1)._ptr
			: preload._ptr + preload.bytes()
	};

	// Load Q from GMEM to SMEM
	GMatrixDynSize<bf16, QK_DIM> gQ_full{gQ_ptr, q_cnt, Q_STRIDE};
	GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gQ_block = gQ_full.template tile_m<Q_PER_BLOCK>(blockIdx.x);
	if constexpr (!LOAD_Q_DIRECTLY) {
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gQ_block, sQ);
	}

	// TODO: this assumes `kv_cnt >= q_cnt`
	usize kv_extra = kv_cnt - q_cnt;
	usize block_q_start = blockIdx.x * Q_PER_BLOCK;
	usize kv_steps = (kv_extra + block_q_start + Q_PER_BLOCK + KV_PER_STEP - 1) / KV_PER_STEP;
	kv_steps = std::min(kv_steps, kv_cnt / KV_PER_STEP);
	usize full_kv_steps = (kv_extra + block_q_start) / KV_PER_STEP;
	full_kv_steps = std::min(full_kv_steps, kv_steps);

	// Start preloading K and V from GMEM to SMEM
	// When Q overlaps KV SMEM, don't use the last preload tile yet (it holds Q)
	constexpr usize EARLY_PRELOAD = SMEM_OVERLAP_Q_WITH_KV ? GMEM_PRELOAD - 1 : GMEM_PRELOAD;
	GMatrixDynSize<bf16, NONROPE_DIM> gKc_full{gKc_ptr, kv_cnt, KC_STRIDE};
	GMatrixDynSize<bf16, ROPE_DIM> gKr_full{gKr_ptr, kv_cnt, KR_STRIDE};
	GMatrixDynSize<bf16, V_DIM> gV_full{gV_ptr, kv_cnt, V_STRIDE};
	X17_UNROLL for (usize p = 0; p < EARLY_PRELOAD; ++p) {
		cp_async_kv(gKc_full, gKr_full, gV_full, preload, p, kv_warp_idx, kv_steps);
		cp_async_commit();
	}

	// Sink: a virtual token with no V contribution - it only adds to the
	// softmax denominator, stealing probability from real tokens.
	// sink[0] = raw score, sink[1] = output gate
	f32 sink_score = -INFINITY;
	f32 gate = 1.0f;
	if (sink != nullptr) {
		load_gmem_2(sink, sink_score, gate);
	}

	// Scalable-Softmax: score_scale = (1.0 / sqrt(QK_DIM)) * ln(n) * logb(e)
	//     1.0 / sqrt(QK_DIM) — standard attention scaling
	//     ln(n)              — SSMax factor (ln(n) = logb(n) / logb(e))
	//     logb(e)            — so we can use expb instead of exp
	// Since we are multiplying and dividing by logb(e), it cancels out, so:
	//     score_scale = (1.0 / sqrt(QK_DIM)) * logb(n)
	usize my_q_start = block_q_start + q_warp_idx * Q_PER_WARP;
	f32 top_n = my_q_start + (threadIdx.x % WARP_SIZE) / 4 + 1 + 1; // the final `+ 1` is for sink
	f32 bot_n = my_q_start + (threadIdx.x % WARP_SIZE) / 4 + 9 + 1;
	f32 top_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(top_n);
	f32 bot_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(bot_n);

	SoftmaxStats r_top; r_top.max = std::numeric_limits<f32>::lowest(); r_top.sum = 0.0f;
	SoftmaxStats r_bot; r_bot.max = std::numeric_limits<f32>::lowest(); r_bot.sum = 0.0f;
	Fragment_16x16<f32> rOut[V_TILES];
	zero_(rOut);

	cp_async_wait<EARLY_PRELOAD - 1>();
	sync_threads();

	// Load Q from SMEM to registers
	Fragment_16x16<bf16> rQ[QK_TILES];
	if constexpr (LOAD_Q_DIRECTLY) {
		load_shuffled(rQ, gQ_block, q_warp_idx * Q_PER_WARP, 0);
	} else {
		X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
			smem_tile_to_fragment(sQ, q_warp_idx * Q_PER_WARP, i * 16, rQ[i]);
		}
	}
	// Load first KV tile from SMEM to registers
	SMatrix<bf16, KV_PER_WARP, PRELOAD_DIM> sKV;
	sKV = preload.template tile_m<KV_PER_STEP>(0).template tile_m<KV_PER_WARP>(kv_warp_idx);
	Fragment_16x16<bf16> rKV[QK_TILES];
	X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
		smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
	}

	if constexpr (SMEM_OVERLAP_Q_WITH_KV) { // Now that Q is in registers, reuse its SMEM for KV
		cp_async_kv(gKc_full, gKr_full, gV_full, preload, GMEM_PRELOAD - 1, kv_warp_idx, kv_steps);
		cp_async_commit();
	}

	if constexpr (LOAD_Q_DIRECTLY) {
		load_unshuffle(rQ);
	}
	// Sequential loop over KV (causal: loop bound uses max_q_start so all
	usize kv_step = 0;
	X17_NO_UNROLL for (; kv_step < kv_steps; ++kv_step) {
		// rScores = Q * K.T, interleaved with V load (rKV: K -> V)
		Fragment_16x16<f32> rScores_f32;
		zero_(rScores_f32);
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			mma_a_bt(rQ[i], rKV[i], rScores_f32);
			smem_tile_to_fragment_trans(sKV, 0, QK_DIM + i * 16, rKV[i]);
		}
		X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
			mma_a_bt(rQ[i], rKV[i], rScores_f32);
		}

		// Scale scores must happen before masking to avoid -inf * 0 == NaN when score_scale == 0
		scale_top_(rScores_f32, top_score_scale);
		scale_bottom_(rScores_f32, bot_score_scale);

		// Causal mask: diagonal tile or full mask if past this warp's boundary
		if (kv_step >= full_kv_steps) {
			usize kv_pos = kv_step * KV_PER_STEP + kv_warp_idx * KV_PER_WARP;
			if (kv_pos == kv_extra + my_q_start) {
				causal_mask_diagonal(rScores_f32);
			} else if (kv_pos > kv_extra + my_q_start) {
				fill_(rScores_f32, -INFINITY);
			}
		}
		online_softmax(kv_step == 0, r_top, r_bot, rScores_f32, rOut);
		Fragment_16x16<bf16> rScores;
		cast(rScores_f32, rScores);

		{ // Get more data from GMEM
			// Wait for the next batch of GMEM -> SMEM preloads to complete
			cp_async_wait<GMEM_PRELOAD - 2>();
			if constexpr (KV_WARPS > 1) {
				bar_sync<Q_WARPS * WARP_SIZE>(kv_warp_idx + 1);
			} else {
				sync_threads();
			}
			sKV = preload
				.template tile_m<KV_PER_STEP>((kv_step + 1) % GMEM_PRELOAD)
				.template tile_m<KV_PER_WARP>(kv_warp_idx);

			// Preload next KV tiles from GMEM
			cp_async_kv(gKc_full, gKr_full, gV_full, preload, kv_step + GMEM_PRELOAD, kv_warp_idx, kv_steps);
			cp_async_commit();
		}

		// rOut += rScores * V, interleaved with next K load (rKV: V -> K)
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			mma_a_bt(rScores, rKV[i], rOut[i]);
			smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
		}
		// Load remaining K tiles that weren't covered by V interleaving
		X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
			smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
		}
	}

	GMatrixDynSize<bf16, V_DIM> gOut_full{gOut_ptr, q_cnt};
	GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = gOut_full.template tile_m<Q_PER_BLOCK>(blockIdx.x);
	combine_and_store(rOut, r_top, r_bot, sink_score, top_score_scale, bot_score_scale, gate, smem, q_warp_idx, kv_warp_idx, gOut_block);
}

int main(int argc, char *argv[]) {
	constexpr usize NONROPE_DIM = 128;
	constexpr usize V_DIM = 64;
	constexpr usize ROPE_DIM = 0;
	constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	{
		f32 diff = fabsf(sqrtf(QK_DIM) - f32(constexpr_sqrt(f64(QK_DIM))));
		printf("sqrtf=%e, constexpr_sqrt=%e, diff=%e\n",
			sqrtf(QK_DIM), f32(constexpr_sqrt(f64(QK_DIM))), diff);
		if (diff > 1e-8f) {
			return 1;
		}
	}
	bool use_real_data = argc <= 1;
	usize Q_LEN, KV_LEN;

	if (use_real_data) {
		Q_LEN = 1024;
		KV_LEN = 1024;
	} else {
		Q_LEN = 2*32768;
		KV_LEN = 2*32768;
	}
	srand(42);

	// allocate q: bf16 [Q_LEN, QK_DIM]
	std::vector<bf16> q_data(Q_LEN * QK_DIM);
	if (use_real_data) {
		std::ifstream in("tmp/q.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(q_data.data()),
			static_cast<std::streamsize>(q_data.size() * sizeof(*q_data.data()))
		);
	} else {
		for (size_t i = 0; i < q_data.size(); ++i) {
			q_data[i] = bf16(float(i));
			q_data[i] = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
		}
		// Save generated Q for Python verification
		std::ofstream q_out("tmp/large_q.bin", std::ios::binary);
		q_out.write(reinterpret_cast<char*>(q_data.data()),
			static_cast<std::streamsize>(q_data.size() * sizeof(bf16)));
	}
	bf16 *q_dev;
	cudaMalloc(&q_dev, q_data.size() * sizeof(bf16));
	cudaMemcpy(q_dev, q_data.data(), q_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(1) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	// allocate kv content: bf16 [KV_LEN, V_DIM]
	// allocate kv rope:    bf16 [KV_LEN, ROPE_DIM]
	std::vector<bf16> kv_data(KV_LEN * QK_DIM);
	if (use_real_data) {
		std::ifstream in("tmp/kv.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(kv_data.data()),
			static_cast<std::streamsize>(kv_data.size() * sizeof(*kv_data.data()))
		);
	} else {
		for (size_t i = 0; i < kv_data.size(); ++i) {
			kv_data[i] = bf16(float(i*100));
			kv_data[i] = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
		}
		// Save generated KV for Python verification
		std::ofstream kv_out("tmp/large_kv.bin", std::ios::binary);
		kv_out.write(reinterpret_cast<char*>(kv_data.data()),
			static_cast<std::streamsize>(kv_data.size() * sizeof(bf16)));
	}
	// Split interleaved [KV_LEN, QK_DIM] into separate K content, K rope, and V arrays
	std::vector<bf16> kc_data(KV_LEN * NONROPE_DIM);
	std::vector<bf16> kr_data(KV_LEN * ROPE_DIM);
	std::vector<bf16> v_data(KV_LEN * V_DIM);
	for (size_t r = 0; r < KV_LEN; r++) {
		for (size_t c = 0; c < NONROPE_DIM; c++) {
			kc_data[r * NONROPE_DIM + c] = kv_data[r * QK_DIM + c];
		}
		for (size_t c = 0; c < V_DIM; c++) {
			v_data[r * V_DIM + c] = kv_data[r * QK_DIM + c];
		}
		if constexpr (ROPE_DIM > 0) {
			#pragma nv_diag_suppress 186
			for (size_t c = 0; c < ROPE_DIM; c++) {
				kr_data[r * ROPE_DIM + c] = kv_data[r * QK_DIM + NONROPE_DIM + c];
			}
			#pragma nv_diag_default 186
		}
	}
	bf16 *kc_dev, *kr_dev, *v_dev;
	cudaMalloc(&kc_dev, kc_data.size() * sizeof(bf16));
	cudaMemcpy(kc_dev, kc_data.data(), kc_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMalloc(&kr_dev, kr_data.size() * sizeof(bf16));
	cudaMemcpy(kr_dev, kr_data.data(), kr_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMalloc(&v_dev, v_data.size() * sizeof(bf16));
	cudaMemcpy(v_dev, v_data.data(), v_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	// allocate output: bf16 [Q_LEN, V_DIM]
	std::vector<bf16> out_data(Q_LEN * V_DIM);
	bf16 *out_dev;
	size_t out_size_bytes = out_data.size() * sizeof(bf16);
	cudaMalloc(&out_dev, out_size_bytes);
	GMatrixDynSize<bf16, V_DIM> out{out_dev, Q_LEN};

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(2) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(3) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	constexpr usize PRELOAD_DIM = QK_DIM + V_DIM;
	usize smem_size =
		sizeof(bf16) * KV_PER_STEP * GMEM_PRELOAD * PRELOAD_DIM
		+ (SMEM_OVERLAP_Q_WITH_KV ? 0 : sizeof(bf16) * Q_PER_BLOCK * QK_DIM);
	printf("smem_size = %d bytes\n", smem_size);
	//smem_size = std::max(smem_size, usize(70 * 1024));

	cudaFuncSetAttribute(attn_forward<V_DIM, NONROPE_DIM, ROPE_DIM, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

	cudaFuncSetAttribute(attn_forward<V_DIM, NONROPE_DIM, ROPE_DIM, 1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	// Allocate sink+gate buffer: [sink_score, gate]
	f32 sink_host[2] = { -0.3f, 0.5f };
	f32 *sink_dev;
	cudaMalloc(&sink_dev, sizeof(sink_host));
	cudaMemcpy(sink_dev, sink_host, sizeof(sink_host), cudaMemcpyHostToDevice);
	f32 *sink_ptr = use_real_data ? sink_dev : nullptr;

	cudaDeviceSynchronize();

	int WARMUP = use_real_data ? 0 : 50;
	//WARMUP = 0;
	for (int i = 0; i < WARMUP; ++i) {
		attn_forward
			<V_DIM, NONROPE_DIM, ROPE_DIM, 1>
			<<<Q_LEN / Q_PER_BLOCK, THREADS_PER_BLOCK, smem_size>>>
			(
				Q_LEN, q_dev,
				KV_LEN, kc_dev, kr_dev, v_dev,
				out_dev,
				sink_ptr
			);
	}

	cudaDeviceSynchronize();

	usize NUM_BLOCKS = Q_LEN / Q_PER_BLOCK;
	int NUM_RUNS = use_real_data ? 1 : 200;
	//NUM_RUNS = 1;
	std::vector<cudaEvent_t> starts(NUM_RUNS), ends(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventRecord(starts[i]);
		attn_forward
			<V_DIM, NONROPE_DIM, ROPE_DIM, 1>
			<<<NUM_BLOCKS, THREADS_PER_BLOCK, smem_size>>>
			(
				Q_LEN, q_dev,
				KV_LEN, kc_dev, kr_dev, v_dev,
				out_dev,
				sink_ptr
			);
		cudaEventRecord(ends[i]);
	}
	cudaDeviceSynchronize();

	std::vector<float> times_ms(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventElapsedTime(&times_ms[i], starts[i], ends[i]);
		cudaEventDestroy(starts[i]);
		cudaEventDestroy(ends[i]);
	}
	std::sort(times_ms.begin(), times_ms.end());

	int mid = NUM_RUNS / 2;
	float median_ms = times_ms[mid];
	float min_ms = times_ms[0];
	//printf("Kernel time over %d runs: median %.4f ms  min %.4f ms\n", NUM_RUNS, median_ms, min_ms);

	// TFLOPS: Q@K^T = 2*Q*KV*QK_DIM, attn@V = 2*Q*KV*V_DIM, softmax ~ 5*Q*KV
	// Causal ≈ half the work
	double flops_causal = (2.0 * Q_LEN * KV_LEN * QK_DIM + 2.0 * Q_LEN * KV_LEN * V_DIM + 5.0 * Q_LEN * KV_LEN) / 2.0;
	printf("TFLOPS (causal): %.2f\n", flops_causal / (median_ms * 1e-3) / 1e12);

	// write output to file
	{
		std::ofstream out_file("tmp/out_cpu.bin", std::ios::binary);
		cudaMemcpy(out_data.data(), out_dev, out_size_bytes, cudaMemcpyDeviceToHost);
		out_file.write(
			reinterpret_cast<char *>(out_data.data()),
			static_cast<std::streamsize>(out_data.size() * sizeof(*out_data.data()))
		);
	}

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(4) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	// Print first 8 rows, first 8 cols
	printf("\nFirst 8 rows, first 8 cols:\n");
	for (size_t r = 0; r < 8; r++) {
		for (size_t c = 0; c < 8; c++) {
			printf("%12.6e ", double(float(out_data[r * V_DIM + c])));
		}
		printf("\n");
	}

	// Print last 8 rows, last 8 cols
	printf("\nLast 8 rows, last 8 cols:\n");
	for (size_t r = Q_LEN - 8; r < Q_LEN; r++) {
		for (size_t c = V_DIM - 8; c < V_DIM; c++) {
			printf("%12.6e ", double(float(out_data[r * V_DIM + c])));
		}
		printf("\n");
	}

	// Check for non-finite values
	size_t nan_count = 0;
	size_t inf_count = 0;
	std::vector<size_t> nan_rows;
	std::vector<size_t> inf_rows;
	for (size_t i = 0; i < out_data.size(); ++i) {
		float v = float(out_data[i]);
		if (!isfinite(v)) {
			if (isnan(v)) {
				nan_count++;
				nan_rows.push_back(i);
			} else {
				inf_count++;
				inf_rows.push_back(i);
			}
		}
	}
	if (inf_count > 0 || nan_count > 0) {
		printf("\n*** WARNING: output contains %zu infinite values and %zu NaNs ***\n", inf_count, nan_count);
		if (nan_count > 0) {
			printf("NaN rows (up to 100): ");
			for (size_t i = 0; i < std::min(nan_rows.size(), size_t(100)); ++i) {
				printf("%zu ", nan_rows[i]);
			}
			printf("\n");
		}
		if (inf_count > 0) {
			printf("Inf rows (up to 100): ");
			for (size_t i = 0; i < std::min(inf_rows.size(), size_t(100)); ++i) {
				printf("%zu ", inf_rows[i]);
			}
			printf("\n");
		}
	}

	return 0;
}
