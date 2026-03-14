#include "utils2.cuh"
#include <vector>
#include <fstream>
#include <array>
#include <algorithm>

constexpr usize Q_PER_WARP = 16;
constexpr usize KV_PER_WARP = 16;
constexpr usize WARPS_PER_BLOCK = 4;
constexpr usize GMEM_PRELOAD = 2;
constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
//constexpr usize BLOCKS_PER_SM = std::max<usize>(1, 256 / THREADS_PER_BLOCK);
constexpr usize Q_PER_BLOCK = Q_PER_WARP;
constexpr usize KV_PER_STEP = KV_PER_WARP * WARPS_PER_BLOCK;

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
		new_top_max = math::max(new_top_max, shfl_xor_sync(new_top_max, 1));
		new_top_max = math::max(new_top_max, shfl_xor_sync(new_top_max, 2));

		new_bot_max = math::max(new_bot_max, shfl_xor_sync(new_bot_max, 1));
		new_bot_max = math::max(new_bot_max, shfl_xor_sync(new_bot_max, 2));

		bool first_step = all_sync(top.sum == 0.0f && bot.sum == 0.0f);
		if (first_step) {
			zero_(rOut);
			top.max = new_top_max + THRESHOLD;
			bot.max = new_bot_max + THRESHOLD;
		} else {
			new_top_max =
				new_top_max - top.max > THRESHOLD
					? new_top_max + THRESHOLD
					: top.max;

			new_bot_max =
				new_bot_max - bot.max > THRESHOLD
					? new_bot_max + THRESHOLD
					: bot.max;

			top_rescale = math::fast::expb(top.max - new_top_max);
			bot_rescale = math::fast::expb(bot.max - new_bot_max);

			scale_top_(rOut, top_rescale);
			scale_bottom_(rOut, bot_rescale);

			top.max = new_top_max;
			bot.max = new_bot_max;
		}
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

template<const usize K, const usize OUT_DIM, const usize Q_PER_BLOCK>
requires(Q_PER_BLOCK == 16 && OUT_DIM == K * 16)
X17_DEVICE void combine_and_store(
	Fragment_16x16<f32> (&rOut)[K],
	SoftmaxStats top,
	SoftmaxStats bot,
	f32 sink_score,
	f32 top_score_scale,
	f32 bot_score_scale,
	f32 gate,
	u32 smem,
	usize warp_idx,
	GMatrix<bf16, Q_PER_BLOCK, OUT_DIM> gOut_block
) {
	// Step 1: All warps store their stats to smem
	top.sum += shfl_xor_sync(top.sum, 1);
	top.sum += shfl_xor_sync(top.sum, 2);

	bot.sum += shfl_xor_sync(bot.sum, 1);
	bot.sum += shfl_xor_sync(bot.sum, 2);

	SMatrix<f32, (WARPS_PER_BLOCK - 1) * Q_PER_WARP, K * 16> sReduce{smem};
	u32 stats_smem = sReduce._ptr + sReduce.bytes();
	store_shared_4(
		stats_smem + threadIdx.x * (4 * sizeof(f32)),
		top.sum, top.max, bot.sum, bot.max
	);
	__syncthreads();

	// Step 2: Every thread reads all warps' stats for its rows, computes global max/sum
	usize tid = threadIdx.x % WARP_SIZE;
	f32 w_top_sum[WARPS_PER_BLOCK], w_top_max[WARPS_PER_BLOCK];
	f32 w_bot_sum[WARPS_PER_BLOCK], w_bot_max[WARPS_PER_BLOCK];
	X17_UNROLL for (usize w = 0; w < WARPS_PER_BLOCK; w++) {
		load_shared_4(
			stats_smem + (w * WARP_SIZE + tid) * (4 * sizeof(f32)),
			w_top_sum[w], w_top_max[w], w_bot_sum[w], w_bot_max[w]
		);
	}

	f32 top_sink_scaled = sink_score * top_score_scale;
	f32 bot_sink_scaled = sink_score * bot_score_scale;

	f32 global_top_max = math::max(math::max(w_top_max), top_sink_scaled);
	f32 global_bot_max = math::max(math::max(w_bot_max), bot_sink_scaled);

	f32 global_top_sum = math::fast::expb(top_sink_scaled - global_top_max);
	f32 global_bot_sum = math::fast::expb(bot_sink_scaled - global_bot_max);
	X17_UNROLL for (usize w = 0; w < WARPS_PER_BLOCK; w++) {
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

	// Step 3: Each warp rescales its values, folding in normalization and gate
	f32 top_L = math::fast::logb(global_top_sum) + global_top_max;
	f32 bot_L = math::fast::logb(global_bot_sum) + global_bot_max;

	f32 top_rescale = math::fast::expb(top.max - top_L) * gate;
	f32 bot_rescale = math::fast::expb(bot.max - bot_L) * gate;

	scale_top_(rOut, top_rescale);
	scale_bottom_(rOut, bot_rescale);

	// Step 4: Warps 1-N store their rescaled+normalized values to smem
	if (warp_idx != 0) {
		SMatrix<f32, Q_PER_WARP, K * 16> slot = sReduce.template tile_m<Q_PER_WARP>(warp_idx - 1);
		fragments_to_smem(rOut, slot);
	}
	__syncthreads();

	// Step 5: Warp 0 accumulates and stores to gmem
	if (warp_idx == 0) {
		X17_UNROLL for (usize w = 0; w < WARPS_PER_BLOCK - 1; w++) {
			SMatrix<f32, Q_PER_WARP, K * 16> slot = sReduce.template tile_m<Q_PER_WARP>(w);
			Fragment_16x16<f32> temp[K];
			smem_to_fragments(temp, slot);
			acc_(rOut, temp);
		}

		store(gOut_block, 0, 0, rOut);

		// TODO - store L
	}
}

template<usize V_DIM, usize ROPE_DIM, usize HEAD_SIZE>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void
attn_kernel(
	usize q_cnt, bf16 *gQ_ptr,
	usize kv_cnt, bf16 *gKVc_ptr, bf16 *gKVr_ptr,
	bf16 *gOut_ptr,
	f32 *sink
) {
	constexpr usize QK_DIM = V_DIM + ROPE_DIM;
	constexpr usize V_TILES = V_DIM / 16;
	constexpr usize ROPE_TILES = ROPE_DIM / 16;
	constexpr usize QK_TILES = V_TILES + ROPE_TILES;

	constexpr usize Q_STRIDE = QK_DIM * HEAD_SIZE;
	constexpr usize KVC_STRIDE = V_DIM * HEAD_SIZE;
	constexpr usize KVR_STRIDE = ROPE_DIM * HEAD_SIZE;

	u32 smem = 0;
	// Two preload buffers back-to-back: content [256, V_DIM] + rope [256, ROPE_DIM]
	SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, V_DIM> preload_c{smem};
	SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, ROPE_DIM> preload_r{preload_c._ptr + preload_c.bytes()};
	usize warp_idx = threadIdx.x / WARP_SIZE;

	// Load Q from GMEM to SMEM. Use the last preload tiles.
	static_assert(Q_PER_BLOCK <= KV_PER_STEP);
	GMatrixDynSize<bf16, QK_DIM> gQ_full{gQ_ptr, q_cnt, Q_STRIDE};
	auto gQc_full = gQ_full.template slice_n<V_DIM>(0);
	auto gQr_full = gQ_full.template slice_n<ROPE_DIM>(V_DIM);
	GMatrix<bf16, Q_PER_BLOCK, V_DIM> gQc_block = gQc_full.template tile_m<Q_PER_BLOCK>(blockIdx.x);
	SMatrix<bf16, Q_PER_BLOCK, V_DIM> sQc;
	sQc = preload_c.template tile_m<KV_PER_STEP>(GMEM_PRELOAD - 1).template tile_m<Q_PER_BLOCK>(0);
	GMatrix<bf16, Q_PER_BLOCK, ROPE_DIM> gQr_block = gQr_full.template tile_m<Q_PER_BLOCK>(blockIdx.x);
	SMatrix<bf16, Q_PER_BLOCK, ROPE_DIM> sQr;
	sQr = preload_r.template tile_m<KV_PER_STEP>(GMEM_PRELOAD - 1).template tile_m<Q_PER_BLOCK>(0);
	cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gQc_block, sQc);
	cp_async_gmem_to_smem<THREADS_PER_BLOCK>(threadIdx.x, gQr_block, sQr);
	cp_async_commit();

	// Start preloading KVs from GMEM to SMEM
	// Don't use the last preload tiles yet because they're used to load Q
	GMatrixDynSize<bf16, V_DIM> gKVc_full{gKVc_ptr, kv_cnt, KVC_STRIDE};
	GMatrixDynSize<bf16, ROPE_DIM> gKVr_full{gKVr_ptr, kv_cnt, KVR_STRIDE};
	size_t kv_steps = gKVc_full.m_rows() / KV_PER_STEP;
	X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD - 1; ++p) {
		if (p < kv_steps) {
			cp_async_gmem_to_smem<WARP_SIZE>(
				threadIdx.x % WARP_SIZE,
				gKVc_full.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx),
				preload_c.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx)
			);
			cp_async_gmem_to_smem<WARP_SIZE>(
				threadIdx.x % WARP_SIZE,
				gKVr_full.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx),
				preload_r.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx)
			);
		}
		cp_async_commit();
	}

	// Load Q from SMEM to registers
	cp_async_wait<GMEM_PRELOAD - 1>();
	__syncthreads();
	Fragment_16x16<bf16> rQ[QK_TILES];
	X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
		smem_tile_to_fragment(sQc, 0, i * 16, rQ[i]);
	}
	if constexpr (ROPE_TILES > 0) {
		X17_UNROLL for (usize i = 0; i < ROPE_TILES; i++) {
			smem_tile_to_fragment(sQr, 0, i * 16, rQ[V_TILES + i]);
		}
	}

	{ // Now that we have Q in registers, use the last preload tiles for KV
		usize p = GMEM_PRELOAD - 1;
		if (p < kv_steps) {
			cp_async_gmem_to_smem<WARP_SIZE>(
				threadIdx.x % WARP_SIZE,
				gKVc_full.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx),
				preload_c.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx)
			);
			cp_async_gmem_to_smem<WARP_SIZE>(
				threadIdx.x % WARP_SIZE,
				gKVr_full.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx),
				preload_r.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx)
			);
		}
		cp_async_commit();
	}

	// Start preloading sKV from SMEM to registers
	cp_async_wait<GMEM_PRELOAD - 1>();
	__syncwarp();
	SMatrix<bf16, KV_PER_WARP, V_DIM> sKVc;
	sKVc = preload_c.template tile_m<KV_PER_STEP>(0).template tile_m<KV_PER_WARP>(warp_idx);
	SMatrix<bf16, KV_PER_WARP, ROPE_DIM> sKVr;
	sKVr = preload_r.template tile_m<KV_PER_STEP>(0).template tile_m<KV_PER_WARP>(warp_idx);
	Fragment_16x16<bf16> rKV[QK_TILES];
	X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
		smem_tile_to_fragment(sKVc, 0, i * 16, rKV[i]);
	}
	if constexpr (ROPE_TILES > 0) {
		X17_UNROLL for (usize i = 0; i < ROPE_TILES; i++) {
			smem_tile_to_fragment(sKVr, 0, i * 16, rKV[V_TILES + i]);
		}
	}

	// Sequential loop over KV (causal: each warp stops at its diagonal)
	SoftmaxStats r_top;
	SoftmaxStats r_bot;
	Fragment_16x16<f32> rOut[V_TILES];

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
	usize q_start = blockIdx.x * Q_PER_BLOCK;
	f32 top_n = q_start + (threadIdx.x % WARP_SIZE) / 4 + 1 + 1; // the final `+ 1` is for sink
	f32 bot_n = q_start + (threadIdx.x % WARP_SIZE) / 4 + 9 + 1;
	f32 top_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(top_n);
	f32 bot_score_scale = f32(1.0 / constexpr_sqrt(f64(QK_DIM))) * math::fast::logb(bot_n);

	usize kv_pos = warp_idx * KV_PER_WARP;
	X17_NO_UNROLL for (size_t kv_step = 0; kv_pos <= q_start && kv_step < kv_steps; ++kv_step, kv_pos += KV_PER_STEP) {
		// rScores = Q * K.T
		Fragment_16x16<f32> rScores_f32;
		zero_(rScores_f32);
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			mma_a_bt(rQ[i], rKV[i], rScores_f32); rKV[i].transpose_();
		}
		X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
			mma_a_bt(rQ[i], rKV[i], rScores_f32);
		}

		// Scale scores must happen before masking to avoid -inf * 0 == NaN when score_scale == 0
		scale_top_(rScores_f32, top_score_scale);
		scale_bottom_(rScores_f32, bot_score_scale);

		// Causal mask on the diagonal tile
		if (kv_pos == q_start) {
			causal_mask_diagonal(rScores_f32);
		}

		online_softmax(r_top, r_bot, rScores_f32, rOut);
		Fragment_16x16<bf16> rScores;
		cast(rScores_f32, rScores);

		{ // Get more data from GMEM
			// Wait for the next batch of GMEM -> SMEM preloads to complete
			cp_async_wait<GMEM_PRELOAD - 2>();
			__syncwarp();
			sKVc = preload_c
				.template tile_m<KV_PER_STEP>((kv_step + 1) % GMEM_PRELOAD)
				.template tile_m<KV_PER_WARP>(warp_idx);
			sKVr = preload_r
				.template tile_m<KV_PER_STEP>((kv_step + 1) % GMEM_PRELOAD)
				.template tile_m<KV_PER_WARP>(warp_idx);

			// Preload next KV tiles from GMEM
			{
				usize p = kv_step + GMEM_PRELOAD;
				if (p < kv_steps) {
					cp_async_gmem_to_smem<WARP_SIZE>(
						threadIdx.x % WARP_SIZE,
						gKVc_full.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx),
						preload_c.template tile_m<KV_PER_STEP>(p % GMEM_PRELOAD).template tile_m<KV_PER_WARP>(warp_idx)
					);
					cp_async_gmem_to_smem<WARP_SIZE>(
						threadIdx.x % WARP_SIZE,
						gKVr_full.template tile_m<KV_PER_STEP>(p).template tile_m<KV_PER_WARP>(warp_idx),
						preload_r.template tile_m<KV_PER_STEP>(p % GMEM_PRELOAD).template tile_m<KV_PER_WARP>(warp_idx)
					);
				}
				cp_async_commit();
			}
		}

		// rOut += rScores * V (content tiles only, already transposed)
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			mma_a_bt(rScores, rKV[i], rOut[i]); smem_tile_to_fragment(sKVc, 0, i * 16, rKV[i]);
		}
		if constexpr (ROPE_TILES > 0) {
			X17_UNROLL for (usize i = 0; i < ROPE_TILES; i++) {
				smem_tile_to_fragment(sKVr, 0, i * 16, rKV[V_TILES + i]);
			}
		}
	}
	cp_async_wait<0>();  // drain any outstanding preloads before cross-warp sync
	__syncthreads();     // ensure all warps' async copies have landed before reusing smem

	GMatrixDynSize<bf16, V_DIM> gOut_full{gOut_ptr, q_cnt};
	GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = gOut_full.template tile_m<Q_PER_BLOCK>(blockIdx.x);
	combine_and_store(rOut, r_top, r_bot, sink_score, top_score_scale, bot_score_scale, gate, preload_c._ptr, warp_idx, gOut_block);
}

int main(int argc, char *argv[]) {
	constexpr usize V_DIM = 128;
	constexpr usize ROPE_DIM = 0;
	constexpr usize QK_DIM = V_DIM + ROPE_DIM;
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
		Q_LEN = 32768;
		KV_LEN = 32768;
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
	}
	// Split interleaved [KV_LEN, QK_DIM] into separate content and rope arrays
	std::vector<bf16> kvc_data(KV_LEN * V_DIM);
	std::vector<bf16> kvr_data(KV_LEN * ROPE_DIM);
	for (size_t r = 0; r < KV_LEN; r++) {
		for (size_t c = 0; c < V_DIM; c++) {
			kvc_data[r * V_DIM + c] = kv_data[r * QK_DIM + c];
		}
		if constexpr (ROPE_DIM > 0) {
			#pragma nv_diag_suppress 186
			for (size_t c = 0; c < ROPE_DIM; c++) {
				kvr_data[r * ROPE_DIM + c] = kv_data[r * QK_DIM + V_DIM + c];
			}
			#pragma nv_diag_default 186
		}
	}
	bf16 *kvc_dev, *kvr_dev;
	cudaMalloc(&kvc_dev, kvc_data.size() * sizeof(bf16));
	cudaMemcpy(kvc_dev, kvc_data.data(), kvc_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMalloc(&kvr_dev, kvr_data.size() * sizeof(bf16));
	cudaMemcpy(kvr_dev, kvr_data.data(), kvr_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);

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

	usize smem_size =
		sizeof(bf16) * KV_PER_STEP * GMEM_PRELOAD * QK_DIM;
	//smem_size = std::max(smem_size, usize(70 * 1024));

	cudaFuncSetAttribute(attn_kernel<V_DIM, ROPE_DIM, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

	cudaFuncSetAttribute(attn_kernel<V_DIM, ROPE_DIM, 1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	// Allocate sink+gate buffer: [sink_score, gate]
	f32 sink_host[2] = { -0.3f, 0.5f };
	f32 *sink_dev;
	cudaMalloc(&sink_dev, sizeof(sink_host));
	cudaMemcpy(sink_dev, sink_host, sizeof(sink_host), cudaMemcpyHostToDevice);
	f32 *sink_ptr = use_real_data ? sink_dev : nullptr;

	int WARMUP = use_real_data ? 0 : 50;
	for (int i = 0; i < WARMUP; ++i) {
		attn_kernel<V_DIM, ROPE_DIM, 1><<<Q_LEN / Q_PER_BLOCK, THREADS_PER_BLOCK, smem_size>>>(
			Q_LEN, q_dev,
			KV_LEN, kvc_dev, kvr_dev,
			out_dev,
			sink_ptr
		);
	}

	usize NUM_BLOCKS = Q_LEN / Q_PER_BLOCK;
	int NUM_RUNS = use_real_data ? 1 : 200;
	std::vector<cudaEvent_t> starts(NUM_RUNS), ends(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventRecord(starts[i]);
		attn_kernel<V_DIM, ROPE_DIM, 1><<<NUM_BLOCKS, THREADS_PER_BLOCK, smem_size>>>(
			Q_LEN, q_dev,
			KV_LEN, kvc_dev, kvr_dev,
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
	printf("Kernel time over %d runs: median %.4f ms  min %.4f ms\n", NUM_RUNS, median_ms, min_ms);

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
	bool has_non_finite = false;
	for (size_t i = 0; i < out_data.size(); ++i) {
		float v = float(out_data[i]);
		if (!isfinite(v)) {
			has_non_finite = true;
			break;
		}
	}
	if (has_non_finite) {
		printf("\n*** WARNING: output contains non-finite values (inf or NaN) ***\n");
	}

	return 0;
}
