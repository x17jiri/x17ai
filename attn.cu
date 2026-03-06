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
constexpr usize Q_PER_BLOCK = Q_PER_WARP;
constexpr usize KV_PER_STEP = KV_PER_WARP * WARPS_PER_BLOCK;

X17_DEVICE void causal_mask_diagonal(Fragment_16x16<f32> &rScores) {
	usize tid = threadIdx.x % WARP_SIZE;
	usize q = tid / 4;           // 0..7
	usize k = 2 * (tid % 4);    // 0,2,4,6
	constexpr f32 NEG_INF = -INFINITY;

	rScores.sub[0][1].val0 = NEG_INF;
	rScores.sub[0][1].val1 = NEG_INF;

	if (k > q) {
		rScores.sub[0][0].val0 = NEG_INF;
		rScores.sub[1][1].val0 = NEG_INF;
	}
	if (k + 1 > q) {
		rScores.sub[0][0].val1 = NEG_INF;
		rScores.sub[1][1].val1 = NEG_INF;
	}
}

/// Online softmax for flash attention.
template<const f64 ATN_SCALE, const usize K>
X17_DEVICE void online_softmax(
	SoftmaxStats &stats,
	Fragment_16x16<f32> &rScores,
	Fragment_16x16<f32> (&rOut)[K]
) {
	bool is_even = threadIdx.x % 2 == 0;

	// Step 1: `max` of the owned row
	f32 new_max;
	{
		// Each thread computes local max for its 4 top-row and 4 bottom-row values
		f32 top_max = fmaxf(
			fmaxf(rScores.sub[0][0].val0, rScores.sub[0][0].val1),
			fmaxf(rScores.sub[0][1].val0, rScores.sub[0][1].val1)
		);
		f32 bot_max = fmaxf(
			fmaxf(rScores.sub[1][0].val0, rScores.sub[1][0].val1),
			fmaxf(rScores.sub[1][1].val0, rScores.sub[1][1].val1)
		);

		// XOR 1
		// — thread 0 combines its top with thread 1's top → 8 values for row 0
		// - thread 1 combines its bottom with thread 0's bottom → 8 values for row 8
		new_max = is_even ? top_max : bot_max;
		f32 send_max = is_even ? bot_max : top_max;
		f32 recv_max = __shfl_xor_sync(0xffffffff, send_max, 1);
		new_max = fmaxf(new_max, recv_max);

		// XOR 2
		// — thread 0 combines with thread 2 → complete row (all 16 values)
		new_max = fmaxf(new_max, stats.max);
		recv_max = __shfl_xor_sync(0xffffffff, new_max, 2);
		new_max = fmaxf(new_max, recv_max);
	}

	// Step 2: Compute rescale factor and rescale output fragments
	// stats.max and new_max are both UNSCALED (raw dot products)
	f32 rescale;
	{
		rescale = math::fast::exp<ATN_SCALE>(stats.max - new_max);

		// XOR 1 — exchange rescale factors so each thread knows both rows
		f32 partner_rescale = __shfl_xor_sync(0xffffffff, rescale, 1);
		f32 top_rescale = is_even ? rescale : partner_rescale;
		f32 bot_rescale = is_even ? partner_rescale : rescale;

		for (usize i = 0; i < K; i++) {
			rOut[i].scale_(top_rescale, bot_rescale);
		}
	}

	// Step 3: Replace scores with math::fast::exp<ATN_SCALE>(score - max)
	// Keeping max unscaled ensures score==max gives exactly 0 after subtraction,
	// so exp(0) = 1 regardless of fast-math rounding.
	{
		// Exchange new_max so each thread has both top and bottom row max
		f32 partner_new_max = __shfl_xor_sync(0xffffffff, new_max, 1);
		f32 top_new_max = is_even ? new_max : partner_new_max;
		f32 bot_new_max = is_even ? partner_new_max : new_max;

		rScores.sub[0][0].val0 = math::fast::exp<ATN_SCALE>(rScores.sub[0][0].val0 - top_new_max);
		rScores.sub[0][0].val1 = math::fast::exp<ATN_SCALE>(rScores.sub[0][0].val1 - top_new_max);
		rScores.sub[0][1].val0 = math::fast::exp<ATN_SCALE>(rScores.sub[0][1].val0 - top_new_max);
		rScores.sub[0][1].val1 = math::fast::exp<ATN_SCALE>(rScores.sub[0][1].val1 - top_new_max);

		rScores.sub[1][0].val0 = math::fast::exp<ATN_SCALE>(rScores.sub[1][0].val0 - bot_new_max);
		rScores.sub[1][0].val1 = math::fast::exp<ATN_SCALE>(rScores.sub[1][0].val1 - bot_new_max);
		rScores.sub[1][1].val0 = math::fast::exp<ATN_SCALE>(rScores.sub[1][1].val0 - bot_new_max);
		rScores.sub[1][1].val1 = math::fast::exp<ATN_SCALE>(rScores.sub[1][1].val1 - bot_new_max);
	}

	// Step 4: `sum` of the owned row
	f32 sum_addition;
	{
		f32 top_sum = (
			(rScores.sub[0][0].val0 + rScores.sub[0][0].val1)
			+ (rScores.sub[0][1].val0 + rScores.sub[0][1].val1)
		);
		f32 bot_sum = (
			(rScores.sub[1][0].val0 + rScores.sub[1][0].val1)
			+ (rScores.sub[1][1].val0 + rScores.sub[1][1].val1)
		);

		// XOR 1
		sum_addition = is_even ? top_sum : bot_sum;
		f32 send_sum = is_even ? bot_sum : top_sum;
		f32 recv_sum = __shfl_xor_sync(0xffffffff, send_sum, 1);
		sum_addition += recv_sum;

		// XOR 2
		recv_sum = __shfl_xor_sync(0xffffffff, sum_addition, 2);
		sum_addition += recv_sum;
	}

	// Update running stats
	stats.max = new_max;
	stats.sum = stats.sum * rescale + sum_addition;
}

template<const usize Q_PER_WARP, const usize M, const usize N, const usize K>
requires(M % Q_PER_WARP == 0 && N == K * 16)
X17_DEVICE void combine_write(
	Fragment_16x16<f32> (&rOut)[K],
	SoftmaxStats r_stats,
	SMatrix<f32, M, N> sReduce,
	u32 stats_smem,
	usize slot
) {
	SMatrix<f32, Q_PER_WARP, N> tile = sReduce.tile_m<Q_PER_WARP>(slot);
	fragments_to_smem(rOut, tile);
	r_stats.store_shared(
		stats_smem + sizeof(f32) * (
			slot * 2 * WARP_SIZE
			+ (threadIdx.x % WARP_SIZE) * 2
		)
	);
}

template<const f64 ATN_SCALE, const usize Q_PER_WARP, const usize M, const usize N, const usize K>
requires(M % Q_PER_WARP == 0 && N == K * 16)
X17_DEVICE void combine_read(
	Fragment_16x16<f32> (&rOut)[K],
	SoftmaxStats &r_stats,
	SMatrix<f32, M, N> sReduce,
	u32 stats_smem,
	usize slot
) {
	SMatrix<f32, Q_PER_WARP, N> tile = sReduce.tile_m<Q_PER_WARP>(slot);
	SoftmaxStats s_stats;
	s_stats.load_shared(
		stats_smem + sizeof(f32) * (
			slot * 2 * WARP_SIZE
			+ (threadIdx.x % WARP_SIZE) * 2
		)
	);
	f32 r_rescale = r_stats.max < s_stats.max ? math::fast::exp<ATN_SCALE>(r_stats.max - s_stats.max) : 1.0f;
	f32 s_rescale = s_stats.max < r_stats.max ? math::fast::exp<ATN_SCALE>(s_stats.max - r_stats.max) : 1.0f;
	f32 new_max = fmaxf(r_stats.max, s_stats.max);

	bool is_even = (threadIdx.x % 2) == 0;
	f32 x_rescale = __shfl_xor_sync(0xffffffff, r_rescale, 1);
	f32 r_top_rescale = is_even ? r_rescale : x_rescale;
	f32 r_bot_rescale = is_even ? x_rescale : r_rescale;
	x_rescale = __shfl_xor_sync(0xffffffff, s_rescale, 1);
	f32 s_top_rescale = is_even ? s_rescale : x_rescale;
	f32 s_bot_rescale = is_even ? x_rescale : s_rescale;

	rescale_acc(
		rOut, r_top_rescale, r_bot_rescale,
		tile, s_top_rescale, s_bot_rescale
	);

	r_stats.max = new_max;
	r_stats.sum = r_stats.sum * r_rescale + s_stats.sum * s_rescale;
}

template<const f64 ATN_SCALE, const usize K, const usize OUT_DIM, const usize Q_PER_BLOCK>
requires(Q_PER_BLOCK == 16 && OUT_DIM == K * 16)
X17_DEVICE void combine_and_store(
	Fragment_16x16<f32> (&rOut)[K],
	SoftmaxStats &r_stats,
	u32 smem,
	usize warp_idx,
	GMatrix<bf16, Q_PER_BLOCK, OUT_DIM> gOut_block
) {
	// Cross-warp reduction: tree reduce WARPS_PER_BLOCK warps -> warp 0.
	// Each round halves the active warp count. Between a read and the next
	// round's write there is no __syncthreads(), so each round must use
	// fresh slots to avoid cross-warp races.
	//   8 warps: 3 rounds, 4+2 = 6 slots  (round 3 reuses slot 0)
	//   4 warps: 2 rounds, 2+1 = 3 slots
	static_assert(WARPS_PER_BLOCK == 4 || WARPS_PER_BLOCK == 8);
	constexpr usize NUM_SLOTS = (WARPS_PER_BLOCK == 8) ? 6 : 3;
	SMatrix<f32, NUM_SLOTS * Q_PER_WARP, K * 16> sReduce{smem};
	u32 stats_smem = sReduce._ptr + sReduce.bytes();

	// Round 1: odd warps write, even warps read
	__syncthreads();
	if (warp_idx % 2 == 1) {
		combine_write<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, warp_idx / 2);
	}
	__syncthreads();
	if (warp_idx % 2 == 0) {
		combine_read<ATN_SCALE, Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, warp_idx / 2);
	}

	if constexpr (WARPS_PER_BLOCK == 8) {
		// Round 2: warps 2,6 write to slots 4,5; warps 0,4 read
		if (warp_idx % 4 == 2) {
			combine_write<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 4 + warp_idx / 4);
		}
		__syncthreads();
		if (warp_idx % 4 == 0) {
			combine_read<ATN_SCALE, Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 4 + warp_idx / 4);
		}

		// Round 3: warp 4 writes to slot 0, warp 0 reads
		if (warp_idx == 4) {
			combine_write<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 0);
		}
		__syncthreads();
		if (warp_idx == 0) {
			combine_read<ATN_SCALE, Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 0);
		}
	} else {
		// Round 2: warp 2 writes to slot 2, warp 0 reads
		if (warp_idx == 2) {
			combine_write<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 2);
		}
		__syncthreads();
		if (warp_idx == 0) {
			combine_read<ATN_SCALE, Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 2);
		}
	}

	if (warp_idx == 0) {
		// Normalize: divide each row by its running_sum
		f32 partner_sum = __shfl_xor_sync(0xffffffff, r_stats.sum, 1);
		bool is_even = (threadIdx.x % 2) == 0;
		f32 my_sum = r_stats.sum;
		f32 top_sum = is_even ? my_sum : partner_sum;
		f32 bot_sum = is_even ? partner_sum : my_sum;
		f32 top_inv_sum = math::fast::rcp(top_sum);
		f32 bot_inv_sum = math::fast::rcp(bot_sum);
		X17_UNROLL for (usize i = 0; i < K; i++) {
			rOut[i].scale_(top_inv_sum, bot_inv_sum);
		}

		X17_UNROLL for (usize i = 0; i < K; i++) {
			rOut[i].store(gOut_block, 0, i*16);
		}
	}
}

template<usize V_DIM, usize ROPE_DIM, usize HEAD_SIZE>
__global__ void
attn_kernel(
	usize q_cnt, bf16 *gQ_ptr,
	usize kv_cnt, bf16 *gKVc_ptr, bf16 *gKVr_ptr,
	bf16 *gOut_ptr
) {
	constexpr usize QK_DIM = V_DIM + ROPE_DIM;
	constexpr usize V_TILES = V_DIM / 16;
	constexpr usize ROPE_TILES = ROPE_DIM / 16;
	constexpr usize QK_TILES = V_TILES + ROPE_TILES;

	// Attention scaling factor: 1/sqrt(QK_DIM)
	// math::fast::exp<ATN_SCALE> folds in the log2(e) conversion internally
	constexpr f64 ATN_SCALE = 1.0 / constexpr_sqrt(f64(QK_DIM));

	constexpr usize Q_STRIDE = QK_DIM * HEAD_SIZE;
	constexpr usize KVC_STRIDE = V_DIM * HEAD_SIZE;
	constexpr usize KVR_STRIDE = ROPE_DIM * HEAD_SIZE;

	extern __shared__ bf16 *smem;
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

	// Prepare outputs
	Fragment_16x16<f32> rOut[V_TILES];
	zero_(rOut);

	// Load Q from SMEM to registers
	cp_async_wait<GMEM_PRELOAD - 1>();
	__syncthreads();
	Fragment_16x16<bf16> rQ[QK_TILES];
	X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
		smem_tile_to_fragment(sQc, 0, i * 16, rQ[i]);
	}
	X17_UNROLL for (usize i = 0; i < ROPE_TILES; i++) {
		smem_tile_to_fragment(sQr, 0, i * 16, rQ[V_TILES + i]);
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
	X17_UNROLL for (usize i = 0; i < ROPE_TILES; i++) {
		smem_tile_to_fragment(sKVr, 0, i * 16, rKV[V_TILES + i]);
	}

	// Sequential loop over KV (causal: each warp stops at its diagonal)
	SoftmaxStats r_stats;
	usize q_start = blockIdx.x * Q_PER_BLOCK;
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

		// Causal mask on the diagonal tile
		if (kv_pos == q_start) {
			causal_mask_diagonal(rScores_f32);
		}

		// Softmax — scores are unscaled; ATN_SCALE is folded into math::fast::exp
		online_softmax<ATN_SCALE>(r_stats, rScores_f32, rOut);

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
		Fragment_16x16<bf16> rScores;
		cast(rScores_f32, rScores);
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			mma_a_bt(rScores, rKV[i], rOut[i]); smem_tile_to_fragment(sKVc, 0, i * 16, rKV[i]);
		}
		X17_UNROLL for (usize i = 0; i < ROPE_TILES; i++) {
			smem_tile_to_fragment(sKVr, 0, i * 16, rKV[V_TILES + i]);
		}
	}
	cp_async_wait<0>();  // drain any outstanding preloads before cross-warp sync

	GMatrixDynSize<bf16, V_DIM> gOut_full{gOut_ptr, q_cnt};
	GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = gOut_full.template tile_m<Q_PER_BLOCK>(blockIdx.x);
	combine_and_store<ATN_SCALE>(rOut, r_stats, preload_c._ptr, warp_idx, gOut_block);
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

	// allocate q: bf16 [Q_LEN, QK_DIM]
	std::vector<bf16> q_data(Q_LEN * QK_DIM);
	if (use_real_data) {
		std::ifstream in("q.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(q_data.data()),
			static_cast<std::streamsize>(q_data.size() * sizeof(*q_data.data()))
		);
	} else {
		for (size_t i = 0; i < q_data.size(); ++i) {
			q_data[i] = bf16(float(i));
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
		std::ifstream in("kv.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(kv_data.data()),
			static_cast<std::streamsize>(kv_data.size() * sizeof(*kv_data.data()))
		);
	} else {
		for (size_t i = 0; i < kv_data.size(); ++i) {
			kv_data[i] = bf16(float(i*100));
		}
	}
	// Split interleaved [KV_LEN, QK_DIM] into separate content and rope arrays
	std::vector<bf16> kvc_data(KV_LEN * V_DIM);
	std::vector<bf16> kvr_data(KV_LEN * ROPE_DIM);
	for (size_t r = 0; r < KV_LEN; r++) {
		for (size_t c = 0; c < V_DIM; c++)
			kvc_data[r * V_DIM + c] = kv_data[r * QK_DIM + c];
		for (size_t c = 0; c < ROPE_DIM; c++)
			kvr_data[r * ROPE_DIM + c] = kv_data[r * QK_DIM + V_DIM + c];
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

	int WARMUP = use_real_data ? 0 : 50;
	for (int i = 0; i < WARMUP; ++i) {
		attn_kernel<V_DIM, ROPE_DIM, 1><<<Q_LEN / Q_PER_BLOCK, THREADS_PER_BLOCK, smem_size>>>(
			Q_LEN, q_dev,
			KV_LEN, kvc_dev, kvr_dev,
			out_dev
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
			out_dev
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
		std::ofstream out_file("out_cpu.bin", std::ios::binary);
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

	// Print first 4 rows, first 8 cols
	printf("\nFirst 4 rows, first 8 cols:\n");
	for (size_t r = 0; r < 4; r++) {
		for (size_t c = 0; c < 8; c++) {
			printf("%12.6e ", double(float(out_data[r * V_DIM + c])));
		}
		printf("\n");
	}

	// Print last 4 rows, last 8 cols
	printf("\nLast 4 rows, last 8 cols:\n");
	for (size_t r = Q_LEN - 4; r < Q_LEN; r++) {
		for (size_t c = V_DIM - 8; c < V_DIM; c++) {
			printf("%12.6e ", double(float(out_data[r * V_DIM + c])));
		}
		printf("\n");
	}

	return 0;
}
