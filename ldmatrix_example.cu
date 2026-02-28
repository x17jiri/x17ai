#include "utils2.cuh"
#include <vector>
#include <fstream>
#include <array>
#include "cutlass/util/GPU_Clock.hpp"

constexpr usize QK_DIM = 192;
constexpr usize V_DIM = 128;

constexpr usize Q_PER_WARP = 16;
constexpr usize KV_PER_WARP = 16;
constexpr usize GMEM_PRELOAD = 2;
constexpr usize WARPS_PER_BLOCK = 8;
constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr usize Q_PER_BLOCK = Q_PER_WARP;
constexpr usize KV_PER_STEP = KV_PER_WARP * WARPS_PER_BLOCK;

constexpr usize QK_TILES = QK_DIM / 16;
constexpr usize V_TILES = V_DIM / 16;

/// Online softmax for flash attention.
template<const usize K>
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
	f32 rescale;
	{
		rescale = expf(stats.max - new_max);

		// XOR 1 — exchange rescale factors so each thread knows both rows
		f32 partner_rescale = __shfl_xor_sync(0xffffffff, rescale, 1);
		f32 top_rescale = is_even ? rescale : partner_rescale;
		f32 bot_rescale = is_even ? partner_rescale : rescale;

		for (usize i = 0; i < K; i++) {
			rOut[i].sub[0][0].val0 *= top_rescale;
			rOut[i].sub[0][0].val1 *= top_rescale;
			rOut[i].sub[0][1].val0 *= top_rescale;
			rOut[i].sub[0][1].val1 *= top_rescale;

			rOut[i].sub[1][0].val0 *= bot_rescale;
			rOut[i].sub[1][0].val1 *= bot_rescale;
			rOut[i].sub[1][1].val0 *= bot_rescale;
			rOut[i].sub[1][1].val1 *= bot_rescale;
		}
	}

	// Step 3: Replace scores with exp(score - new_max)
	{
		// Exchange new_max so each thread has both top and bottom row max
		f32 partner_new_max = __shfl_xor_sync(0xffffffff, new_max, 1);
		f32 top_new_max = is_even ? new_max : partner_new_max;
		f32 bot_new_max = is_even ? partner_new_max : new_max;

		rScores.sub[0][0].val0 = expf(rScores.sub[0][0].val0 - top_new_max);
		rScores.sub[0][0].val1 = expf(rScores.sub[0][0].val1 - top_new_max);
		rScores.sub[0][1].val0 = expf(rScores.sub[0][1].val0 - top_new_max);
		rScores.sub[0][1].val1 = expf(rScores.sub[0][1].val1 - top_new_max);

		rScores.sub[1][0].val0 = expf(rScores.sub[1][0].val0 - bot_new_max);
		rScores.sub[1][0].val1 = expf(rScores.sub[1][0].val1 - bot_new_max);
		rScores.sub[1][1].val0 = expf(rScores.sub[1][1].val0 - bot_new_max);
		rScores.sub[1][1].val1 = expf(rScores.sub[1][1].val1 - bot_new_max);
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

template<const usize Q_PER_WARP, const usize M, const usize N, const usize K>
requires(M % Q_PER_WARP == 0 && N == K * 16)
X17_DEVICE void combine_read(
	Fragment_16x16<f32> (&rOut)[K],
	SoftmaxStats &r_stats,
	SMatrix<f32, M, N> sReduce,
	u32 stats_smem,
	usize slot
) {
	SMatrix<f32, Q_PER_WARP, V_DIM> tile = sReduce.tile_m<Q_PER_WARP>(slot);
	SoftmaxStats s_stats;
	s_stats.load_shared(
		stats_smem + sizeof(f32) * (
			slot * 2 * WARP_SIZE
			+ (threadIdx.x % WARP_SIZE) * 2
		)
	);
	f32 new_max = fmaxf(r_stats.max, s_stats.max);
	f32 r_rescale = expf(r_stats.max - new_max);
	f32 s_rescale = expf(s_stats.max - new_max);

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

template<const usize K, const usize OUT_DIM, const usize Q_PER_BLOCK>
requires(Q_PER_BLOCK == 16 && OUT_DIM == K * 16)
X17_DEVICE void combine_and_store(
	Fragment_16x16<f32> (&rOut)[K],
	SoftmaxStats &r_stats,
	u32 smem,
	usize warp_idx,
	GMatrix<bf16, Q_PER_BLOCK, OUT_DIM> gOut_block
) {
	// Cross-warp reduction: tree reduce 8 warps -> warp 0 in 3 rounds.
	static_assert(WARPS_PER_BLOCK == 8, "This reduction code assumes 8 warps per block");
	SMatrix<f32, 6 * Q_PER_WARP, K * 16> sReduce{smem};
	u32 stats_smem = sReduce._ptr + sReduce.bytes();

	// Round 1: warps 1,3,5,7 write to slots 0-3, warps 0,2,4,6 read
	__syncthreads();
	if (warp_idx % 2 == 1) {
		combine_write<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, warp_idx / 2);
	}
	__syncthreads();
	if (warp_idx % 2 == 0) {
		combine_read<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, warp_idx / 2);
	}

	// Round 2: warps 2,6 write to slots 4,5, warps 0,4 read
	if (warp_idx % 4 == 2) {
		combine_write<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 4 + warp_idx / 4);
	}
	__syncthreads();
	if (warp_idx % 4 == 0) {
		combine_read<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 4 + warp_idx / 4);
	}

	// Round 3: warp 4 writes to slot 0, warp 0 reads
	if (warp_idx == 4) {
		combine_write<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 0);
	}
	__syncthreads();
	if (warp_idx == 0) {
		combine_read<Q_PER_WARP>(rOut, r_stats, sReduce, stats_smem, 0);

		// Normalize: divide each row by its running_sum
		f32 partner_sum = __shfl_xor_sync(0xffffffff, r_stats.sum, 1);
		bool is_even = (threadIdx.x % 2) == 0;
		f32 top_inv_sum = 1.0f / (is_even ? r_stats.sum : partner_sum);
		f32 bot_inv_sum = 1.0f / (is_even ? partner_sum : r_stats.sum);

		X17_UNROLL for (usize i = 0; i < K; i++) {
			auto &f = rOut[i];
			f.sub[0][0].val0 *= top_inv_sum; f.sub[0][0].val1 *= top_inv_sum;
			f.sub[0][1].val0 *= top_inv_sum; f.sub[0][1].val1 *= top_inv_sum;
			f.sub[1][0].val0 *= bot_inv_sum; f.sub[1][0].val1 *= bot_inv_sum;
			f.sub[1][1].val0 *= bot_inv_sum; f.sub[1][1].val1 *= bot_inv_sum;
		}

		X17_UNROLL for (usize i = 0; i < K; i++) {
			rOut[i].store(gOut_block, 0, i*16);
		}
	}
}

__global__ void
attn_kernel(
	bf16 *gQ_ptr, bf16 *gKV_ptr, bf16 *gOut_ptr,
	usize q_cnt, usize kv_cnt,
	f32 *debug_scores
) {
	extern __shared__ bf16 *smem;
	SMatrix<bf16, KV_PER_WARP * WARPS_PER_BLOCK * GMEM_PRELOAD, QK_DIM> preload{smem};
	usize warp_idx = threadIdx.x / WARP_SIZE;

	// Load Q from GMEM to SMEM. Use the last preload tile
	static_assert(Q_PER_BLOCK <= KV_PER_STEP);
	GMatrixDynSize<bf16, QK_DIM> gQ_full{gQ_ptr, q_cnt};
	GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gQ_block = gQ_full.tile_m<Q_PER_BLOCK>(blockIdx.x);
	SMatrix<bf16, Q_PER_BLOCK, QK_DIM> sQ;
	sQ = preload.tile_m<KV_PER_STEP>(GMEM_PRELOAD - 1).tile_m<Q_PER_BLOCK>(0);
	if (threadIdx.x < 128) {
		cp_async_gmem_to_smem<128>(threadIdx.x, gQ_block, sQ);
	}
	cp_async_commit();

	// Start preloading KVs from GMEM to SMEM
	// Don't use the last preload tile yet because it's used to load Q
	GMatrixDynSize<bf16, QK_DIM> gKV_full{gKV_ptr, kv_cnt};
	size_t kv_steps = gKV_full.m_rows() / KV_PER_STEP;
	X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD - 1; ++p) {
		if (p < kv_steps) {
			cp_async_gmem_to_smem<WARP_SIZE>(
				threadIdx.x % WARP_SIZE,
				gKV_full.tile_m<KV_PER_STEP>(p).tile_m<KV_PER_WARP>(warp_idx),
				preload.tile_m<KV_PER_STEP>(p).tile_m<KV_PER_WARP>(warp_idx)
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
	X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
		smem_tile_to_fragment(sQ, 0, i * 16, rQ[i]);
	}

	{ // Now that we have Q in registers, use the last preload tile for KV
		usize p = GMEM_PRELOAD - 1;
		if (p < kv_steps) {
			cp_async_gmem_to_smem<WARP_SIZE>(
				threadIdx.x % WARP_SIZE,
				gKV_full.tile_m<KV_PER_STEP>(p).tile_m<KV_PER_WARP>(warp_idx),
				preload.tile_m<KV_PER_STEP>(p).tile_m<KV_PER_WARP>(warp_idx)
			);
		}
		cp_async_commit();
	}

	// Start preloading sKV from SMEM to registers
	cp_async_wait<GMEM_PRELOAD - 1>();
	__syncwarp();
	SMatrix<bf16, KV_PER_WARP, QK_DIM> sKV;
	sKV = preload.tile_m<KV_PER_STEP>(0).tile_m<KV_PER_WARP>(warp_idx);
	Fragment_16x16<bf16> rKV[QK_TILES];
	X17_UNROLL for (usize i = 0; i < QK_TILES; i++) {
		smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
	}

	// Sequential loop over KV
	SoftmaxStats r_stats;
	X17_NO_UNROLL for (size_t kv_step = 0; kv_step < kv_steps; ++kv_step) {
		// rScores = Q * K.T
		Fragment_16x16<f32> rScores_f32;
		rScores_f32.zero_();
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			mma_a_bt(rQ[i], rKV[i], rScores_f32); rKV[i].transpose_();
		}
		X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
			mma_a_bt(rQ[i], rKV[i], rScores_f32);
		}

		// Softmax
		online_softmax(r_stats, rScores_f32, rOut);

		{ // Get more data from GMEM
			// Wait for the next batch of GMEM -> SMEM preloads to complete
			cp_async_wait<GMEM_PRELOAD - 2>();
			__syncwarp();
			sKV = preload
				.tile_m<KV_PER_STEP>((kv_step + 1) % GMEM_PRELOAD)
				.tile_m<KV_PER_WARP>(warp_idx);

			// Preload next KV tile from GMEM
			{
				usize p = kv_step + GMEM_PRELOAD;
				if (p < kv_steps) {
					cp_async_gmem_to_smem<WARP_SIZE>(
						threadIdx.x % WARP_SIZE,
						gKV_full.tile_m<KV_PER_STEP>(p).tile_m<KV_PER_WARP>(warp_idx),
						preload.tile_m<KV_PER_STEP>(p % GMEM_PRELOAD).tile_m<KV_PER_WARP>(warp_idx)
					);
				}
				cp_async_commit();
			}
		}

		// rOut += rScores * V
		Fragment_16x16<bf16> rScores;
		cast(rScores_f32, rScores);
		X17_UNROLL for (usize i = 0; i < V_TILES; i++) {
			mma_a_bt(rScores, rKV[i], rOut[i]); smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
		}
		X17_UNROLL for (usize i = V_TILES; i < QK_TILES; i++) {
			smem_tile_to_fragment(sKV, 0, i * 16, rKV[i]);
		}
	}

	GMatrixDynSize<bf16, V_DIM> gOut_full{gOut_ptr, q_cnt};
	GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = gOut_full.tile_m<Q_PER_BLOCK>(blockIdx.x);
	combine_and_store(rOut, r_stats, preload._ptr, warp_idx, gOut_block);
}

int main() {
	bool use_real_data = true;
	constexpr size_t Q_LEN = 1024;
	constexpr size_t KV_LEN = 1024;

	// allocate q: bf16 [Q_LEN, QK_DIM]
	std::vector<bf16> q_data(Q_LEN * QK_DIM);
	if (use_real_data) {
		std::ifstream in("q.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(q_data.data()),
			static_cast<std::streamsize>(q_data.size() * sizeof(*q_data.data()))
		);
	} else {
		// Initialize with dummy data for testing
		for (size_t i = 0; i < q_data.size(); ++i) {
			q_data[i] = bf16(float(i));
		}
	}
	bf16 *q_dev;
	size_t q_size_bytes = q_data.size() * sizeof(bf16);
	cudaMalloc(&q_dev, q_size_bytes);
	cudaMemcpy(q_dev, q_data.data(), q_size_bytes, cudaMemcpyHostToDevice);
	GMatrixDynSize<bf16, QK_DIM> q{q_dev, Q_LEN};

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(1) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	// allocate kv: bf16 [KV_LEN, QK_DIM]
	std::vector<bf16> kv_data(KV_LEN * QK_DIM);
	if (use_real_data) {
		std::ifstream in("kv.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(kv_data.data()),
			static_cast<std::streamsize>(kv_data.size() * sizeof(*kv_data.data()))
		);
	} else {
		// Initialize with dummy data for testing
		for (size_t i = 0; i < kv_data.size(); ++i) {
			kv_data[i] = bf16(float(i*100));
		}
	}
	bf16 *kv_dev;
	size_t kv_size_bytes = kv_data.size() * sizeof(bf16);
	cudaMalloc(&kv_dev, kv_size_bytes);
	cudaMemcpy(kv_dev, kv_data.data(), kv_size_bytes, cudaMemcpyHostToDevice);
	GMatrixDynSize<bf16, QK_DIM> kv{kv_dev, KV_LEN};

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

	// Allocate debug scores buffer
	f32 *debug_scores_dev;
	cudaMalloc(&debug_scores_dev, 16 * 16 * sizeof(f32));
	cudaMemset(debug_scores_dev, 0, 16 * 16 * sizeof(f32));

	usize smem_size =
		sizeof(bf16) * std::max((Q_PER_BLOCK * QK_DIM), (KV_PER_STEP * GMEM_PRELOAD * QK_DIM));
	smem_size = std::max(smem_size, usize(70 * 1024));

	cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

	cudaFuncSetAttribute(attn_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	GPU_Clock timer;
	timer.start();
	constexpr int NUM_RUNS = 1;
	for (int i = 0; i < NUM_RUNS; ++i) {
		attn_kernel<<<Q_LEN / Q_PER_BLOCK, THREADS_PER_BLOCK, smem_size>>>(
			q._ptr,
			kv._ptr,
			out._ptr,
			q.m_rows(),
			kv.m_rows(),
			debug_scores_dev
		);
	}
	cudaDeviceSynchronize();
	double cute_time = timer.seconds() / NUM_RUNS;
	printf("Average kernel time over %d runs: %f ms\n", NUM_RUNS, cute_time * 1e3);

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

	// Print debug scores (16x16 from block 0, warp 0, kv_step 0)
	{
		std::vector<f32> debug_scores(16 * 16);
		cudaMemcpy(debug_scores.data(), debug_scores_dev, 16 * 16 * sizeof(f32), cudaMemcpyDeviceToHost);
		printf("\nScores (block 0, warp 0, kv_step 0) [16x16]:\n");
		for (size_t r = 0; r < 16; r++) {
			for (size_t c = 0; c < 16; c++) {
				printf("%10.4f ", double(debug_scores[r * 16 + c]));
			}
			printf("\n");
		}
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
