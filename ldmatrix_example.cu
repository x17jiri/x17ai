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

__global__ void
attn_kernel(bf16 *gQ_ptr, bf16 *gKV_ptr, bf16 *gOut_ptr, usize q_cnt, usize kv_cnt, f32 *debug_scores) {
	extern __shared__ bf16 *smem;
	SMatrix<bf16, KV_PER_WARP * WARPS_PER_BLOCK * GMEM_PRELOAD, QK_DIM> preload {smem};
	usize warp_idx = threadIdx.x / WARP_SIZE;

	// Load Q from GMEM to SMEM. Use the last preload tile
	static_assert(Q_PER_BLOCK <= KV_PER_STEP);
	GMatrixDynSize<bf16, QK_DIM> gQ_full {gQ_ptr, q_cnt};
	GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gQ_block = gQ_full.tile_m<Q_PER_BLOCK>(blockIdx.x);
	SMatrix<bf16, Q_PER_BLOCK, QK_DIM> sQ;
	sQ = preload.tile_m<KV_PER_STEP>(GMEM_PRELOAD - 1).tile_m<Q_PER_BLOCK>(0);
	if (threadIdx.x < 128) {
		cp_async_gmem_to_smem<128>(threadIdx.x, gQ_block, sQ);
	}
	cp_async_commit();

	// Start preloading KVs from GMEM to SMEM
	// Don't use the last preload tile yet because it's used to load Q
	GMatrixDynSize<bf16, QK_DIM> gKV_full {gKV_ptr, kv_cnt};
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
	Fragment_16x16<f32> rOut0, rOut1, rOut2, rOut3, rOut4, rOut5, rOut6, rOut7;
	zero_(rOut0, rOut1, rOut2, rOut3, rOut4, rOut5, rOut6, rOut7);

	// Load Q from SMEM to registers
	cp_async_wait<GMEM_PRELOAD - 1>();
	__syncthreads();
	Fragment_16x16<bf16> rQ0, rQ1, rQ2, rQ3, rQ4, rQ5, rQ6, rQ7, rQ8, rQ9, rQ10, rQ11;
	smem_tile_to_fragment(sQ, 0, 0 * 16, rQ0);
	smem_tile_to_fragment(sQ, 0, 1 * 16, rQ1);
	smem_tile_to_fragment(sQ, 0, 2 * 16, rQ2);
	smem_tile_to_fragment(sQ, 0, 3 * 16, rQ3);
	smem_tile_to_fragment(sQ, 0, 4 * 16, rQ4);
	smem_tile_to_fragment(sQ, 0, 5 * 16, rQ5);
	smem_tile_to_fragment(sQ, 0, 6 * 16, rQ6);
	smem_tile_to_fragment(sQ, 0, 7 * 16, rQ7);
	smem_tile_to_fragment(sQ, 0, 8 * 16, rQ8);
	smem_tile_to_fragment(sQ, 0, 9 * 16, rQ9);
	smem_tile_to_fragment(sQ, 0, 10 * 16, rQ10);
	smem_tile_to_fragment(sQ, 0, 11 * 16, rQ11);

	// Now that we have Q in registers, use the last preload tile for KV
	{
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
	Fragment_16x16<bf16> r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

	smem_tile_to_fragment(sKV, 0, 0*16, r0);
	smem_tile_to_fragment(sKV, 0, 1*16, r1);
	smem_tile_to_fragment(sKV, 0, 2*16, r2);
	smem_tile_to_fragment(sKV, 0, 3*16, r3);

	smem_tile_to_fragment(sKV, 0, 4*16, r4);
	smem_tile_to_fragment(sKV, 0, 5*16, r5);
	smem_tile_to_fragment(sKV, 0, 6*16, r6);
	smem_tile_to_fragment(sKV, 0, 7*16, r7);

	smem_tile_to_fragment(sKV, 0, 8*16, r8);
	smem_tile_to_fragment(sKV, 0, 9*16, r9);
	smem_tile_to_fragment(sKV, 0, 10*16, r10);
	smem_tile_to_fragment(sKV, 0, 11*16, r11);

	// Sequential loop over KV
	X17_NO_UNROLL for (size_t kv_step = 0; kv_step < kv_steps; ++kv_step) {
		// rScores = Q * K.T
		Fragment_16x16<f32> rScores_f32;
		rScores_f32.zero_();

		mma_a_bt(rQ0, r0, rScores_f32); r0.transpose_();
		mma_a_bt(rQ1, r1, rScores_f32); r1.transpose_();
		mma_a_bt(rQ2, r2, rScores_f32); r2.transpose_();
		mma_a_bt(rQ3, r3, rScores_f32); r3.transpose_();

		mma_a_bt(rQ4, r4, rScores_f32); r4.transpose_();
		mma_a_bt(rQ5, r5, rScores_f32); r5.transpose_();
		mma_a_bt(rQ6, r6, rScores_f32); r6.transpose_();
		mma_a_bt(rQ7, r7, rScores_f32); r7.transpose_();

		mma_a_bt(rQ8, r8, rScores_f32);
		mma_a_bt(rQ9, r9, rScores_f32);
		mma_a_bt(rQ10, r10, rScores_f32);
		mma_a_bt(rQ11, r11, rScores_f32);

		Fragment_16x16<bf16> rScores;
		cast(rScores_f32, rScores);

		{
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
		mma_a_bt(rScores, r0, rOut0); smem_tile_to_fragment(sKV, 0, 0*16, r0);
		mma_a_bt(rScores, r1, rOut1); smem_tile_to_fragment(sKV, 0, 1*16, r1);
		mma_a_bt(rScores, r2, rOut2); smem_tile_to_fragment(sKV, 0, 2*16, r2);
		mma_a_bt(rScores, r3, rOut3); smem_tile_to_fragment(sKV, 0, 3*16, r3);

		mma_a_bt(rScores, r4, rOut4); smem_tile_to_fragment(sKV, 0, 4*16, r4);
		mma_a_bt(rScores, r5, rOut5); smem_tile_to_fragment(sKV, 0, 5*16, r5);
		mma_a_bt(rScores, r6, rOut6); smem_tile_to_fragment(sKV, 0, 6*16, r6);
		mma_a_bt(rScores, r7, rOut7); smem_tile_to_fragment(sKV, 0, 7*16, r7);

		smem_tile_to_fragment(sKV, 0, 8*16, r8);
		smem_tile_to_fragment(sKV, 0, 9*16, r9);
		smem_tile_to_fragment(sKV, 0, 10*16, r10);
		smem_tile_to_fragment(sKV, 0, 11*16, r11);
	}

	// Cross-warp reduction: tree reduce 8 warps -> warp 0 in 3 rounds.
	SMatrix<f32, 6 * Q_PER_WARP, V_DIM> sReduce{smem};

	// Helper lambdas to store/load all 8 output tiles at a given m_idx offset
	auto store_all_tiles = [&](usize m_idx) {
		fragment_to_smem_tile(rOut0, sReduce, m_idx, 0*16);
		fragment_to_smem_tile(rOut1, sReduce, m_idx, 1*16);
		fragment_to_smem_tile(rOut2, sReduce, m_idx, 2*16);
		fragment_to_smem_tile(rOut3, sReduce, m_idx, 3*16);
		fragment_to_smem_tile(rOut4, sReduce, m_idx, 4*16);
		fragment_to_smem_tile(rOut5, sReduce, m_idx, 5*16);
		fragment_to_smem_tile(rOut6, sReduce, m_idx, 6*16);
		fragment_to_smem_tile(rOut7, sReduce, m_idx, 7*16);
	};
	auto acc_all_tiles = [&](usize m_idx) {
		acc_smem_tile_to_fragment(sReduce, m_idx, 0*16, rOut0);
		acc_smem_tile_to_fragment(sReduce, m_idx, 1*16, rOut1);
		acc_smem_tile_to_fragment(sReduce, m_idx, 2*16, rOut2);
		acc_smem_tile_to_fragment(sReduce, m_idx, 3*16, rOut3);
		acc_smem_tile_to_fragment(sReduce, m_idx, 4*16, rOut4);
		acc_smem_tile_to_fragment(sReduce, m_idx, 5*16, rOut5);
		acc_smem_tile_to_fragment(sReduce, m_idx, 6*16, rOut6);
		acc_smem_tile_to_fragment(sReduce, m_idx, 7*16, rOut7);
	};

	// Round 1: warps 1,3,5,7 write -> warps 0,2,4,6 read and add
	if (warp_idx % 2 == 1) {
		store_all_tiles((warp_idx / 2) * Q_PER_WARP);
	}
	__syncthreads();
	if (warp_idx % 2 == 0) {
		acc_all_tiles((warp_idx / 2) * Q_PER_WARP);
	}

	// Round 2: warps 2,6 write -> warps 0,4 read and add
	if (warp_idx % 4 == 2) {
		store_all_tiles((warp_idx / 4 + 4) * Q_PER_WARP);
	}
	__syncthreads();
	if (warp_idx % 4 == 0) {
		acc_all_tiles((warp_idx / 4 + 4) * Q_PER_WARP);
	}

	// Round 3: warp 4 writes -> warp 0 reads and adds
	if (warp_idx == 4) {
		store_all_tiles(0);
	}
	__syncthreads();
	// Warp 0 stores final result to GMEM
	if (warp_idx == 0) {
		acc_all_tiles(0);

		GMatrixDynSize<bf16, V_DIM> gOut_full {gOut_ptr, q_cnt};
		GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = gOut_full.tile_m<Q_PER_BLOCK>(blockIdx.x);
		rOut0.store(gOut_block, 0, 0*16);
		rOut1.store(gOut_block, 0, 1*16);
		rOut2.store(gOut_block, 0, 2*16);
		rOut3.store(gOut_block, 0, 3*16);
		rOut4.store(gOut_block, 0, 4*16);
		rOut5.store(gOut_block, 0, 5*16);
		rOut6.store(gOut_block, 0, 6*16);
		rOut7.store(gOut_block, 0, 7*16);
	}
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
	GMatrixDynSize<bf16, QK_DIM> q {q_dev, Q_LEN};

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
	GMatrixDynSize<bf16, QK_DIM> kv {kv_dev, KV_LEN};

	// allocate output: bf16 [Q_LEN, V_DIM]
	std::vector<bf16> out_data(Q_LEN * V_DIM);
	bf16 *out_dev;
	size_t out_size_bytes = out_data.size() * sizeof(bf16);
	cudaMalloc(&out_dev, out_size_bytes);
	GMatrixDynSize<bf16, V_DIM> out {out_dev, Q_LEN};

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
