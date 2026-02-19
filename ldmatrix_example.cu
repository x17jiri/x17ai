#include "utils2.cuh"
#include <vector>
#include <fstream>
#include <array>
#include "cutlass/util/GPU_Clock.hpp"

constexpr usize QK_DIM = 192;
constexpr usize V_DIM = 128;

constexpr usize WARPS_PER_BLOCK = 4;
constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr usize Q_PER_WARP = 16;
constexpr usize Q_PER_BLOCK = Q_PER_WARP * WARPS_PER_BLOCK;
constexpr usize KV_PER_STEP = 16;
constexpr usize GMEM_PRELOAD = 8;

__global__ void attn_kernel(
	bf16 *gQ_ptr,
	bf16 *gKV_ptr,
	bf16 *gOut_ptr,
	usize q_cnt,
	usize kv_cnt
) {
    extern __shared__ bf16 *smem;

	bf16 *q_smem = smem; // Q_PER_BLOCK * QK_DIM
	bf16 *kv_smem = smem; // KV_PER_STEP * GMEM_PRELOAD * QK_DIM

	// Load Q from GMEM to SMEM
	GMatrixDynSize<bf16, QK_DIM> gQ_full{gQ_ptr, q_cnt};
	GMatrix<bf16, Q_PER_BLOCK, QK_DIM> gQ_block = gQ_full.tile_m<Q_PER_BLOCK>(blockIdx.x);
	SMatrix<bf16, Q_PER_BLOCK, QK_DIM> sQ_block{q_smem};

	sQ_block.cp_async_from<THREADS_PER_BLOCK>(gQ_block);

	SMatrix<bf16, Q_PER_WARP, QK_DIM> sQ_warp = sQ_block.tile_m<Q_PER_WARP>(threadIdx.x / WARP_SIZE);

	cp_async_commit();
	cp_async_wait<0>();
	__syncthreads();

	RMatrix<bf16, 16, QK_DIM> rQ;
	sQ_warp.load_to_registers(rQ);

	__syncthreads();

	// Start preloading KVs from GMEM to SMEM
	GMatrixDynSize<bf16, QK_DIM> gKV_full{gKV_ptr, kv_cnt};
	SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, QK_DIM> sKV_preload{kv_smem};

	X17_UNROLL for (usize preload = 0; preload < GMEM_PRELOAD - 1; ++preload) {
		if (preload * KV_PER_STEP < gKV_full.m_rows()) {
			sKV_preload.tile_m<KV_PER_STEP>(preload).cp_async_from<THREADS_PER_BLOCK>(
				gKV_full.tile_m<KV_PER_STEP>(preload)
			);
		}
		cp_async_commit();
	}
	// Wait for the first batch of GMEM -> SMEM preloads to complete
	cp_async_wait<GMEM_PRELOAD - 2>();
	__syncthreads();

	// Prepare outputs
	RMatrix<f32, Q_PER_WARP, V_DIM> rOut;
	rOut.zero_();

	// Start preloading sKV to registers
	SMatrix<bf16, KV_PER_STEP, QK_DIM> sKV = sKV_preload.tile_m<KV_PER_STEP>(0);
	Fragment_16x16<bf16> r0, r1, r2;
	sKV.load_tile_to_fragment(0, 0*16, r0);
	sKV.load_tile_to_fragment(0, 1*16, r1);
	sKV.load_tile_to_fragment(0, 2*16, r2);

	// Sequential loop over KV
	size_t kv_steps = gKV_full.m_rows() / KV_PER_STEP;
	X17_NO_UNROLL for (size_t kv_step = 0; kv_step < kv_steps; ++kv_step) {
		// rScores = Q * K.T
		Fragment_16x16<f32> rScores_f32;
		rScores_f32.zero_();
		mma_a_bt(rQ.tiles[0][0], r0, rScores_f32);
		sKV.load_tile_to_fragment(0, 3*16, r0);
		mma_a_bt(rQ.tiles[0][1], r1, rScores_f32);
		sKV.load_tile_to_fragment(0, 4*16, r1);
		mma_a_bt(rQ.tiles[0][2], r2, rScores_f32);
		sKV.load_tile_to_fragment(0, 5*16, r2);
		mma_a_bt(rQ.tiles[0][3], r0, rScores_f32);
		sKV.load_tile_to_fragment(0, 6*16, r0);
		mma_a_bt(rQ.tiles[0][4], r1, rScores_f32);
		sKV.load_tile_to_fragment(0, 7*16, r1);
		mma_a_bt(rQ.tiles[0][5], r2, rScores_f32);
		sKV.load_tile_to_fragment(0, 8*16, r2);
		mma_a_bt(rQ.tiles[0][6], r0, rScores_f32);
		sKV.load_tile_to_fragment(0, 9*16, r0);
		mma_a_bt(rQ.tiles[0][7], r1, rScores_f32);
		sKV.load_tile_to_fragment(0, 10*16, r1);
		mma_a_bt(rQ.tiles[0][8], r2, rScores_f32);
		sKV.load_tile_to_fragment(0, 11*16, r2);
		mma_a_bt(rQ.tiles[0][9], r0, rScores_f32);
		//ldmatrix_t(sKV.tile_n<16>(2), r0);
		mma_a_bt(rQ.tiles[0][10], r1, rScores_f32);
		//ldmatrix_t(sKV.tile_n<16>(0), r1);
		mma_a_bt(rQ.tiles[0][11], r2, rScores_f32);
		//ldmatrix_t(sKV.tile_n<16>(1), r2);

		// rOut += rScores * V
		Fragment_16x16<bf16> rScores;
		cast(rScores_f32, rScores);
		mma_a_bt(rScores, r1, rOut.tiles[0][0]);
		//ldmatrix_t(sKV.tile_n<16>(3), r1);
		mma_a_bt(rScores, r2, rOut.tiles[0][1]);
		//ldmatrix_t(sKV.tile_n<16>(4), r2);
		mma_a_bt(rScores, r0, rOut.tiles[0][2]);
		//ldmatrix_t(sKV.tile_n<16>(5), r0);
		mma_a_bt(rScores, r1, rOut.tiles[0][3]);
		//ldmatrix_t(sKV.tile_n<16>(6), r1);
		mma_a_bt(rScores, r2, rOut.tiles[0][4]);
		//ldmatrix_t(sKV.tile_n<16>(7), r2);
		mma_a_bt(rScores, r0, rOut.tiles[0][5]);
		{
			//__syncthreads();
			// Preload next KV tile from GMEM
			if ((kv_step + GMEM_PRELOAD - 1) * KV_PER_STEP < gKV_full.m_rows()) {
				sKV_preload.tile_m<KV_PER_STEP>(
					(kv_step + GMEM_PRELOAD - 1) % GMEM_PRELOAD
				).cp_async_from<THREADS_PER_BLOCK>(
					gKV_full.tile_m<KV_PER_STEP>(kv_step + GMEM_PRELOAD - 1)
				);
			}
			cp_async_commit();
			// Wait for the next batch of GMEM -> SMEM preloads to complete
			cp_async_wait<GMEM_PRELOAD - 2>();
			__syncthreads();
			sKV = sKV_preload.tile_m<KV_PER_STEP>((kv_step + 1) % GMEM_PRELOAD);
		}
		//ldmatrix(sKV.tile_n<16>(0), r0);
		mma_a_bt(rScores, r1, rOut.tiles[0][6]);
		//ldmatrix(sKV.tile_n<16>(1), r1);
		mma_a_bt(rScores, r2, rOut.tiles[0][7]);
		//ldmatrix(sKV.tile_n<16>(2), r2);
	}

	GMatrixDynSize<bf16, V_DIM> gOut_full{gOut_ptr, q_cnt};
	GMatrix<bf16, Q_PER_BLOCK, V_DIM> gOut_block = gOut_full.tile_m<Q_PER_BLOCK>(blockIdx.x);
	GMatrix<bf16, Q_PER_WARP, V_DIM> gOut_warp = gOut_block.tile_m<Q_PER_WARP>(threadIdx.x / WARP_SIZE);
	//stmatrix(rOut, gOut_warp);

	/*if (threadIdx.x < 32) {
		auto &t = rOut.tiles[0][0];
		printf("Thread %d: a00.a = %e, a00.b = %e, a01.a = %e, a01.b = %e, a10.a = %e, a10.b = %e, a11.a = %e, a11.b = %e\n",
			threadIdx.x,
			double(t.sub[0][0].first()), double(t.sub[0][0].second()),
			double(t.sub[0][1].first()), double(t.sub[0][1].second()),
			double(t.sub[1][0].first()), double(t.sub[1][0].second()),
			double(t.sub[1][1].first()), double(t.sub[1][1].second())
		);
	}*/
}

int main() {
	bool use_real_data = true;
	constexpr size_t Q_LEN = 4*4096;
	constexpr size_t KV_LEN = 4*4096;

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

	/*cudaFuncSetAttribute(
		attn_kernel,
		cudaFuncAttributePreferredSharedMemoryCarveout, 100);*/

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("(3) CUDA Error: %s\n", cudaGetErrorString(err));
    }

	usize smem_size = sizeof(bf16) * std::max(
		(Q_PER_BLOCK * QK_DIM),
		(KV_PER_STEP * GMEM_PRELOAD * QK_DIM)
	);

	GPU_Clock timer;
	timer.start();
	for (int i = 0; i < 100; ++i) {
		attn_kernel<<<Q_LEN/Q_PER_BLOCK, THREADS_PER_BLOCK, smem_size>>>(
			q._ptr, kv._ptr, out._ptr,
			q.m_rows(), kv.m_rows()
		);
	}
	double cute_time = timer.seconds() / 100;
	printf("Average kernel time over 100 runs: %f ms\n", cute_time * 1e3);

	// write output to file
	{
		std::ofstream out_file("out_cpu.bin", std::ios::binary);
		cudaMemcpy(out_data.data(), out_dev, out_size_bytes, cudaMemcpyDeviceToHost);
		out_file.write(
			reinterpret_cast<char*>(out_data.data()),
			static_cast<std::streamsize>(out_data.size() * sizeof(*out_data.data()))
		);
	}

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("(4) CUDA Error: %s\n", cudaGetErrorString(err));
    }

	// Wait for kernel to complete
    cudaDeviceSynchronize();

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

/*
    // Cleanup
    cudaFree(d_matrix);
    free(h_matrix);
*/
    return 0;
}
