#include "utils.cuh"
#include <vector>
#include <fstream>
#include <array>

constexpr usize QK_DIM = 192;
constexpr usize V_DIM = 128;

constexpr usize Q_PER_BLOCK = 16;
constexpr usize Q_PER_WARP = 16;
constexpr usize GEMM_TILE = 16;
constexpr usize GEMMS_PER_STEP = 2;
constexpr usize KV_PER_STEP = (GEMM_TILE * GEMMS_PER_STEP);
constexpr usize GMEM_PRELOAD = 3;

constexpr usize BLOCK_DIM = Q_PER_BLOCK / Q_PER_WARP * 32;

__global__ void attn_kernel(
	GMatrix<bf16, -1, QK_DIM> const &gQ,
	GMatrix<bf16, -1, QK_DIM> const &gKV,
	GMatrix<bf16, -1, V_DIM> const &gOut
) {
    __shared__ bf16 q_smem[Q_PER_BLOCK * QK_DIM];
	__shared__ bf16 kv_smem[KV_PER_STEP * GMEM_PRELOAD * QK_DIM];

/*	GMatrix<bf16, 16, 16, RowMajor, 16> ggQ{gQ.data._ptr};
	SMatrix<bf16, 16, 16, RowMajor, 16> sQ{q_smem};
	cp_async<32>(
		threadIdx.x,
		ggQ,
		sQ
	);*/

	SMatrix<bf16, Q_PER_BLOCK, QK_DIM> sQ{q_smem};
	SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, QK_DIM> sKV{kv_smem};

	// Preload Q from GMEM to SMEM
	cp_async<BLOCK_DIM>(
		threadIdx.x,
		gQ.tile_m<Q_PER_BLOCK>(blockIdx.x),
		sQ
	);

	// Start preloading KVs from GMEM to SMEM
	X17_UNROLL for (usize preload = 0; preload < GMEM_PRELOAD - 1; ++preload) {
		if (preload * KV_PER_STEP < gKV.m_rows()) {
			cp_async<BLOCK_DIM>(
				threadIdx.x,
				gKV.tile_m<KV_PER_STEP>(preload),
				sKV.tile_m<KV_PER_STEP>(preload)
			);
		}
		cp_async_commit();
	}
	// Wait for the first batch of GMEM -> SMEM preloads to complete
	cp_async_wait<GMEM_PRELOAD - 2>();
	__syncthreads();

	// Sub-matrix with Qs for this warp
	std::array<RMatrix<bf16, 16, 16>, QK_DIM / 16> rQ;
	SMatrix<bf16, Q_PER_WARP, QK_DIM> sQ_warp = sQ.tile_m<Q_PER_WARP>(threadIdx.x / 32);
	X17_UNROLL for (usize i = 0; i < QK_DIM / 16; ++i) {
		ldmatrix(threadIdx.x, sQ_warp.tile_n<16>(i), rQ[i]);
	}

/*
	RMatrix<bf16, 16, 16> r0, r1, r2;
	RMatrix<bf16, 16, 8> u0, u1;
	#include "gemm/Init.h"

	// Sequential loop over KV
	X17_NO_UNROLL for (usize kv_step = 0; kv_step < gKV.m_rows() / KV_PER_STEP; ++kv_step) {
		// Preload next KV tile from GMEM
		usize preload = kv_step + GMEM_PRELOAD - 1;
		if (preload * KV_PER_STEP < gKV.m_rows()) {
			preload %= GMEM_PRELOAD;
			cp_async<BLOCK_DIM>(
				threadIdx.x,
				gKV.tile_m<KV_PER_STEP>(preload),
				sKV.tile_m<KV_PER_STEP>(preload)
			);
		}
		cp_async_commit();

		RMatrix<f32, KV_PER_STEP, Q_PER_WARP> rScores_f32;
		rScores_f32.zero_();
		//*** rScores += sKV x sQ.T
		#include "gemm/rScores.h"

	}
*/
/*
	GMatrix<bf16, 16, 16> gA{pA};
	GMatrix<bf16, 16, 16> gB{pB};

	SMatrix<bf16, 16, 16> sA{smem};
	SMatrix<bf16, 16, 16> sB{smem + sA.elems()};

	cp_async<BLOCK_DIM>(threadIdx.x, gA, sA);
	cp_async<32>(threadIdx.x, gB, sB);
	cp_async_commit();
	cp_async_wait();
	__syncwarp();

	RMatrix<bf16, 16, 16, ColumnMajor> rA;
	ldmatrix(threadIdx.x, sA.t(), rA);

	RMatrix<bf16, 16, 16, RowMajor> rB;
	ldmatrix(threadIdx.x, sB, rB);

	RMatrix<f32, 16, 16, ColumnMajor> rC;
	rC.zero_();

	gemm(rC, rA, rB);

	__syncthreads();
	printf("Thread %d: a00.a = %f, a00.b = %f, a01.a = %f, a01.b = %f, a10.a = %f, a10.b = %f, a11.a = %f, a11.b = %f\n",
		threadIdx.x,
		double(rA.tiles[0][0].first()), double(rA.tiles[0][0].second()),
		double(rA.tiles[0][1].first()), double(rA.tiles[0][1].second()),
		double(rA.tiles[1][0].first()), double(rA.tiles[1][0].second()),
		double(rA.tiles[1][1].first()), double(rA.tiles[1][1].second())
	);

	__syncthreads();
	printf("Thread %d: b00.a = %f, b00.b = %f, b01.a = %f, b01.b = %f, b10.a = %f, b10.b = %f, b11.a = %f, b11.b = %f\n",
		threadIdx.x,
		double(rB.tiles[0][0].first()), double(rB.tiles[0][0].second()),
		double(rB.tiles[0][1].first()), double(rB.tiles[0][1].second()),
		double(rB.tiles[1][0].first()), double(rB.tiles[1][0].second()),
		double(rB.tiles[1][1].first()), double(rB.tiles[1][1].second())
	);

	__syncthreads();
	printf("Thread %d: c00.a = %f, c00.b = %f, c01.a = %f, c01.b = %f, c10.a = %f, c10.b = %f, c11.a = %f, c11.b = %f\n",
		threadIdx.x,
		double(rC.tiles[0][0].first()), double(rC.tiles[0][0].second()),
		double(rC.tiles[0][1].first()), double(rC.tiles[0][1].second()),
		double(rC.tiles[1][0].first()), double(rC.tiles[1][0].second()),
		double(rC.tiles[1][1].first()), double(rC.tiles[1][1].second())
	);
*/
}

int main() {
	constexpr size_t Q_LEN = 4096;
	constexpr size_t KV_LEN = 4096;

	// allocate q: bf16 [Q_LEN, QK_DIM]
	std::vector<bf16> q_data(Q_LEN * QK_DIM);
	{
		std::ifstream in("q.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(q_data.data()),
			static_cast<std::streamsize>(q_data.size() * sizeof(*q_data.data()))
		);
	}
	bf16 *q_dev;
	size_t q_size_bytes = q_data.size() * sizeof(bf16);
	cudaMalloc(&q_dev, q_size_bytes);
	cudaMemcpy(q_dev, q_data.data(), q_size_bytes, cudaMemcpyHostToDevice);
	GMatrix<bf16, -1, QK_DIM, RowMajor> q{GPtr<bf16>(q_dev), Q_LEN};

	// allocate kv: bf16 [KV_LEN, QK_DIM]
	std::vector<bf16> kv_data(KV_LEN * QK_DIM);
	{
		std::ifstream in("kv.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(kv_data.data()),
			static_cast<std::streamsize>(kv_data.size() * sizeof(*kv_data.data()))
		);
	}
	bf16 *kv_dev;
	size_t kv_size_bytes = kv_data.size() * sizeof(bf16);
	cudaMalloc(&kv_dev, kv_size_bytes);
	cudaMemcpy(kv_dev, kv_data.data(), kv_size_bytes, cudaMemcpyHostToDevice);
	GMatrix<bf16, -1, QK_DIM, RowMajor> kv{GPtr<bf16>{kv_dev}, KV_LEN};

	// allocate output: f16 [Q_LEN, V_DIM]
	std::vector<bf16> out_data(Q_LEN * V_DIM);
	bf16 *out_dev;
	size_t out_size_bytes = out_data.size() * sizeof(bf16);
	cudaMalloc(&out_dev, out_size_bytes);
	GMatrix<bf16, -1, V_DIM, RowMajor> out{GPtr<bf16>{out_dev}, Q_LEN};

	attn_kernel<<<1, 32>>>(q, kv, out);

/*    attn_kernel<<<1, 32, 0>>>(
		GMatrix<bf16, -1, QK_DIM>{nullptr, 0},
		GMatrix<bf16, -1, QK_DIM>{nullptr, 0},
		GMatrix<bf16, -1, V_DIM>{nullptr, 0}
	);*/

	// Wait for kernel to complete
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
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
