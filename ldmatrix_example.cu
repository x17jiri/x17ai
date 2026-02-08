#include "utils.cuh"
#include <vector>

constexpr usize QK_DIM = 192;
constexpr usize V_DIM = 128;

constexpr usize Q_PER_BLOCK = 64;
constexpr usize Q_PER_WARP = 8;
constexpr usize GEMM_TILE = 16;
constexpr usize GEMMS_PER_STEP = 2;
constexpr usize KV_PER_STEP = (GEMM_TILE * GEMMS_PER_STEP);
constexpr usize GMEM_PRELOAD = 3;

constexpr usize BLOCK_DIM = Q_PER_BLOCK / Q_PER_WARP * 32;

__global__ void attn_kernel(
	bf16 *pA, bf16 *pB,
	GMatrix<bf16, -1, QK_DIM> const &gQ,
	GMatrix<bf16, -1, QK_DIM> const &gKV,
	GMatrix<bf16, -1, V_DIM> const &gOut
) {
    __shared__ bf16 smem[256 + 128];

	GMatrix<bf16, 16, 16> gA{pA};
	GMatrix<bf16, 8, 16> gB{pB};

	SMatrix<bf16, 16, 16> sA{smem};
	SMatrix<bf16, 8, 16> sB{smem + sA.elems()};

	cp_async<32>(threadIdx.x, gA, sA);
	cp_async<32>(threadIdx.x, gB, sB);
	cp_async_commit();
	cp_async_wait();
	__syncwarp();

	RMatrix<bf16, 16, 16> rA;
	ldmatrix(threadIdx.x, sA.t(), rA);

	RMatrix<bf16, 16, 8, ColumnMajor> rB;
	ldmatrix(threadIdx.x, sB.t(), rB);

	RMatrix<f32, 16, 8, ColumnMajor> rC;
	rC.zero_();

	acc_gemm(rC, rA, rB);

	printf("Thread %d: c00 = %f, c01 = %f, c10 = %f, c11 = %f\n",
		threadIdx.x,
		rC.tiles[0][0].first(), rC.tiles[0][0].second(),
		rC.tiles[1][0].first(), rC.tiles[1][0].second()
	);

/*
	SMatrix<bf16, Q_PER_BLOCK, QK_DIM> sQ{smem};
	SMatrix<bf16, KV_PER_STEP * GMEM_PRELOAD, QK_DIM> sKV{smem + sQ.elems()};

	// Preload Q from GMEM to SMEM
	cp_async<BLOCK_DIM>(
		threadIdx.x,
		gQ.tile_m<Q_PER_BLOCK>(blockIdx.x),
		sQ
	);
	// Sub-matrix with Qs for this warp. Note that this is just pointer arithmetic.
	// The preload doesn't have to be finished yet.
	SMatrix<bf16, Q_PER_WARP, QK_DIM> sQ_warp = sQ.tile_m<Q_PER_WARP>(threadIdx.x / 32);

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
}

int main() {
	std::vector<bf16> A(256);
	for (int i = 0; i < 256; i++) {
		A[i] = bf16(float(i));
	}

	std::vector<bf16> B(128);
	for (int i = 0; i < 128; i++) {
		B[i] = bf16(float(i * 100));
	}

	bf16 *dA, *dB;
	cudaMalloc(&dA, A.size() * sizeof(bf16));
	cudaMalloc(&dB, B.size() * sizeof(bf16));

	cudaMemcpy(dA, A.data(), A.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B.data(), B.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	attn_kernel<<<1, 32>>>(
		dA, dB,
		GMatrix<bf16, -1, QK_DIM>{nullptr, 0},
		GMatrix<bf16, -1, QK_DIM>{nullptr, 0},
		GMatrix<bf16, -1, V_DIM>{nullptr, 0}
	);

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
