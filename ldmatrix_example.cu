#include "utils.cuh"

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
	GMatrix<bf16, -1, QK_DIM> const &gQ,
	GMatrix<bf16, -1, QK_DIM> const &gKV,
	GMatrix<bf16, -1, V_DIM> const &gOut
) {
    extern __shared__ bf16 smem[];

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
/*
    // Allocate host memory
    half* h_matrix = (half*)malloc(MATRIX_SIZE * sizeof(half));

    // Initialize with sequential values: 1.0, 2.0, 3.0, ...
    printf("Input matrix (row-major 8x8):\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        h_matrix[i] = __float2half((float)(i + 1));
        printf("%5.1f", __half2float(h_matrix[i]));
        if ((i + 1) % COLS == 0) printf("\n");
    }
    printf("\n");

    // Allocate device memory
    half* d_matrix;
    cudaMalloc(&d_matrix, MATRIX_SIZE * sizeof(half));

    // Copy to device
    cudaMemcpy(d_matrix, h_matrix, MATRIX_SIZE * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel with 1 warp (32 threads)
    constexpr usize SMEM_SIZE = MATRIX_SIZE * sizeof(u16);
    attn_kernel<<<1, 32, SMEM_SIZE>>>(d_matrix);
	*/
    attn_kernel<<<1, 32, 0>>>(
		GMatrix<bf16, -1, QK_DIM>{nullptr, 0},
		GMatrix<bf16, -1, QK_DIM>{nullptr, 0},
		GMatrix<bf16, -1, V_DIM>{nullptr, 0}
	);
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
