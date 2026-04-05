#include "utils2.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include "cutlass/util/GPU_Clock.hpp"

constexpr usize M_WARPS = 2;
constexpr usize N_WARPS = 2;
constexpr usize M_PER_WARP = 32;
constexpr usize N_PER_WARP = 64;
constexpr usize M_PER_BLOCK = M_WARPS * M_PER_WARP;
constexpr usize N_PER_BLOCK = N_WARPS * N_PER_WARP;
constexpr usize WARPS_PER_BLOCK = M_WARPS * N_WARPS;
constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr usize K_STEP = 64;
constexpr usize GMEM_PRELOAD = 4;
constexpr usize M_TILES = M_PER_WARP / 16;
constexpr usize N_TILES = N_PER_WARP / 16;

static_assert(WARPS_PER_BLOCK == 4);
static_assert(K_STEP % 16 == 0);

template<usize K>
X17_DEVICE void cp_async_ab(
	GMatrix<bf16, M_PER_BLOCK, K> gA_block,
	GMatrix<bf16, N_PER_BLOCK, K> gB_block,
	SMatrix<bf16, M_PER_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload,
	SMatrix<bf16, N_PER_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload,
	usize p,
	usize k_end
) {
	if (p < k_end) {
		SMatrix<bf16, M_PER_BLOCK, K_STEP> sA_tile = tile_m<M_PER_BLOCK>(sA_preload, p % GMEM_PRELOAD);
		SMatrix<bf16, N_PER_BLOCK, K_STEP> sB_tile = tile_m<N_PER_BLOCK>(sB_preload, p % GMEM_PRELOAD);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
			threadIdx.x,
			gA_block.template slice_n<K_STEP>(p * K_STEP),
			sA_tile
		);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
			threadIdx.x,
			gB_block.template slice_n<K_STEP>(p * K_STEP),
			sB_tile
		);
	}
}

// GEMM: C[M,N] = A[M,K] * B^T[N,K]
// A is [M, K] row-major
// B is [N, K] row-major (transposed)
// C is [M, N] row-major
template<usize K>
__global__ void gemm_kernel(
	usize M, usize N,
	bf16 *A,
	bf16 *B,
	bf16 *C
) {
	static_assert(K % K_STEP == 0);
	constexpr usize K_ITERS = K / K_STEP;
	constexpr usize K_TILES = K_STEP / 16;

	usize tid = threadIdx.x;
	usize warp_idx = tid / WARP_SIZE;
	usize warp_m = (warp_idx / N_WARPS) * M_PER_WARP;
	usize warp_n = (warp_idx % N_WARPS) * N_PER_WARP;

	GMatrixDynSize<bf16, K> gA{A, M};
	GMatrixDynSize<bf16, K> gB{B, N};
	GMatrix<bf16, M_PER_BLOCK, K> gA_block = tile_m<M_PER_BLOCK>(gA, blockIdx.x);
	GMatrix<bf16, N_PER_BLOCK, K> gB_block = tile_m<N_PER_BLOCK>(gB, blockIdx.y);

	extern __shared__ bf16 smem[];
	SMatrix<bf16, M_PER_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload{smem};
	SMatrix<bf16, N_PER_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload{sA_preload._ptr + sA_preload.bytes()};

	X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD - 1; ++p) {
		cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, p, K_ITERS);
		cp_async_commit();
	}

	Fragment_16x16<f32> acc[M_TILES][N_TILES];
	zero_(acc);

	SMatrix<bf16, M_PER_BLOCK, K_STEP> sA;
	SMatrix<bf16, N_PER_BLOCK, K_STEP> sB;

	Fragment_16x16<bf16> rA[K_TILES][M_TILES];
	Fragment_16x16<bf16> rB[K_TILES][N_TILES];

	X17_NO_UNROLL for (usize k_step = 0; k_step < K_ITERS; ++k_step) {
		{ // Get more data from GMEM
			cp_async_wait<GMEM_PRELOAD - 2>();
			sync_threads();
			sA = tile_m<M_PER_BLOCK>(sA_preload, k_step % GMEM_PRELOAD);
			sB = tile_m<N_PER_BLOCK>(sB_preload, k_step % GMEM_PRELOAD);

			X17_UNROLL for (usize k_tile = 0; k_tile < K_TILES; ++k_tile) {
				X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
					smem_tile_to_fragment(sA, warp_m + mi * 16, k_tile * 16, rA[k_tile][mi]);
				}
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					smem_tile_to_fragment(sB, warp_n + ni * 16, k_tile * 16, rB[k_tile][ni]);
				}
			}

			cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, k_step + GMEM_PRELOAD - 1, K_ITERS);
			cp_async_commit();
		}

		X17_UNROLL for (usize k_tile = 0; k_tile < K_TILES; ++k_tile) {
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					mma_a_bt(rA[k_tile][mi], rB[k_tile][ni], acc[mi][ni]);
				}
			}
		}
	}

	bf16 *c_ptr = C + blockIdx.x * M_PER_BLOCK * N + blockIdx.y * N_PER_BLOCK;
	GMatrix<bf16, M_PER_BLOCK, N_PER_BLOCK> gC_block{c_ptr, N};
	X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
		store(acc[mi], gC_block, warp_m + mi * 16, warp_n);
	}
}

int main(int argc, char *argv[]) {
	constexpr usize K = 1024;
	bool perf_mode = argc > 1;
	usize M = perf_mode ? 32768 : 256;
	usize N = perf_mode ? 4096 : 256;

	std::vector<bf16> h_A(M * K), h_B(N * K), h_C(M * N);
	std::vector<float> h_ref(M * N, 0.0f);

	srand(42);
	for (bf16 &x : h_A) x = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
	for (bf16 &x : h_B) x = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);

	if (!perf_mode) {
		for (usize m = 0; m < M; m++) {
			for (usize n = 0; n < N; n++) {
				float sum = 0.0f;
				for (usize k = 0; k < K; k++) {
					sum += float(h_A[m * K + k]) * float(h_B[n * K + k]);
				}
				h_ref[m * N + n] = sum;
			}
		}
	}

	bf16 *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, M * K * sizeof(bf16));
	cudaMalloc(&d_B, N * K * sizeof(bf16));
	cudaMalloc(&d_C, M * N * sizeof(bf16));
	cudaMemcpy(d_A, h_A.data(), M * K * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), N * K * sizeof(bf16), cudaMemcpyHostToDevice);

	dim3 grid(M / M_PER_BLOCK, N / N_PER_BLOCK);
	usize smem_bytes = GMEM_PRELOAD * K_STEP * (M_PER_BLOCK + N_PER_BLOCK) * sizeof(bf16);

	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	GPU_Clock timer;
	timer.start();
	constexpr int NUM_RUNS = 100;
	for (int i = 0; i < NUM_RUNS; ++i) {
		gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(M, N, d_A, d_B, d_C);
	}
	cudaDeviceSynchronize();
	double elapsed = timer.seconds() / NUM_RUNS;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	double flops = 2.0 * M * N * K;
	double tflops = flops / elapsed / 1e12;
	printf("M=%u, N=%u, K=%u\n", M, N, K);
	printf("Average kernel time over %d runs: %.3f ms\n", NUM_RUNS, elapsed * 1e3);
	printf("%.2f TFLOPS\n", tflops);

	cudaMemcpy(h_C.data(), d_C, M * N * sizeof(bf16), cudaMemcpyDeviceToHost);

	if (!perf_mode) {
		usize exact_match = 0;
		float max_abs_diff = 0.0f;
		for (usize i = 0; i < M * N; i++) {
			float got = float(h_C[i]);
			float ref = h_ref[i];
			bf16 ref_bf16 = bf16(ref);
			u16 ref_bits, got_bits;
			memcpy(&ref_bits, &ref_bf16, 2);
			memcpy(&got_bits, &h_C[i], 2);
			if (ref_bits == got_bits) exact_match++;
			max_abs_diff = fmaxf(max_abs_diff, fabsf(got - ref));
		}

		printf("Exact bf16 match: %u/%u (%.2f%%)\n",
			exact_match, M * N, 100.0 * exact_match / (M * N));
		printf("Max abs diff: %e\n", max_abs_diff);

		printf("\nFirst 4x4 (GPU):\n");
		for (usize m = 0; m < 4; m++) {
			for (usize n = 0; n < 4; n++)
				printf(" %10.4f", float(h_C[m * N + n]));
			printf("\n");
		}
		printf("\nFirst 4x4 (ref):\n");
		for (usize m = 0; m < 4; m++) {
			for (usize n = 0; n < 4; n++)
				printf(" %10.4f", h_ref[m * N + n]);
			printf("\n");
		}
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
