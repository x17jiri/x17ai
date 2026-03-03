#include "utils2.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include "cutlass/util/GPU_Clock.hpp"

constexpr usize WARPS_PER_BLOCK = 4;
constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr usize K_STEP = 64;
constexpr usize GMEM_PRELOAD = 2;

// Warp tiling: 2x2 warps, each warp owns WARP_M x WARP_N of the output
constexpr usize WARP_M = 32;
constexpr usize WARP_N = 64;
constexpr usize M_BLOCK = 2 * WARP_M;  // 128
constexpr usize N_BLOCK = 2 * WARP_N;  // 128
constexpr usize MT = WARP_M / 16;      // m tiles per warp
constexpr usize NT = WARP_N / 16;      // n tiles per warp

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

	usize tid = threadIdx.x;
	usize warp_idx = tid / WARP_SIZE;

	// 2x2 warp tiling: each warp owns WARP_M x WARP_N
	usize warp_m = (warp_idx / 2) * WARP_M;
	usize warp_n = (warp_idx % 2) * WARP_N;

	// This block's input tiles (full K dimension)
	GMatrix<bf16, M_BLOCK, K> gA_block{A + blockIdx.x * M_BLOCK * K};
	GMatrix<bf16, N_BLOCK, K> gB_block{B + blockIdx.y * N_BLOCK * K};

	extern __shared__ bf16 smem[];
	SMatrix<bf16, M_BLOCK * GMEM_PRELOAD, K_STEP> sA_preload{smem};
	SMatrix<bf16, N_BLOCK * GMEM_PRELOAD, K_STEP> sB_preload{sA_preload._ptr + sA_preload.bytes()};

	// Load A[M_BLOCK, K_STEP] and B[N_BLOCK, K_STEP] to SMEM
	X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
		if (p < K_ITERS) {
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				tid,
				gA_block.template slice_n<K_STEP>(p * K_STEP),
				sA_preload.template tile_m<M_BLOCK>(p)
			);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				tid,
				gB_block.template slice_n<K_STEP>(p * K_STEP),
				sB_preload.template tile_m<N_BLOCK>(p)
			);
			cp_async_commit();
		}
	}
	cp_async_wait<GMEM_PRELOAD - 1>();
	__syncthreads();

	SMatrix<bf16, M_BLOCK, K_STEP> sA = sA_preload.template tile_m<M_BLOCK>(0);
	SMatrix<bf16, N_BLOCK, K_STEP> sB = sB_preload.template tile_m<N_BLOCK>(0);

	// MT x NT accumulators per warp
	Fragment_16x16<f32> acc[MT][NT];
	X17_UNROLL for (usize mi = 0; mi < MT; mi++)
		X17_UNROLL for (usize ni = 0; ni < NT; ni++)
			acc[mi][ni].zero_();

	Fragment_16x16<bf16> a[4][MT], b[4][NT]; // [k_tile][m_or_n_tile]

	// Preload first two k-tiles from SMEM to registers
	X17_UNROLL for (usize mi = 0; mi < MT; mi++)
		sA.load_tile_to_fragment(warp_m + mi*16, 0*16, a[0][mi]);
	X17_UNROLL for (usize ni = 0; ni < NT; ni++)
		sB.load_tile_to_fragment(warp_n + ni*16, 0*16, b[0][ni]);

	X17_UNROLL for (usize mi = 0; mi < MT; mi++)
		sA.load_tile_to_fragment(warp_m + mi*16, 1*16, a[1][mi]);
	X17_UNROLL for (usize ni = 0; ni < NT; ni++)
		sB.load_tile_to_fragment(warp_n + ni*16, 1*16, b[1][ni]);

	for (usize k = 0; k < K_ITERS; k++) {
		// Load k-tiles 2,3 from SMEM
		X17_UNROLL for (usize mi = 0; mi < MT; mi++)
			sA.load_tile_to_fragment(warp_m + mi*16, 2*16, a[2][mi]);
		X17_UNROLL for (usize ni = 0; ni < NT; ni++)
			sB.load_tile_to_fragment(warp_n + ni*16, 2*16, b[2][ni]);

		X17_UNROLL for (usize mi = 0; mi < MT; mi++)
			sA.load_tile_to_fragment(warp_m + mi*16, 3*16, a[3][mi]);
		X17_UNROLL for (usize ni = 0; ni < NT; ni++)
			sB.load_tile_to_fragment(warp_n + ni*16, 3*16, b[3][ni]);

		// MMA for k-tiles 0,1
		X17_UNROLL for (usize mi = 0; mi < MT; mi++)
			X17_UNROLL for (usize ni = 0; ni < NT; ni++)
				mma_a_bt(a[0][mi], b[0][ni], acc[mi][ni]);
		X17_UNROLL for (usize mi = 0; mi < MT; mi++)
			X17_UNROLL for (usize ni = 0; ni < NT; ni++)
				mma_a_bt(a[1][mi], b[1][ni], acc[mi][ni]);

		// Ensure all warps finished reading from SMEM before overwriting
		__syncthreads();

		usize p = k + GMEM_PRELOAD;
		if (p < K_ITERS) {
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				tid,
				gA_block.template slice_n<K_STEP>(p * K_STEP),
				sA_preload.template tile_m<M_BLOCK>(p % GMEM_PRELOAD)
			);
			cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
				tid,
				gB_block.template slice_n<K_STEP>(p * K_STEP),
				sB_preload.template tile_m<N_BLOCK>(p % GMEM_PRELOAD)
			);
		}
		cp_async_commit();

		cp_async_wait<GMEM_PRELOAD - 1>();
		__syncthreads();
		sA = sA_preload.template tile_m<M_BLOCK>((k + 1) % GMEM_PRELOAD);
		sB = sB_preload.template tile_m<N_BLOCK>((k + 1) % GMEM_PRELOAD);

		// Preload k-tiles 0,1 from next buffer
		X17_UNROLL for (usize mi = 0; mi < MT; mi++)
			sA.load_tile_to_fragment(warp_m + mi*16, 0*16, a[0][mi]);
		X17_UNROLL for (usize ni = 0; ni < NT; ni++)
			sB.load_tile_to_fragment(warp_n + ni*16, 0*16, b[0][ni]);

		X17_UNROLL for (usize mi = 0; mi < MT; mi++)
			sA.load_tile_to_fragment(warp_m + mi*16, 1*16, a[1][mi]);
		X17_UNROLL for (usize ni = 0; ni < NT; ni++)
			sB.load_tile_to_fragment(warp_n + ni*16, 1*16, b[1][ni]);

		// MMA for k-tiles 2,3
		X17_UNROLL for (usize mi = 0; mi < MT; mi++)
			X17_UNROLL for (usize ni = 0; ni < NT; ni++)
				mma_a_bt(a[2][mi], b[2][ni], acc[mi][ni]);
		X17_UNROLL for (usize mi = 0; mi < MT; mi++)
			X17_UNROLL for (usize ni = 0; ni < NT; ni++)
				mma_a_bt(a[3][mi], b[3][ni], acc[mi][ni]);
	}

	// Store: each warp writes its MT x NT tile of 16x16 fragments to GMEM
	bf16 *c_ptr = C + blockIdx.x * M_BLOCK * N + blockIdx.y * N_BLOCK;
	GMatrix<bf16, M_BLOCK, N_BLOCK> gC_block{c_ptr, N};
	X17_UNROLL for (usize mi = 0; mi < MT; mi++)
		X17_UNROLL for (usize ni = 0; ni < NT; ni++)
			acc[mi][ni].store(gC_block, warp_m + mi*16, warp_n + ni*16);
}

int main(int argc, char *argv[]) {
	constexpr usize K = 1024;
	bool perf_mode = argc > 1;
	usize M = perf_mode ? 4*4096 : 256;
	usize N = perf_mode ? 4*4096 : 256;

	std::vector<bf16> h_A(M * K), h_B(N * K), h_C(M * N);
	std::vector<float> h_ref(M * N, 0.0f);

	// Random init
	srand(42);
	for (auto &x : h_A) x = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
	for (auto &x : h_B) x = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);

	// CPU reference: C = A * B^T, accumulate in f32
	if (!perf_mode) {
		for (usize m = 0; m < M; m++) {
			for (usize n = 0; n < N; n++) {
				float sum = 0;
				for (usize k = 0; k < K; k++) {
					sum += float(h_A[m * K + k]) * float(h_B[n * K + k]);
				}
				h_ref[m * N + n] = sum;
			}
		}
	}

	// Device
	bf16 *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, M * K * sizeof(bf16));
	cudaMalloc(&d_B, N * K * sizeof(bf16));
	cudaMalloc(&d_C, M * N * sizeof(bf16));
	cudaMemcpy(d_A, h_A.data(), M * K * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), N * K * sizeof(bf16), cudaMemcpyHostToDevice);

	// Launch
	dim3 grid(M / M_BLOCK, N / N_BLOCK);
	usize smem_bytes = (M_BLOCK + N_BLOCK) * GMEM_PRELOAD * K_STEP * sizeof(bf16);

	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	GPU_Clock timer;
	timer.start();
	constexpr int NUM_RUNS = 2;
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

	// Compute TFLOPS: 2*M*N*K FLOPs per GEMM
	double flops = 2.0 * M * N * K;
	double tflops = flops / elapsed / 1e12;
	printf("M=%u, N=%u, K=%u\n", M, N, K);
	printf("Average kernel time over %d runs: %.3f ms\n", NUM_RUNS, elapsed * 1e3);
	printf("%.2f TFLOPS\n", tflops);

	cudaMemcpy(h_C.data(), d_C, M * N * sizeof(bf16), cudaMemcpyDeviceToHost);

	if (!perf_mode) {
		// Compare
		usize exact_match = 0;
		float max_abs_diff = 0;
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
