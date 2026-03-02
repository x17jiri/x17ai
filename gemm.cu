#include "utils2.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

constexpr usize WARPS_PER_BLOCK = 4;
constexpr usize THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr usize M_BLOCK = 32;
constexpr usize N_BLOCK = 32;
constexpr usize K_STEP = 64;

// GEMM: C[M,N] = A[M,K] * B^T[N,K]
// A is [M, K] row-major
// B is [N, K] row-major (transposed)
// C is [M, N] row-major
template<usize K>
__global__ void gemm_kernel(
	usize M, usize N,
	bf16 const *A,
	bf16 const *B,
	bf16 *C
) {
	static_assert(K % K_STEP == 0);
	constexpr usize K_ITERS = K / K_STEP;

	usize tid = threadIdx.x;
	usize warp_idx = tid / WARP_SIZE;

	// 2x2 warp tiling of the 32x32 output block
	usize warp_m = (warp_idx / 2) * 16;  // 0 or 16
	usize warp_n = (warp_idx % 2) * 16;  // 0 or 16

	extern __shared__ bf16 smem[];
	SMatrix<bf16, M_BLOCK, K_STEP> sA{smem};
	SMatrix<bf16, N_BLOCK, K_STEP> sB{sA._ptr + sA.bytes()};

	// Accumulator (f32)
	Fragment_16x16<f32> acc;
	acc.zero_();

	// This block's input tiles (full K dimension)
	GMatrix<bf16, M_BLOCK, K> gA_block{
		const_cast<bf16 *>(A) + blockIdx.x * M_BLOCK * K
	};
	GMatrix<bf16, N_BLOCK, K> gB_block{
		const_cast<bf16 *>(B) + blockIdx.y * N_BLOCK * K
	};

	for (usize ki = 0; ki < K_ITERS; ki++) {
		usize k = ki * K_STEP;

		// Load A[M_BLOCK, K_STEP] and B[N_BLOCK, K_STEP] to SMEM
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
			tid, gA_block.template slice_n<K_STEP>(k), sA
		);
		cp_async_gmem_to_smem<THREADS_PER_BLOCK>(
			tid, gB_block.template slice_n<K_STEP>(k), sB
		);
		cp_async_commit();
		cp_async_wait();
		__syncthreads();

		// Inner loop: K_STEP in steps of 16
		X17_UNROLL for (usize kk = 0; kk < K_STEP; kk += 16) {
			Fragment_16x16<bf16> a_frag, b_frag;
			sA.load_tile_to_fragment(warp_m, kk, a_frag);
			sB.load_tile_to_fragment(warp_n, kk, b_frag);
			mma_a_bt(a_frag, b_frag, acc);
		}

		__syncthreads();
	}

	// Store: each warp writes its 16x16 tile to GMEM
	bf16 *c_ptr = C + blockIdx.x * M_BLOCK * N + blockIdx.y * N_BLOCK;
	GMatrix<bf16, M_BLOCK, N_BLOCK> gC_block{c_ptr, N};
	acc.store(gC_block, warp_m, warp_n);
}

int main() {
	constexpr usize K = 1024;
	usize M = 256;
	usize N = 256;

	std::vector<bf16> h_A(M * K), h_B(N * K), h_C(M * N);
	std::vector<float> h_ref(M * N, 0.0f);

	// Random init
	srand(42);
	for (auto &x : h_A) x = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
	for (auto &x : h_B) x = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);

	// CPU reference: C = A * B^T, accumulate in f32
	for (usize m = 0; m < M; m++) {
		for (usize n = 0; n < N; n++) {
			float sum = 0;
			for (usize k = 0; k < K; k++) {
				sum += float(h_A[m * K + k]) * float(h_B[n * K + k]);
			}
			h_ref[m * N + n] = sum;
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
	usize smem_bytes = (M_BLOCK + N_BLOCK) * K_STEP * sizeof(bf16);
	gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(M, N, d_A, d_B, d_C);

	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	cudaMemcpy(h_C.data(), d_C, M * N * sizeof(bf16), cudaMemcpyDeviceToHost);

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

	printf("M=%u, N=%u, K=%u\n", M, N, K);
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

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
