#include "utils2.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
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
constexpr usize A_ROWS_DEFAULT = 2048;
constexpr usize K = 4096;
constexpr usize B_COLS_DEFAULT = 32768;
constexpr usize OUT_M_PER_BLOCK = M_PER_BLOCK / 2;

static_assert(WARPS_PER_BLOCK == 4);
static_assert(K_STEP % 16 == 0);
static_assert(K % K_STEP == 0);
static_assert(K % 16 == 0);
static_assert(M_PER_WARP % 2 == 0);
static_assert((K * sizeof(bf16)) % 16 == 0);

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

X17_DEVICE void store_geglu_8x8(
	Fragment_8x8<f32> const &frag,
	usize dst_row,
	usize dst_col,
	GMatrix<bf16, OUT_M_PER_BLOCK, N_PER_BLOCK> const &gC_block
) {
	usize lane = threadIdx.x % WARP_SIZE;
	usize row = lane / 4;
	usize col_pair = lane % 4;
	f32 lin0 = frag.val0;
	f32 lin1 = frag.val1;
	f32 gate0 = shuffle_xor_sync(frag.val0, 4);
	f32 gate1 = shuffle_xor_sync(frag.val1, 4);
	if ((row & 1) != 0) {
		return;
	}
	FragmentReg<bf16> packed;
	packed.set(
		bf16(math::fast::gelu(lin0) * gate0),
		bf16(math::fast::gelu(lin1) * gate1)
	);
	u8 *dst_base = reinterpret_cast<u8 *>(gC_block._ptr);
	usize dst_stride = gC_block.stride_bytes();
	*reinterpret_cast<u32 *>(dst_base + (dst_row + row / 2) * dst_stride + (dst_col + col_pair * 2) * sizeof(bf16)) = packed.val;
}

template<usize N_TILE_CNT>
X17_DEVICE void store_geglu(
	Fragment_16x16<f32> (&acc)[M_TILES][N_TILE_CNT],
	usize warp_m,
	usize warp_n,
	GMatrix<bf16, OUT_M_PER_BLOCK, N_PER_BLOCK> const &gC_block
) {
	X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
		usize dst_row = warp_m / 2 + mi * 8;
		X17_UNROLL for (usize ni = 0; ni < N_TILE_CNT; ++ni) {
			usize dst_col = warp_n + ni * 16;
			store_geglu_8x8(acc[mi][ni].sub[0][0], dst_row + 0, dst_col + 0, gC_block);
			store_geglu_8x8(acc[mi][ni].sub[0][1], dst_row + 0, dst_col + 8, gC_block);
			store_geglu_8x8(acc[mi][ni].sub[1][0], dst_row + 4, dst_col + 0, gC_block);
			store_geglu_8x8(acc[mi][ni].sub[1][1], dst_row + 4, dst_col + 8, gC_block);
		}
	}
}

// GEMM: C[A_ROWS/2, B_COLS] = GeGLU(A[A_ROWS, K] @ B[K, B_COLS])
// A is the weight matrix [A_ROWS, K] row-major.
// B is the input matrix stored transposed as [B_COLS, K] row-major so K is contiguous.
// Neighboring output rows are paired as GeGLU(gelu(even_row) * odd_row).
template<usize K>
__global__ void gemm_kernel(
	usize A_ROWS, usize B_COLS,
	bf16 *A,
	bf16 *B,
	bf16 *C
) {
	constexpr usize K_ITERS = K / K_STEP;
	constexpr usize K_TILES = K_STEP / 16;

	usize tid = threadIdx.x;
	usize warp_idx = tid / WARP_SIZE;
	usize warp_m = (warp_idx / N_WARPS) * M_PER_WARP;
	usize warp_n = (warp_idx % N_WARPS) * N_PER_WARP;
	GMatrixDynSize<bf16, K> gA{A, A_ROWS};
	GMatrixDynSize<bf16, K> gB{B, B_COLS};
	GMatrix<bf16, M_PER_BLOCK, K> gA_block = tile_m<M_PER_BLOCK>(gA, blockIdx.x);
	GMatrix<bf16, N_PER_BLOCK, K> gB_block = tile_m<N_PER_BLOCK>(gB, blockIdx.y);

	u32 smem = 0;
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

	bf16 *c_ptr = C + blockIdx.x * OUT_M_PER_BLOCK * B_COLS + blockIdx.y * N_PER_BLOCK;
	GMatrix<bf16, OUT_M_PER_BLOCK, N_PER_BLOCK> gC_block{c_ptr, B_COLS};
	store_geglu(acc, warp_m, warp_n, gC_block);
}

int main(int argc, char *argv[]) {
	usize A_ROWS = A_ROWS_DEFAULT;
	usize B_COLS = B_COLS_DEFAULT;
	if (argc == 3) {
		A_ROWS = static_cast<usize>(strtoul(argv[1], nullptr, 10));
		B_COLS = static_cast<usize>(strtoul(argv[2], nullptr, 10));
	} else if (argc != 1) {
		printf("Usage: %s [A_ROWS B_COLS]\n", argv[0]);
		return 1;
	}

	if (A_ROWS % M_PER_BLOCK != 0 || A_ROWS % 2 != 0 || B_COLS % N_PER_BLOCK != 0) {
		printf("Expected A_ROWS %% %u == 0, A_ROWS %% 2 == 0, and B_COLS %% %u == 0\n", M_PER_BLOCK, N_PER_BLOCK);
		return 1;
	}

	{
		std::ofstream config_file("tmp/o_gemm.config.json", std::ios::binary);
		config_file << "{\n"
			<< "  \"A_ROWS\": " << A_ROWS << ",\n"
			<< "  \"K\": " << K << ",\n"
			<< "  \"B_COLS\": " << B_COLS << "\n"
			<< "}\n";
	}

	std::vector<bf16> h_A(A_ROWS * K), h_B(B_COLS * K), h_C((A_ROWS / 2) * B_COLS);
	std::ifstream a_in("tmp/o_a.bin", std::ios::binary);
	if (!a_in) {
		printf("Failed to open tmp/o_a.bin\n");
		return 1;
	}
	if (!a_in.read(
		reinterpret_cast<char *>(h_A.data()),
		static_cast<std::streamsize>(h_A.size() * sizeof(bf16))
	)) {
		printf("Failed to read tmp/o_a.bin as A=[%u, %u]\n", A_ROWS, K);
		return 1;
	}

	std::ifstream b_in("tmp/o_b.bin", std::ios::binary);
	if (!b_in) {
		printf("Failed to open tmp/o_b.bin\n");
		return 1;
	}
	if (!b_in.read(
		reinterpret_cast<char *>(h_B.data()),
		static_cast<std::streamsize>(h_B.size() * sizeof(bf16))
	)) {
		printf("Failed to read tmp/o_b.bin as B^T=[%u, %u]\n", B_COLS, K);
		return 1;
	}

	bf16 *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, A_ROWS * K * sizeof(bf16));
	cudaMalloc(&d_B, B_COLS * K * sizeof(bf16));
	cudaMalloc(&d_C, (A_ROWS / 2) * B_COLS * sizeof(bf16));
	cudaMemcpy(d_A, h_A.data(), A_ROWS * K * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), B_COLS * K * sizeof(bf16), cudaMemcpyHostToDevice);

	dim3 grid(A_ROWS / M_PER_BLOCK, B_COLS / N_PER_BLOCK);
	usize smem_bytes = GMEM_PRELOAD * K_STEP * (M_PER_BLOCK + N_PER_BLOCK) * sizeof(bf16);

	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	int warmup = argc == 1 ? 0 : 50;
	for (int i = 0; i < warmup; ++i) {
		gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(A_ROWS, B_COLS, d_A, d_B, d_C);
	}
	cudaDeviceSynchronize();

	GPU_Clock timer;
	timer.start();
	int NUM_RUNS = argc == 1 ? 1 : 100;
	for (int i = 0; i < NUM_RUNS; ++i) {
		gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(A_ROWS, B_COLS, d_A, d_B, d_C);
	}
	cudaDeviceSynchronize();
	double elapsed = timer.seconds() / NUM_RUNS;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	double flops = 2.0 * A_ROWS * B_COLS * K;
	double tflops = flops / elapsed / 1e12;
	printf("A=[%u, %u], B=[%u, %u] stored as B^T=[%u, %u], out=[%u, %u]\n", A_ROWS, K, K, B_COLS, B_COLS, K, A_ROWS / 2, B_COLS);
	printf("Average kernel time over %d runs: %.3f ms\n", NUM_RUNS, elapsed * 1e3);
	printf("%.2f TFLOPS\n", tflops);

	cudaMemcpy(h_C.data(), d_C, (A_ROWS / 2) * B_COLS * sizeof(bf16), cudaMemcpyDeviceToHost);

	std::ofstream out_file("tmp/o_out_cpu.bin", std::ios::binary);
	out_file.write(
		reinterpret_cast<char *>(h_C.data()),
		static_cast<std::streamsize>(h_C.size() * sizeof(bf16))
	);

	printf("\nFirst 4x4 (GPU):\n");
	for (usize m = 0; m < 4; m++) {
		for (usize n = 0; n < 4; n++)
			printf(" %10.4f", float(h_C[m * B_COLS + n]));
		printf("\n");
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
