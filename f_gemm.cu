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
constexpr usize GMEM_PRELOAD = 2;
constexpr usize M_TILES = M_PER_WARP / 16;
constexpr usize N_TILES = N_PER_WARP / 16;
constexpr usize A_ROWS_DEFAULT = 1024;
constexpr usize K = 2048;
constexpr usize B_COLS_DEFAULT = 32768;

static_assert(WARPS_PER_BLOCK == 4);
static_assert(K_STEP % 16 == 0);
static_assert(K % K_STEP == 0);
static_assert(K % 16 == 0);
static_assert(M_PER_WARP % 2 == 0);
static_assert((K * sizeof(bf16)) % 16 == 0);
static_assert(N_TILES % 2 == 0);

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

X17_DEVICE f32 sum_squares_4(f32 a, f32 b, f32 c, f32 d) {
	f32 sum = 0.0f;
	sum = math::fma(a, a, sum);
	sum = math::fma(b, b, sum);
	sum = math::fma(c, c, sum);
	sum = math::fma(d, d, sum);
	return sum;
}

// GEMM: C[B_COLS, A_ROWS] = (A[A_ROWS, K] @ B[K, B_COLS])^T
// A is the weight matrix [A_ROWS, K] row-major.
// B is the input matrix stored transposed as [B_COLS, K] row-major so K is contiguous.
template<usize K>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void gemm_kernel(
	usize A_ROWS, usize B_COLS,
	bf16 *A,
	bf16 *B,
	bf16 *C,
	f32 *L2
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

	X17_UNROLL for (usize p = 0; p < GMEM_PRELOAD; ++p) {
		cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, p, K_ITERS);
		cp_async_commit();
	}

	Fragment_16x16<f32> acc_t[N_TILES][M_TILES];
	zero_(acc_t);

	Fragment_16x16<bf16> rA[K_TILES][M_TILES];
	Fragment_16x16<bf16> rB[K_TILES][N_TILES];

	SMatrix<bf16, M_PER_BLOCK, K_STEP> sA = tile_m<M_PER_BLOCK>(sA_preload, 0);
	SMatrix<bf16, N_PER_BLOCK, K_STEP> sB = tile_m<N_PER_BLOCK>(sB_preload, 0);

	cp_async_wait<GMEM_PRELOAD - 1>();
	sync_threads();

	X17_UNROLL for (usize k_tile = 0; k_tile < K_TILES; ++k_tile) {
		X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
			smem_tile_to_fragment(sB, warp_n + ni * 16, k_tile * 16, rB[k_tile][ni]);
		}
		X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
			smem_tile_to_fragment(sA, warp_m + mi * 16, k_tile * 16, rA[k_tile][mi]);
		}
	}

	X17_UNROLL for (usize k_step = 0; k_step < K_ITERS; ++k_step) {
		{ // Get more data from GMEM
			cp_async_wait<GMEM_PRELOAD - 2>();
			sync_threads();
			sA = tile_m<M_PER_BLOCK>(sA_preload, (k_step + 1) % GMEM_PRELOAD);
			sB = tile_m<N_PER_BLOCK>(sB_preload, (k_step + 1) % GMEM_PRELOAD);

			cp_async_ab(gA_block, gB_block, sA_preload, sB_preload, k_step + GMEM_PRELOAD, K_ITERS);
			cp_async_commit();
		}

		X17_UNROLL for (usize k_tile = 0; k_tile < K_TILES; ++k_tile) {
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
					mma_a_bt(rB[k_tile][ni], rA[k_tile][mi], acc_t[ni][mi]);
				}
				smem_tile_to_fragment(sA, warp_m + mi * 16, k_tile * 16, rA[k_tile][mi]);
			}
			X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
				smem_tile_to_fragment(sB, warp_n + ni * 16, k_tile * 16, rB[k_tile][ni]);
			}
		}
	}

	bf16 *c_ptr = C + blockIdx.y * N_PER_BLOCK * A_ROWS + blockIdx.x * M_PER_BLOCK;
	GMatrix<bf16, N_PER_BLOCK, M_PER_BLOCK> gC_block{c_ptr, A_ROWS};
	X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
		store(acc_t[ni], gC_block, warp_n + ni * 16, warp_m);
	}

	if (L2 != nullptr) {
		usize lane_id = tid % WARP_SIZE;
		usize row_in_tile = lane_id / 4;
		usize row_base = blockIdx.y * N_PER_BLOCK + warp_n;
		X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
			f32 top_sum_squared = 0.0f;
			f32 bot_sum_squared = 0.0f;
			X17_UNROLL for (usize mi = 0; mi < M_TILES; ++mi) {
				auto const &tile = acc_t[ni][mi];
				top_sum_squared += sum_squares_4(
					tile.sub[0][0].val0,
					tile.sub[0][0].val1,
					tile.sub[0][1].val0,
					tile.sub[0][1].val1
				);
				bot_sum_squared += sum_squares_4(
					tile.sub[1][0].val0,
					tile.sub[1][0].val1,
					tile.sub[1][1].val0,
					tile.sub[1][1].val1
				);
			}
			atomicAdd(L2 + row_base + ni * 16 + row_in_tile, top_sum_squared);
			atomicAdd(L2 + row_base + ni * 16 + row_in_tile + 8, bot_sum_squared);
		}
	}
}

int main(int argc, char *argv[]) {
	usize A_ROWS = A_ROWS_DEFAULT;
	usize B_COLS = B_COLS_DEFAULT;
	if (argc == 3) {
		A_ROWS = static_cast<usize>(strtoul(argv[1], nullptr, 10));
		B_COLS = static_cast<usize>(strtoul(argv[2], nullptr, 10));
	} /*else if (argc != 1) {
		printf("Usage: %s [A_ROWS B_COLS]\n", argv[0]);
		printf("   or: %s --test-geglu\n", argv[0]);
		return 1;
	}*/

	if (A_ROWS % M_PER_BLOCK != 0 || B_COLS % N_PER_BLOCK != 0) {
		printf("Expected A_ROWS %% %u == 0 and B_COLS %% %u == 0\n", M_PER_BLOCK, N_PER_BLOCK);
		return 1;
	}

	{
		std::ofstream config_file("tmp/f_gemm.config.json", std::ios::binary);
		config_file << "{\n"
			<< "  \"A_ROWS\": " << A_ROWS << ",\n"
			<< "  \"K\": " << K << ",\n"
			<< "  \"B_COLS\": " << B_COLS << "\n"
			<< "}\n";
	}

	std::vector<bf16> h_A(A_ROWS * K), h_B(B_COLS * K), h_C(A_ROWS * B_COLS);
	std::vector<f32> h_L2(B_COLS);
	std::ifstream a_in("tmp/f_a.bin", std::ios::binary);
	if (!a_in) {
		printf("Failed to open tmp/f_a.bin\n");
		return 1;
	}
	if (!a_in.read(
		reinterpret_cast<char *>(h_A.data()),
		static_cast<std::streamsize>(h_A.size() * sizeof(bf16))
	)) {
		printf("Failed to read tmp/f_a.bin as A=[%u, %u]\n", A_ROWS, K);
		return 1;
	}

	std::ifstream b_in("tmp/f_b.bin", std::ios::binary);
	if (!b_in) {
		printf("Failed to open tmp/f_b.bin\n");
		return 1;
	}
	if (!b_in.read(
		reinterpret_cast<char *>(h_B.data()),
		static_cast<std::streamsize>(h_B.size() * sizeof(bf16))
	)) {
		printf("Failed to read tmp/f_b.bin as B^T=[%u, %u]\n", B_COLS, K);
		return 1;
	}

	bf16 *d_A, *d_B, *d_C;
	f32 *d_L2;
	cudaMalloc(&d_A, A_ROWS * K * sizeof(bf16));
	cudaMalloc(&d_B, B_COLS * K * sizeof(bf16));
	cudaMalloc(&d_C, A_ROWS * B_COLS * sizeof(bf16));
	cudaMalloc(&d_L2, B_COLS * sizeof(f32));
	cudaMemcpy(d_A, h_A.data(), A_ROWS * K * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), B_COLS * K * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemset(d_L2, 0, B_COLS * sizeof(f32));

	dim3 grid(A_ROWS / M_PER_BLOCK, B_COLS / N_PER_BLOCK);
	usize smem_bytes = GMEM_PRELOAD * K_STEP * (M_PER_BLOCK + N_PER_BLOCK) * sizeof(bf16);

	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
	cudaFuncSetAttribute(gemm_kernel<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	int warmup = argc == 1 ? 0 : 50;
	for (int i = 0; i < warmup; ++i) {
		gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(A_ROWS, B_COLS, d_A, d_B, d_C, d_L2);
	}
	cudaDeviceSynchronize();

	GPU_Clock timer;
	timer.start();
	int NUM_RUNS = argc == 1 ? 1 : 100;
	for (int i = 0; i < NUM_RUNS; ++i) {
		gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(A_ROWS, B_COLS, d_A, d_B, d_C, d_L2);
	}
	cudaDeviceSynchronize();
	double elapsed = timer.seconds() / NUM_RUNS;

	cudaMemset(d_L2, 0, B_COLS * sizeof(f32));
	gemm_kernel<K><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(A_ROWS, B_COLS, d_A, d_B, d_C, d_L2);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	double flops = 2.0 * A_ROWS * B_COLS * K;
	double tflops = flops / elapsed / 1e12;
	printf("A=[%u, %u], B=[%u, %u] stored as B^T=[%u, %u], out=[%u, %u]\n", A_ROWS, K, K, B_COLS, B_COLS, K, B_COLS, A_ROWS);
	printf("Average kernel time over %d runs: %.3f ms\n", NUM_RUNS, elapsed * 1e3);
	printf("%.2f TFLOPS\n", tflops);

	cudaMemcpy(h_C.data(), d_C, A_ROWS * B_COLS * sizeof(bf16), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_L2.data(), d_L2, B_COLS * sizeof(f32), cudaMemcpyDeviceToHost);

	std::ofstream out_file("tmp/f_out_cpu.bin", std::ios::binary);
	out_file.write(
		reinterpret_cast<char *>(h_C.data()),
		static_cast<std::streamsize>(h_C.size() * sizeof(bf16))
	);
	std::ofstream l2_file("tmp/f_L2.bin", std::ios::binary);
	l2_file.write(
		reinterpret_cast<char *>(h_L2.data()),
		static_cast<std::streamsize>(h_L2.size() * sizeof(f32))
	);

	printf("\nFirst 4x4 (GPU):\n");
	for (usize m = 0; m < 4; m++) {
		for (usize n = 0; n < 4; n++)
			printf(" %10.4f", float(h_C[m * A_ROWS + n]));
		printf("\n");
	}
	printf("\nFirst 4 L2 values:\n");
	for (usize m = 0; m < 4 && m < B_COLS; ++m) {
		printf(" %12.6f", h_L2[m]);
	}
	printf("\n");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_L2);
	return 0;
}
