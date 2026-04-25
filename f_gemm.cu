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
constexpr usize A_ROWS_DEFAULT = 4096;
constexpr usize K = 1024;
constexpr usize B_COLS_DEFAULT = 32768;
constexpr usize OUT_M_PER_BLOCK = M_PER_BLOCK / 2;

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

X17_DEVICE void geglu(Fragment_8x8<bf16> &o, Fragment_8x8<f32> const &i1, Fragment_8x8<f32> const &i2) {
	o.set(
		bf16(math::fast::geglu<K>(i1.val0, i1.val1)),
		bf16(math::fast::geglu<K>(i2.val0, i2.val1))
	);
	o.transpose_();
	usize tid = threadIdx.x % WARP_SIZE;
	o.val = shuffle_sync(o.val, (tid & 12) * 2 + (tid & 16) / 4 + (tid & ~28));
	o.transpose_();
}

X17_DEVICE void geglu(Fragment_16x16<bf16> &o, Fragment_16x16<f32> const &i1, Fragment_16x16<f32> const &i2) {
	geglu(o.sub[0][0], i1.sub[0][0], i1.sub[0][1]);
	geglu(o.sub[0][1], i2.sub[0][0], i2.sub[0][1]);
	geglu(o.sub[1][0], i1.sub[1][0], i1.sub[1][1]);
	geglu(o.sub[1][1], i2.sub[1][0], i2.sub[1][1]);
}

template<usize N>
X17_DEVICE void geglu(Fragment_16x16<bf16> (&o)[N], Fragment_16x16<f32> const (&i)[2 * N]) {
	X17_UNROLL for (usize tile = 0; tile < N; ++tile) {
		geglu(o[tile], i[2 * tile], i[2 * tile + 1]);
	}
}

// GEMM: C[B_COLS, A_ROWS/2] = GeGLU((A[A_ROWS, K] @ B[K, B_COLS])^T)
// A is the weight matrix [A_ROWS, K] row-major.
// B is the input matrix stored transposed as [B_COLS, K] row-major so K is contiguous.
// After transposing the matmul output to [B_COLS, A_ROWS], GeGLU reduces adjacent
// values along the contiguous A_ROWS dimension, producing [B_COLS, A_ROWS/2].
template<usize K>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void gemm_kernel(
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

	Fragment_16x16<bf16> out[N_TILES];
	X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
		geglu(out[ni], acc_t[ni][0], acc_t[ni][1]);
	}

	usize out_cols = A_ROWS / 2;
	bf16 *c_ptr = C + blockIdx.y * N_PER_BLOCK * out_cols + blockIdx.x * OUT_M_PER_BLOCK;
	GMatrix<bf16, N_PER_BLOCK, OUT_M_PER_BLOCK> gC_block{c_ptr, out_cols};
	X17_UNROLL for (usize ni = 0; ni < N_TILES; ++ni) {
		store_2x2_8x8(
			gC_block,
			warp_n + ni * 16,
			warp_m / 2,
			out[ni].sub[0][0], out[ni].sub[0][1],
			out[ni].sub[1][0], out[ni].sub[1][1]
		);
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
		std::ofstream config_file("tmp/o_gemm.config.json", std::ios::binary);
		config_file << "{\n"
			<< "  \"A_ROWS\": " << A_ROWS << ",\n"
			<< "  \"K\": " << K << ",\n"
			<< "  \"B_COLS\": " << B_COLS << "\n"
			<< "}\n";
	}

	std::vector<bf16> h_A(A_ROWS * K), h_B(B_COLS * K), h_C(B_COLS * (A_ROWS / 2));
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
	cudaMalloc(&d_C, B_COLS * (A_ROWS / 2) * sizeof(bf16));
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
	printf("A=[%u, %u], B=[%u, %u] stored as B^T=[%u, %u], out=[%u, %u]\n", A_ROWS, K, K, B_COLS, B_COLS, K, B_COLS, A_ROWS / 2);
	printf("Average kernel time over %d runs: %.3f ms\n", NUM_RUNS, elapsed * 1e3);
	printf("%.2f TFLOPS\n", tflops);

	cudaMemcpy(h_C.data(), d_C, B_COLS * (A_ROWS / 2) * sizeof(bf16), cudaMemcpyDeviceToHost);

	std::ofstream out_file("tmp/o_out_cpu.bin", std::ios::binary);
	out_file.write(
		reinterpret_cast<char *>(h_C.data()),
		static_cast<std::streamsize>(h_C.size() * sizeof(bf16))
	);

	printf("\nFirst 4x4 (GPU):\n");
	for (usize m = 0; m < 4; m++) {
		for (usize n = 0; n < 4; n++)
			printf(" %10.4f", float(h_C[m * (A_ROWS / 2) + n]));
		printf("\n");
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
