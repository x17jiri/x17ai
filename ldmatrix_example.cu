#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdint.h>

using f16 = __half;
using bf16 = __nv_bfloat16;
using uint128_t = unsigned __int128;

#define CUTE_HOST_DEVICE __forceinline__ __host__ __device__
#define CUTE_DEVICE      __forceinline__          __device__
#define CUTE_HOST        __forceinline__ __host__

/// CUTE helper to cast SMEM pointer to unsigned
CUTE_DEVICE uint32_t cast_smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

/// Copy via cp.async with caching at all levels
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEALWAYS {
	using SRegisters = TS[1];
	using DRegisters = TD[1];

	static_assert(
		sizeof(TS) == sizeof(TD),
		"cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)"
	);
	static_assert(
		sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16,
		"cp.async sizeof(TS) is not supported"
	);

	CUTE_DEVICE static void copy(TS const *gmem_src, TD *smem_dst) {
		uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_dst);
		asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
			:: "r"(smem_int_ptr),
			"l"(gmem_src),
			"n"(sizeof(TS)));
	}
};

CUTE_DEVICE void cp_async_commit() {
	asm volatile("cp.async.commit_group;\n" ::);
}

/// Blocks until all but N previous cp.async.commit_group operations have committed.
template<int N>
CUTE_DEVICE void cp_async_wait() {
	if constexpr (N == 0) {
		asm volatile("cp.async.wait_all;\n" ::);
	} else {
		asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
	}
}

struct SM75_LDMATRIX_8x8xU16 {
	using SRegisters = uint128_t[1];
	using DRegisters = uint32_t[4];

	CUTE_DEVICE static void copy(
		uint128_t const* smem_src,
		uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3
	) {
		uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
		asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
			: "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
			:  "r"(smem_int_ptr));
	}
};

struct SM75_LDMATRIX_8x8xU16_T {
	using SRegisters = uint128_t[1];
	using DRegisters = uint32_t[4];

	CUTE_DEVICE static void copy(
		uint128_t const& smem_src,
		uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3
	) {
		uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
		asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
			: "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
			:  "r"(smem_int_ptr));
	}
};

// Matrix dimensions - hardcoded for simplicity
#define ROWS 16
#define COLS 16
#define MATRIX_SIZE (ROWS * COLS)

// Kernel that demonstrates cp.async and ldmatrix
__global__ void ldmatrix_kernel(f16* gmem) {
    // Shared memory for the 8x8 matrix
    __shared__ f16 smem[MATRIX_SIZE];

    int tid = threadIdx.x;

	using CP_BLOCK = f16[8];
	SM80_CP_ASYNC_CACHEALWAYS<CP_BLOCK, CP_BLOCK>::copy(
		reinterpret_cast<CP_BLOCK const *>(gmem + 8*tid),
		reinterpret_cast<CP_BLOCK *>(smem + 8*tid)
	);

	// Wait for all async copies to complete
	cp_async_commit();
	cp_async_wait<0>();
	__syncwarp();

	for (int i = 0; i < 8; i++) {
		printf("smem[%2d] = %.1f\n", 8*tid + i, __half2float(smem[8*tid + i]));
	}
/*
	//

	// Step 2: Use ldmatrix to load 8x8 block from shared memory to registers
	// ldmatrix.sync.aligned.m8n8.x4.shared.b16 loads a matrix fragment
	// The matrix must be stored in row-major format in shared memory
	// Each thread will hold 8 half values (4 x 32-bit registers with 2 halves each)

	unsigned reg[4];  // 4 x 32-bit registers = 8 x 16-bit halves

	// Calculate shared memory address for ldmatrix
	// ldmatrix expects 16-byte aligned address
	unsigned smem_addr = __cvta_generic_to_shared(smem);

	// Use ldmatrix to load 8x8 matrix fragment (m8n8)
	// This loads 4 x 32-bit registers per thread
	asm volatile(
		"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
		: "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
		: "r"(smem_addr)
	);

	//__syncwarp();

	// Step 3: Print the values held by each thread
	// Each 32-bit register contains 2 half values
	half* reg_as_half = (half*)reg;
*/

/*
	printf("Thread %2d: [", tid);
	for (int i = 0; i < 8; i++) {
		printf("%.1f", __half2float(reg_as_half[i]));
		if (i < 7) printf(", ");
	}
	printf("]\n");
*/
}

int main() {
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
    printf("Register values per thread after ldmatrix:\n");
    ldmatrix_kernel<<<1, 32>>>(d_matrix);

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Cleanup
    cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}
