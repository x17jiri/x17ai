#include "utils.cuh"

// Matrix dimensions - hardcoded for simplicity
#define ROWS 16
#define COLS 16
#define MATRIX_SIZE (ROWS * COLS)

// Kernel that demonstrates cp.async and ldmatrix
__global__ void ldmatrix_kernel(f16* gmem) {
	//using namespace sm80;
    // Shared memory for the 8x8 matrix
    __shared__ f16 smem[MATRIX_SIZE];

    usize tid = threadIdx.x;

	Matrix<GPtr<f16>, ROWS, COLS> gmatrix{GPtr<f16>(gmem)};
	Matrix<SPtr<f16>, ROWS, COLS> smatrix{SPtr<f16>(smem)};
	cp_asyncx<32, ROWS, COLS>(tid, gmatrix, smatrix);

	/*cp_async(
		reinterpret_cast<u128 const *>(gmem + 8*tid),
		ldmatrix_swizzle(smem, 8*tid)
	);*/

	// Wait for all async copies to complete
	cp_async_commit();
	cp_async_wait<0>();
	__syncwarp();

	union {
		u32 reg;
		f16 halves[2];
	} a, b, c, d;
	u32 off = ((tid & 16) / 2) + ((tid & 15) * COLS);
	ldmatrix_8x8xu16_t_x4(
		ldmatrix_swizzle(smem, off),
		a.reg, b.reg, c.reg, d.reg
	);
/*
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
--section SourceCounters \
./your_application
*/
	printf("Thread %2d: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]; off = %d\n",
		tid,
		__half2float(a.halves[0]), __half2float(a.halves[1]),
		__half2float(b.halves[0]), __half2float(b.halves[1]),
		__half2float(c.halves[0]), __half2float(c.halves[1]),
		__half2float(d.halves[0]), __half2float(d.halves[1]),
		off
	);

	//for (int i = 0; i < 8; i++) {
	//	printf("smem[%2d] = %.1f\n", 8*tid + i, __half2float(smem[8*tid + i]));
	//}
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
