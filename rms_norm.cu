#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

using namespace config;

namespace Rms_norm {
	static constexpr usize ROWS_PER_BLOCK = 32;
	static constexpr usize THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
	static constexpr usize VALUES_PER_LANE = 16;
	static constexpr usize VALUES_PER_WARP_STEP = WARP_SIZE * VALUES_PER_LANE;
	static constexpr usize SMEM_BYTES = ROWS_PER_BLOCK * MODEL_DIM * sizeof(b8::E4m3);

	static_assert(MODEL_DIM % VALUES_PER_WARP_STEP == 0);

	union PackedE4m3x16 {
		u32 words[4];
		b8::E4m3 values[VALUES_PER_LANE];
	};

	union PackedI8x16 {
		u32 words[4];
		i8 values[VALUES_PER_LANE];
	};

	X17_DEVICE f32 e4m3_to_f32(b8::E4m3 value) {
		return __bfloat162float(b8::e4m3_to_bf16(value));
	}

	X17_DEVICE f32 warp_sum(f32 value) {
		X17_UNROLL for (usize mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
			value += shuffle_xor_sync(value, int(mask));
		}
		return value;
	}

	X17_DEVICE f32 sum_squares(PackedE4m3x16 const &packed) {
		f32 sum = 0.0f;
		X17_UNROLL for (usize i = 0; i < VALUES_PER_LANE; ++i) {
			f32 value = e4m3_to_f32(packed.values[i]);
			sum = math::fma(value, value, sum);
		}
		return sum;
	}

	X17_DEVICE void normalize(PackedE4m3x16 const &inp, f32 rrms, PackedI8x16 &out) {
		X17_UNROLL for (usize i = 0; i < VALUES_PER_LANE; ++i) {
			f32 value = e4m3_to_f32(inp.values[i]);
			out.values[i] = b8::f32_to_fixedi8(value * rrms * f32(b8::FIXED_I8_SCALE));
		}
	}

	X17_KERNEL(THREADS_PER_BLOCK)
	void kernel(
		b8::E4m3 const *inp,
		usize n_inputs,
		b8::FixedI8 *out
	) {
		extern __shared__ __align__(16) u8 smem[];

		usize block_row = blockIdx.x * ROWS_PER_BLOCK;
		usize tid = threadIdx.x;

		constexpr usize CP_BYTES = sizeof(u128);
		constexpr usize CP_COUNT = SMEM_BYTES / CP_BYTES;
		static_assert(SMEM_BYTES % CP_BYTES == 0);

		u32 smem_ptr = cast_smem_ptr_to_uint(smem);
		b8::E4m3 const *gmem_inp = inp + block_row * MODEL_DIM;
		for (usize cp = tid; cp < CP_COUNT; cp += THREADS_PER_BLOCK) {
			u8 const *src = reinterpret_cast<u8 const *>(gmem_inp) + cp * CP_BYTES;
			sm80::cp_async(reinterpret_cast<u128 const *>(src), smem_ptr + cp * CP_BYTES);
		}
		async_load_commit();
		async_load_wait<0>();
		sync_threads();

		usize warp = tid / WARP_SIZE;
		usize lane = tid % WARP_SIZE;
		usize row = block_row + warp;
		if (row >= n_inputs) {
			return;
		}

		f32 local_sum = 0.0f;
		X17_UNROLL for (usize col = 0; col < MODEL_DIM; col += VALUES_PER_WARP_STEP) {
			PackedE4m3x16 packed;
			u32 ptr = smem_ptr + warp * MODEL_DIM + col + lane * VALUES_PER_LANE;
			load_shared_4x32b(ptr, packed.words[0], packed.words[1], packed.words[2], packed.words[3]);
			local_sum += sum_squares(packed);
		}

		f32 sum = warp_sum(local_sum);
		f32 rrms = rsqrtf(sum * f32(1.0 / f64(MODEL_DIM)) + f32(L2_NORM_EPS));

		X17_UNROLL for (usize col = 0; col < MODEL_DIM; col += VALUES_PER_WARP_STEP) {
			PackedE4m3x16 packed_inp;
			PackedI8x16 packed_out;
			usize local_col = col + lane * VALUES_PER_LANE;
			u32 ptr = smem_ptr + warp * MODEL_DIM + local_col;
			load_shared_4x32b(ptr, packed_inp.words[0], packed_inp.words[1], packed_inp.words[2], packed_inp.words[3]);
			normalize(packed_inp, rrms, packed_out);
			store_gmem_4x32b(
				reinterpret_cast<f32 *>(out + row * MODEL_DIM + local_col),
				__uint_as_float(packed_out.words[0]),
				__uint_as_float(packed_out.words[1]),
				__uint_as_float(packed_out.words[2]),
				__uint_as_float(packed_out.words[3])
			);
		}
	}
}

using namespace Rms_norm;

static bool check(cudaError_t err, char const *what) {
	if (err == cudaSuccess) {
		return true;
	}
	printf("%s failed: %s\n", what, cudaGetErrorString(err));
	return false;
}

int main() {
	if (seq_len % ROWS_PER_BLOCK != 0) {
		printf("Expected seq_len to be a multiple of %u, got %u\n", ROWS_PER_BLOCK, seq_len);
		return 1;
	}

	std::vector<b8::E4m3> h_inp = load_e4m3_tensor(torch_tensor_path("ffn_y_f8.bin"), seq_len, MODEL_DIM);
	if (h_inp.empty()) {
		return 1;
	}

	std::vector<b8::FixedI8> h_out(seq_len * MODEL_DIM);

	b8::E4m3 *d_inp = nullptr;
	b8::FixedI8 *d_out = nullptr;

	if (!check(cudaMalloc(&d_inp, h_inp.size() * sizeof(b8::E4m3)), "cudaMalloc(d_inp)")) {
		return 1;
	}
	if (!check(cudaMalloc(&d_out, h_out.size() * sizeof(b8::FixedI8)), "cudaMalloc(d_out)")) {
		return 1;
	}
	if (!check(cudaMemcpy(d_inp, h_inp.data(), h_inp.size() * sizeof(b8::E4m3), cudaMemcpyHostToDevice), "cudaMemcpy(d_inp)")) {
		return 1;
	}

	if (!check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES), "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)")) {
		return 1;
	}
	if (!check(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100), "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)")) {
		return 1;
	}

	dim3 grid(seq_len / ROWS_PER_BLOCK);
	int warmup = 20;
	for (int i = 0; i < warmup; ++i) {
		kernel<<<grid, THREADS_PER_BLOCK, SMEM_BYTES>>>(d_inp, seq_len, d_out);
	}

	if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)")) {
		return 1;
	}
	if (!check(cudaGetLastError(), "rms_norm warmup")) {
		return 1;
	}

	int num_runs = 100;
	std::vector<cudaEvent_t> starts(num_runs), ends(num_runs);
	for (int i = 0; i < num_runs; ++i) {
		if (!check(cudaEventCreate(&starts[i]), "cudaEventCreate(start)")) {
			return 1;
		}
		if (!check(cudaEventCreate(&ends[i]), "cudaEventCreate(end)")) {
			return 1;
		}
	}
	for (int i = 0; i < num_runs; ++i) {
		cudaEventRecord(starts[i]);
		kernel<<<grid, THREADS_PER_BLOCK, SMEM_BYTES>>>(d_inp, seq_len, d_out);
		cudaEventRecord(ends[i]);
	}
	if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(timing)")) {
		return 1;
	}
	if (!check(cudaGetLastError(), "rms_norm timing")) {
		return 1;
	}

	std::vector<float> times_ms(num_runs);
	for (int i = 0; i < num_runs; ++i) {
		cudaEventElapsedTime(&times_ms[i], starts[i], ends[i]);
		cudaEventDestroy(starts[i]);
		cudaEventDestroy(ends[i]);
	}
	std::sort(times_ms.begin(), times_ms.end());

	float median_ms = times_ms[num_runs / 2];
	float min_ms = times_ms[0];
	double bytes = double(seq_len) * double(MODEL_DIM) * (sizeof(b8::E4m3) + sizeof(b8::FixedI8));
	double gbps = bytes / (median_ms * 1e-3) / 1e9;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("Bandwidth: %.2f GB/s\n", gbps);

	if (!check(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(b8::FixedI8), cudaMemcpyDeviceToHost), "cudaMemcpy(h_out)")) {
		return 1;
	}
	std::filesystem::create_directories("tmp/block_cuda");
	store_i8_tensor("tmp/block_cuda/ffn_z_i8.bin", h_out, seq_len, MODEL_DIM);

	printf("Used SMEM per kernel: %u\n", SMEM_BYTES);

	cudaFree(d_inp);
	cudaFree(d_out);
	return 0;
}
