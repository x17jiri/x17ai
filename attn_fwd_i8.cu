#include "cuda/attn_fwd_i8.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

using namespace config;

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, true, cli)) {
		return 1;
	}
	if (cli.use_torch_maxes) {
		printf("Using torch attention maxes\n");
	}

	constexpr usize HEADS_PER_KERNEL = 4;
	constexpr usize SCORE_DUMP_HEAD = 62;
	constexpr usize PACKED_DIM = N_HEADS * HEAD_DIM;
	constexpr usize KV_PACKED_DIM = 2 * PACKED_DIM;

	static_assert(MODEL_DIM == N_HEADS * HEAD_DIM);

	using AF = AttnForward<
		N_HEADS,
		HEADS_PER_KERNEL,
		HEAD_DIM,
		MODEL_DIM,
		V_SCALE_FIX,
		PACKED_DIM,
		KV_PACKED_DIM,
		PACKED_DIM
	>;

	if (seq_len % AF::Q_PER_BLOCK != 0) {
		printf("Expected seq_len %% %u == 0\n", AF::Q_PER_BLOCK);
		return 1;
	}

	std::vector<b8::FixedI8> h_Q = load_i8_tensor(
		tensor_path(cli.input_dir, "q_i8.bin"),
		seq_len,
		PACKED_DIM
	);
	std::vector<b8::FixedI8> h_KV = load_i8_tensor(
		tensor_path(cli.input_dir, "kv_i8.bin"),
		seq_len,
		KV_PACKED_DIM
	);
	std::vector<b8::FixedI8> h_sink_k = load_i8_tensor(
		torch_tensor_path("sinks_k_i8.bin"),
		N_HEADS,
		HEAD_DIM
	);
	std::vector<b8::FixedI8> h_sink_v = load_i8_tensor(
		torch_tensor_path("sinks_v_i8.bin"),
		N_HEADS,
		HEAD_DIM
	);
	std::vector<i32> h_maxes;
	if (cli.use_torch_maxes && std::filesystem::exists(torch_tensor_path("attn_maxes_i32.bin"))) {
		h_maxes = load_i32_tensor(torch_tensor_path("attn_maxes_i32.bin"), N_HEADS, seq_len);
		if (h_maxes.empty()) {
			return 1;
		}
		printf("Loaded precomputed attention maxes\n");
	} else if (cli.use_torch_maxes) {
		printf("Torch attention maxes requested but %s does not exist\n", torch_tensor_path("attn_maxes_i32.bin").c_str());
	}
	if (h_Q.empty() || h_KV.empty() || h_sink_k.empty() || h_sink_v.empty()) {
		return 1;
	}

	std::vector<b8::FixedI8> h_out_i8(seq_len * PACKED_DIM);
	std::vector<bf16> h_out(h_out_i8.size());
	std::vector<f32> h_L(N_HEADS * seq_len);
	std::vector<i32> h_scores_i32(seq_len * seq_len);

	b8::FixedI8 *d_Q = nullptr;
	b8::FixedI8 *d_KV = nullptr;
	b8::FixedI8 *d_sink_k = nullptr;
	b8::FixedI8 *d_sink_v = nullptr;
	i32 *d_maxes = nullptr;
	i32 *d_scores = nullptr;
	b8::FixedI8 *d_out = nullptr;
	f32 *d_L = nullptr;

	cudaMalloc(&d_Q, h_Q.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_KV, h_KV.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_sink_k, h_sink_k.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_sink_v, h_sink_v.size() * sizeof(b8::FixedI8));
	if (!h_maxes.empty()) {
		cudaMalloc(&d_maxes, h_maxes.size() * sizeof(i32));
	}
	cudaMalloc(&d_scores, h_scores_i32.size() * sizeof(i32));
	cudaMalloc(&d_out, h_out_i8.size() * sizeof(b8::FixedI8));
	cudaMalloc(&d_L, h_L.size() * sizeof(f32));

	cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_KV, h_KV.data(), h_KV.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sink_k, h_sink_k.data(), h_sink_k.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sink_v, h_sink_v.data(), h_sink_v.size() * sizeof(b8::FixedI8), cudaMemcpyHostToDevice);
	if (d_maxes != nullptr) {
		cudaMemcpy(d_maxes, h_maxes.data(), h_maxes.size() * sizeof(i32), cudaMemcpyHostToDevice);
	}
	cudaMemset(d_scores, 0x80, h_scores_i32.size() * sizeof(i32));

	cudaFuncSetAttribute(attn_forward<AF>, cudaFuncAttributeMaxDynamicSharedMemorySize, AF::SMEM_BYTES);
	cudaFuncSetAttribute(attn_forward<AF>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(seq_len / AF::Q_PER_BLOCK, AF::HEAD_GROUP_CNT);

	int warmup = 0;
	for (int i = 0; i < warmup; ++i) {
		attn_forward<AF>
			<<<grid, AF::THREADS_PER_BLOCK, AF::SMEM_BYTES>>>(
				seq_len,
				d_Q,
				d_KV,
				d_sink_k,
				d_sink_v,
				d_maxes,
				d_scores,
				d_out,
				d_L,
				WINDOW_SIZE
			);
	}
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	int num_runs = 1;
	std::vector<cudaEvent_t> starts(num_runs), ends(num_runs);
	for (int i = 0; i < num_runs; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < num_runs; ++i) {
		cudaEventRecord(starts[i]);
		attn_forward<AF>
			<<<grid, AF::THREADS_PER_BLOCK, AF::SMEM_BYTES>>>(
				seq_len,
				d_Q,
				d_KV,
				d_sink_k,
				d_sink_v,
				d_maxes,
				d_scores,
				d_out,
				d_L,
				WINDOW_SIZE
			);
		cudaEventRecord(ends[i]);
	}
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
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
	double tflops = AF::flops(seq_len, WINDOW_SIZE) * N_HEADS / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out_i8.data(), d_out, h_out_i8.size() * sizeof(b8::FixedI8), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_L.data(), d_L, h_L.size() * sizeof(f32), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_scores_i32.data(), d_scores, h_scores_i32.size() * sizeof(i32), cudaMemcpyDeviceToHost);

	for (usize i = 0; i < h_out_i8.size(); ++i) {
		h_out[i] = bf16(f32(h_out_i8[i]) / f32(b8::FIXED_I8_SCALE));
	}

	std::filesystem::create_directories("tmp/block_cuda");
	store_i8_tensor("tmp/block_cuda/attn_out_i8.bin", h_out_i8, seq_len, PACKED_DIM);
	store_tensor("tmp/block_cuda/attn_out.bin", h_out, seq_len, PACKED_DIM);
	store_f32_tensor("tmp/block_cuda/attn_L_f32.bin", h_L, N_HEADS, seq_len);
	store_i32_tensor("tmp/block_cuda/attn_scores_h62_i32.bin", h_scores_i32, seq_len, seq_len);
	printf("Wrote raw score dump for head %u\n", SCORE_DUMP_HEAD);

	printf("Used SMEM per kernel: %u\n", AF::SMEM_BYTES);

	cudaFree(d_Q);
	cudaFree(d_KV);
	cudaFree(d_sink_k);
	cudaFree(d_sink_v);
	cudaFree(d_maxes);
	cudaFree(d_scores);
	cudaFree(d_out);
	cudaFree(d_L);
	return 0;
}
