#include "cuda/attn_fwd.cuh"
#include "block.config.hpp"
#include "utils2.cuh"

#include <algorithm>
#include <filesystem>

int main(int argc, char *argv[]) {
	HarnessCliOptions cli;
	if (!parse_harness_cli_args(argc, argv, true, cli)) {
		return 1;
	}
	if (cli.use_torch_maxes) {
		printf("Using torch attention maxes\n");
	}

	constexpr usize HEADS_PER_KERNEL = 2;
	constexpr usize QK_DIM = config::head_dim;
	constexpr usize V_DIM = config::head_dim;
	constexpr bool V_EQUALS_K = false;
	constexpr usize SEQ_LEN = config::n_inputs;
	constexpr usize HEAD_CNT = config::n_heads;
	constexpr usize PACKED_DIM = HEAD_CNT * V_DIM;
	constexpr usize D_MODEL = config::d_model;
	constexpr usize QKV_FAN_IN = config::qkv_fan_in;

	static_assert(config::d_model == config::n_heads * config::head_dim);

	using AF = AttnForward<HEAD_CNT, HEADS_PER_KERNEL, QK_DIM, V_DIM, D_MODEL, QKV_FAN_IN, V_EQUALS_K>;
	printf("sqrt 2 = %e, %e, %e, %d\n",
		math::constexpr_sqrt(2.0),
		M_SQRT2,
		math::constexpr_sqrt(2.0) - M_SQRT2,
		math::constexpr_sqrt(2.0) == M_SQRT2
	);
	constexpr f64 ONLINE_SOFTMAX_THRESHOLD = AF::ONLINE_SOFTMAX_THRESHOLD;
	constexpr f64 EXPB = math::fast::constexpr_expb(-ONLINE_SOFTMAX_THRESHOLD);
	printf("T = %e, expb(T) = %e\n", ONLINE_SOFTMAX_THRESHOLD, EXPB);

	if (SEQ_LEN % AF::Q_PER_BLOCK != 0) {
		printf("Expected n_inputs %% %u == 0\n", AF::Q_PER_BLOCK);
		return 1;
	}

	std::vector<bf16> h_Q = load_tensor(tensor_path(cli.input_dir, "q.bin"), SEQ_LEN, PACKED_DIM);
	std::vector<bf16> h_K = load_tensor(tensor_path(cli.input_dir, "k.bin"), SEQ_LEN, PACKED_DIM);
	std::vector<bf16> h_V = load_tensor(tensor_path(cli.input_dir, "v.bin"), SEQ_LEN, PACKED_DIM);
	std::vector<bf16> h_G = load_tensor(tensor_path(cli.input_dir, "g.bin"), SEQ_LEN, PACKED_DIM);
	std::vector<bf16> h_sink_v = load_tensor(torch_tensor_path("sinks_v.bin"), HEAD_CNT, V_DIM);
	std::vector<f32> h_sink_scores = load_f32_tensor(tensor_path(cli.input_dir, "sink_scores_f32.bin"), HEAD_CNT, SEQ_LEN);
	std::vector<f32> h_maxes;
	if (cli.use_torch_maxes && std::filesystem::exists(torch_tensor_path("attn_maxes_f32.bin"))) {
		h_maxes = load_f32_tensor(torch_tensor_path("attn_maxes_f32.bin"), HEAD_CNT, SEQ_LEN);
		if (h_maxes.empty()) {
			return 1;
		}
		printf("Loaded precomputed attention maxes\n");
	} else if (cli.use_torch_maxes) {
		printf("Torch attention maxes requested but %s does not exist\n", torch_tensor_path("attn_maxes_f32.bin").c_str());
	}
	if (h_Q.empty() || h_K.empty() || h_V.empty() || h_G.empty() || h_sink_v.empty() || h_sink_scores.empty()) {
		return 1;
	}

	std::vector<bf16> h_out(SEQ_LEN * PACKED_DIM);
	std::vector<f32> h_L(HEAD_CNT * SEQ_LEN);

	bf16 *d_Q = nullptr;
	bf16 *d_K = nullptr;
	bf16 *d_V = nullptr;
	bf16 *d_G = nullptr;
	bf16 *d_sink_v = nullptr;
	f32 *d_sink_scores = nullptr;
	f32 *d_maxes = nullptr;
	bf16 *d_out = nullptr;
	f32 *d_L = nullptr;

	cudaMalloc(&d_Q, h_Q.size() * sizeof(bf16));
	cudaMalloc(&d_K, h_K.size() * sizeof(bf16));
	cudaMalloc(&d_V, h_V.size() * sizeof(bf16));
	cudaMalloc(&d_G, h_G.size() * sizeof(bf16));
	cudaMalloc(&d_sink_v, h_sink_v.size() * sizeof(bf16));
	cudaMalloc(&d_sink_scores, h_sink_scores.size() * sizeof(f32));
	if (!h_maxes.empty()) {
		cudaMalloc(&d_maxes, h_maxes.size() * sizeof(f32));
	}
	cudaMalloc(&d_out, h_out.size() * sizeof(bf16));
	cudaMalloc(&d_L, h_L.size() * sizeof(f32));

	cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_G, h_G.data(), h_G.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sink_v, h_sink_v.data(), h_sink_v.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sink_scores, h_sink_scores.data(), h_sink_scores.size() * sizeof(f32), cudaMemcpyHostToDevice);
	if (d_maxes != nullptr) {
		cudaMemcpy(d_maxes, h_maxes.data(), h_maxes.size() * sizeof(f32), cudaMemcpyHostToDevice);
	}

	cudaFuncSetAttribute(attn_forward<AF>, cudaFuncAttributeMaxDynamicSharedMemorySize, AF::SMEM_BYTES);
	cudaFuncSetAttribute(attn_forward<AF>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	dim3 grid(SEQ_LEN / AF::Q_PER_BLOCK, AF::HEAD_GROUP_CNT);

	int warmup = 50;
	for (int i = 0; i < warmup; ++i) {
		attn_forward<AF>
			<<<grid, AF::THREADS_PER_BLOCK, AF::SMEM_BYTES>>>(
				SEQ_LEN,
				d_Q,
				d_K,
				d_V,
				d_G,
				d_sink_v,
				d_sink_scores,
				d_maxes,
				d_out,
				d_L,
				config::window_size
			);
	}
	cudaDeviceSynchronize();

	int num_runs = 100;
	std::vector<cudaEvent_t> starts(num_runs), ends(num_runs);
	for (int i = 0; i < num_runs; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < num_runs; ++i) {
		cudaEventRecord(starts[i]);
		attn_forward<AF>
			<<<grid, AF::THREADS_PER_BLOCK, AF::SMEM_BYTES>>>(
				SEQ_LEN,
				d_Q,
				d_K,
				d_V,
				d_G,
				d_sink_v,
				d_sink_scores,
				d_maxes,
				d_out,
				d_L,
				config::window_size
			);
		cudaEventRecord(ends[i]);
	}
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
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
	double tflops = AF::flops(SEQ_LEN, config::window_size) * HEAD_CNT / (median_ms * 1e-3) / 1e12;
	printf("Kernel time over %d runs: median %.3f ms  min %.3f ms\n", num_runs, median_ms, min_ms);
	printf("TFLOPS: %.2f\n", tflops);

	cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
	std::filesystem::create_directories("tmp/block_cuda");
	store_tensor("tmp/block_cuda/attn_out.bin", h_out, SEQ_LEN, PACKED_DIM);

	printf("Used SMEM per kernel: %d\n", AF::SMEM_BYTES);

	cudaFree(d_Q);
	cudaFree(d_K);
	cudaFree(d_V);
	cudaFree(d_G);
	cudaFree(d_sink_v);
	cudaFree(d_sink_scores);
	cudaFree(d_maxes);
	cudaFree(d_out);
	cudaFree(d_L);
	return 0;
}
