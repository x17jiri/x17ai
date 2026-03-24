#include "attn_forward.cuh"
#include "attn_d_q.cuh"
#include "attn_d_kv.cuh"

#include <vector>
#include <fstream>
#include <array>
#include <algorithm>

#pragma nv_diag_suppress 186

int main(int argc, char *argv[]) {
	constexpr usize NONROPE_DIM = 128;
	constexpr usize V_DIM = 64;
	constexpr usize ROPE_DIM = 0;
	constexpr usize QK_DIM = NONROPE_DIM + ROPE_DIM;
	{
		f32 diff = fabsf(sqrtf(QK_DIM) - f32(constexpr_sqrt(f64(QK_DIM))));
		printf("sqrtf=%e, constexpr_sqrt=%e, diff=%e\n",
			sqrtf(QK_DIM), f32(constexpr_sqrt(f64(QK_DIM))), diff);
		if (diff > 1e-8f) {
			return 1;
		}
	}
	bool use_real_data = argc <= 1;
	usize Q_LEN, KV_LEN;

	if (use_real_data) {
		Q_LEN = 1024;
		KV_LEN = 1024;
	} else {
		Q_LEN = 2*32768;
		KV_LEN = 2*32768;
	}
	srand(42);

	// allocate q: bf16 [Q_LEN, QK_DIM]
	std::vector<bf16> q_data(Q_LEN * QK_DIM);
	if (use_real_data) {
		std::ifstream in("tmp/q.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(q_data.data()),
			static_cast<std::streamsize>(q_data.size() * sizeof(*q_data.data()))
		);
	} else {
		for (size_t i = 0; i < q_data.size(); ++i) {
			q_data[i] = bf16(float(i));
			q_data[i] = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
		}
		// Save generated Q for Python verification
		std::ofstream q_out("tmp/large_q.bin", std::ios::binary);
		q_out.write(reinterpret_cast<char*>(q_data.data()),
			static_cast<std::streamsize>(q_data.size() * sizeof(bf16)));
	}
	bf16 *q_dev;
	cudaMalloc(&q_dev, q_data.size() * sizeof(bf16));
	cudaMemcpy(q_dev, q_data.data(), q_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(1) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	// allocate kv content: bf16 [KV_LEN, V_DIM]
	// allocate kv rope:    bf16 [KV_LEN, ROPE_DIM]
	std::vector<bf16> kv_data(KV_LEN * QK_DIM);
	if (use_real_data) {
		std::ifstream in("tmp/kv.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(kv_data.data()),
			static_cast<std::streamsize>(kv_data.size() * sizeof(*kv_data.data()))
		);
	} else {
		for (size_t i = 0; i < kv_data.size(); ++i) {
			kv_data[i] = bf16(float(i*100));
			kv_data[i] = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
		}
		// Save generated KV for Python verification
		std::ofstream kv_out("tmp/large_kv.bin", std::ios::binary);
		kv_out.write(reinterpret_cast<char*>(kv_data.data()),
			static_cast<std::streamsize>(kv_data.size() * sizeof(bf16)));
	}
	// Split interleaved [KV_LEN, QK_DIM] into separate K content, K rope, and V arrays
	std::vector<bf16> kc_data(KV_LEN * NONROPE_DIM);
	std::vector<bf16> kr_data(KV_LEN * ROPE_DIM);
	std::vector<bf16> v_data(KV_LEN * V_DIM);
	for (size_t r = 0; r < KV_LEN; r++) {
		for (size_t c = 0; c < NONROPE_DIM; c++) {
			kc_data[r * NONROPE_DIM + c] = kv_data[r * QK_DIM + c];
		}
		for (size_t c = 0; c < V_DIM; c++) {
			v_data[r * V_DIM + c] = kv_data[r * QK_DIM + c];
		}
		if constexpr (ROPE_DIM > 0) {
			for (size_t c = 0; c < ROPE_DIM; c++) {
				kr_data[r * ROPE_DIM + c] = kv_data[r * QK_DIM + NONROPE_DIM + c];
			}
		}
	}
	bf16 *kc_dev, *kr_dev, *v_dev;
	cudaMalloc(&kc_dev, kc_data.size() * sizeof(bf16));
	cudaMemcpy(kc_dev, kc_data.data(), kc_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMalloc(&kr_dev, kr_data.size() * sizeof(bf16));
	cudaMemcpy(kr_dev, kr_data.data(), kr_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMalloc(&v_dev, v_data.size() * sizeof(bf16));
	cudaMemcpy(v_dev, v_data.data(), v_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	// allocate dO: bf16 [Q_LEN, V_DIM]
	std::vector<bf16> dO_data(Q_LEN * V_DIM);
	if (use_real_data) {
		std::ifstream in("tmp/dO.bin", std::ios::binary);
		in.read(
			reinterpret_cast<char*>(dO_data.data()),
			static_cast<std::streamsize>(dO_data.size() * sizeof(*dO_data.data()))
		);
	} else {
		for (size_t i = 0; i < dO_data.size(); ++i) {
			dO_data[i] = bf16(float(rand()) / RAND_MAX * 2.0f - 1.0f);
		}
		std::ofstream dO_out("tmp/large_dO.bin", std::ios::binary);
		dO_out.write(reinterpret_cast<char*>(dO_data.data()),
			static_cast<std::streamsize>(dO_data.size() * sizeof(bf16)));
	}
	bf16 *dO_dev;
	cudaMalloc(&dO_dev, dO_data.size() * sizeof(bf16));
	cudaMemcpy(dO_dev, dO_data.data(), dO_data.size() * sizeof(bf16), cudaMemcpyHostToDevice);

	// allocate output: bf16 [Q_LEN, V_DIM]
	std::vector<bf16> out_data(Q_LEN * V_DIM);
	bf16 *out_dev;
	size_t out_size_bytes = out_data.size() * sizeof(bf16);
	cudaMalloc(&out_dev, out_size_bytes);
	GMatrixDynSize<bf16, V_DIM> out{out_dev, Q_LEN};

	// allocate logsumexp: f32 [Q_LEN]
	std::vector<f32> L_data(Q_LEN);
	f32 *L_dev;
	cudaMalloc(&L_dev, Q_LEN * sizeof(f32));

	// allocate dQ output: bf16 [Q_LEN, QK_DIM]
	bf16 *dQ_dev;
	cudaMalloc(&dQ_dev, Q_LEN * QK_DIM * sizeof(bf16));

	// allocate D output: f32 [Q_LEN]
	f32 *D_dev;
	cudaMalloc(&D_dev, Q_LEN * sizeof(f32));

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(2) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(3) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	constexpr bool V_EQ_K = true;
	using AF = Attn_forward<1, NONROPE_DIM, ROPE_DIM, V_DIM, 4, 1, V_EQ_K>;
	using ADQ = Attn_d_q<AF>;
	using ADKV = Attn_d_kv<AF>;
	usize smem_size = AF::SMEM_BYTES;
	printf("smem_size = %d bytes (forward), %d bytes (dQ), %d bytes (dKV)\n", smem_size, ADQ::SMEM_BYTES, ADKV::SMEM_BYTES);
	//smem_size = std::max(smem_size, usize(70 * 1024));

	cudaFuncSetAttribute(attn_forward<AF>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
	cudaFuncSetAttribute(attn_d_q<ADQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, ADQ::SMEM_BYTES);
	cudaFuncSetAttribute(attn_d_kv<ADKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, ADKV::SMEM_BYTES);

	cudaFuncSetAttribute(attn_forward<AF>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

	// Allocate sink+gate buffer: [sink_score, gate]
	f32 sink_host[2] = { -0.3f, 0.5f };
	f32 *sink_dev;
	cudaMalloc(&sink_dev, sizeof(sink_host));
	cudaMemcpy(sink_dev, sink_host, sizeof(sink_host), cudaMemcpyHostToDevice);
	f32 *sink_ptr = use_real_data ? sink_dev : nullptr;

	cudaDeviceSynchronize();

	int WARMUP = use_real_data ? 0 : 50;
	//WARMUP = 0;
	for (int i = 0; i < WARMUP; ++i) {
		attn_forward<AF>
			<<<Q_LEN / AF::Q_PER_BLOCK, AF::THREADS_PER_BLOCK, smem_size>>>
			(
				Q_LEN, q_dev,
				KV_LEN, kc_dev, kr_dev, v_dev,
				out_dev,
				L_dev,
				sink_ptr
			);
	}

	cudaDeviceSynchronize();

	usize NUM_BLOCKS = Q_LEN / AF::Q_PER_BLOCK;
	int NUM_RUNS = use_real_data ? 1 : 200;
	//NUM_RUNS = 1;
	std::vector<cudaEvent_t> starts(NUM_RUNS), ends(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventCreate(&starts[i]);
		cudaEventCreate(&ends[i]);
	}
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventRecord(starts[i]);
		attn_forward<AF>
			<<<NUM_BLOCKS, AF::THREADS_PER_BLOCK, smem_size>>>
			(
				Q_LEN, q_dev,
				KV_LEN, kc_dev, kr_dev, v_dev,
				out_dev,
				L_dev,
				sink_ptr
			);
		cudaEventRecord(ends[i]);
	}
	cudaDeviceSynchronize();

	std::vector<float> times_ms(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventElapsedTime(&times_ms[i], starts[i], ends[i]);
		cudaEventDestroy(starts[i]);
		cudaEventDestroy(ends[i]);
	}
	std::sort(times_ms.begin(), times_ms.end());

	int mid = NUM_RUNS / 2;
	float median_ms = times_ms[mid];
	float min_ms = times_ms[0];
	//printf("Kernel time over %d runs: median %.4f ms  min %.4f ms\n", NUM_RUNS, median_ms, min_ms);

	// TFLOPS: Q@K^T = 2*Q*KV*QK_DIM, attn@V = 2*Q*KV*V_DIM, softmax ~ 5*Q*KV
	// Causal ≈ half the work
	double flops_causal = (2.0 * Q_LEN * KV_LEN * QK_DIM + 2.0 * Q_LEN * KV_LEN * V_DIM + 5.0 * Q_LEN * KV_LEN) / 2.0;
	printf("TFLOPS (causal): %.2f\n", flops_causal / (median_ms * 1e-3) / 1e12);

	// write output to file
	{
		std::ofstream out_file("tmp/out_cpu.bin", std::ios::binary);
		cudaMemcpy(out_data.data(), out_dev, out_size_bytes, cudaMemcpyDeviceToHost);
		out_file.write(
			reinterpret_cast<char *>(out_data.data()),
			static_cast<std::streamsize>(out_data.size() * sizeof(*out_data.data()))
		);
	}

	// write logsumexp to file
	{
		std::ofstream L_file("tmp/L.bin", std::ios::binary);
		cudaMemcpy(L_data.data(), L_dev, Q_LEN * sizeof(f32), cudaMemcpyDeviceToHost);
		L_file.write(
			reinterpret_cast<char *>(L_data.data()),
			static_cast<std::streamsize>(L_data.size() * sizeof(f32))
		);
	}

	// Run d_q backward kernel (with optional benchmarking)
	for (int i = 0; i < WARMUP; ++i) {
		attn_d_q<ADQ>
			<<<NUM_BLOCKS, ADQ::THREADS_PER_BLOCK, ADQ::SMEM_BYTES>>>
			(
				Q_LEN, q_dev,
				KV_LEN, kc_dev, kr_dev, v_dev,
				out_dev, dO_dev, dQ_dev,
				L_dev, D_dev,
				sink_ptr
			);
	}
	cudaDeviceSynchronize();

	std::vector<cudaEvent_t> dq_starts(NUM_RUNS), dq_ends(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventCreate(&dq_starts[i]);
		cudaEventCreate(&dq_ends[i]);
	}
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventRecord(dq_starts[i]);
		attn_d_q<ADQ>
			<<<NUM_BLOCKS, ADQ::THREADS_PER_BLOCK, ADQ::SMEM_BYTES>>>
			(
				Q_LEN, q_dev,
				KV_LEN, kc_dev, kr_dev, v_dev,
				out_dev, dO_dev, dQ_dev,
				L_dev, D_dev,
				sink_ptr
			);
		cudaEventRecord(dq_ends[i]);
	}
	cudaDeviceSynchronize();

	std::vector<float> dq_times_ms(NUM_RUNS);
	for (int i = 0; i < NUM_RUNS; ++i) {
		cudaEventElapsedTime(&dq_times_ms[i], dq_starts[i], dq_ends[i]);
		cudaEventDestroy(dq_starts[i]);
		cudaEventDestroy(dq_ends[i]);
	}
	std::sort(dq_times_ms.begin(), dq_times_ms.end());
	float dq_median_ms = dq_times_ms[NUM_RUNS / 2];

	// dQ FLOPS: Q@K^T + dO@V^T + dS@K = 2*Q*KV*(QK_DIM + V_DIM + QK_DIM), causal ≈ half
	double dq_flops_causal = (2.0 * Q_LEN * KV_LEN * (QK_DIM + V_DIM + QK_DIM) + 5.0 * Q_LEN * KV_LEN) / 2.0;
	printf("TFLOPS dQ (causal): %.2f\n", dq_flops_causal / (dq_median_ms * 1e-3) / 1e12);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(dQ) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	// write dQ to file
	{
		std::vector<bf16> dQ_data(Q_LEN * QK_DIM);
		cudaMemcpy(dQ_data.data(), dQ_dev, dQ_data.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
		std::ofstream f("tmp/dQ.bin", std::ios::binary);
		f.write(reinterpret_cast<char*>(dQ_data.data()),
			static_cast<std::streamsize>(dQ_data.size() * sizeof(bf16)));
	}

	// write D to file
	{
		std::vector<f32> D_data(Q_LEN);
		cudaMemcpy(D_data.data(), D_dev, Q_LEN * sizeof(f32), cudaMemcpyDeviceToHost);
		std::ofstream f("tmp/D.bin", std::ios::binary);
		f.write(reinterpret_cast<char*>(D_data.data()),
			static_cast<std::streamsize>(Q_LEN * sizeof(f32)));
	}

	// write dO to file (for Python verification)
	{
		std::ofstream f(use_real_data ? "tmp/dO.bin" : "tmp/large_dO.bin", std::ios::binary);
		f.write(reinterpret_cast<char*>(dO_data.data()),
			static_cast<std::streamsize>(dO_data.size() * sizeof(bf16)));
	}

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("(4) CUDA Error: %s\n", cudaGetErrorString(err));
	}

	// Print first 8 rows, first 8 cols
	printf("\nFirst 8 rows, first 8 cols:\n");
	for (size_t r = 0; r < 8; r++) {
		for (size_t c = 0; c < 8; c++) {
			printf("%12.6e ", double(float(out_data[r * V_DIM + c])));
		}
		printf("\n");
	}

	// Print last 8 rows, last 8 cols
	printf("\nLast 8 rows, last 8 cols:\n");
	for (size_t r = Q_LEN - 8; r < Q_LEN; r++) {
		for (size_t c = V_DIM - 8; c < V_DIM; c++) {
			printf("%12.6e ", double(float(out_data[r * V_DIM + c])));
		}
		printf("\n");
	}

	// Check for non-finite values
	size_t nan_count = 0;
	size_t inf_count = 0;
	std::vector<size_t> nan_rows;
	std::vector<size_t> inf_rows;
	for (size_t i = 0; i < out_data.size(); ++i) {
		float v = float(out_data[i]);
		if (!isfinite(v)) {
			if (isnan(v)) {
				nan_count++;
				nan_rows.push_back(i);
			} else {
				inf_count++;
				inf_rows.push_back(i);
			}
		}
	}
	if (inf_count > 0 || nan_count > 0) {
		printf("\n*** WARNING: output contains %zu infinite values and %zu NaNs ***\n", inf_count, nan_count);
		if (nan_count > 0) {
			printf("NaN rows (up to 100): ");
			for (size_t i = 0; i < std::min(nan_rows.size(), size_t(100)); ++i) {
				printf("%zu ", nan_rows[i]);
			}
			printf("\n");
		}
		if (inf_count > 0) {
			printf("Inf rows (up to 100): ");
			for (size_t i = 0; i < std::min(inf_rows.size(), size_t(100)); ++i) {
				printf("%zu ", inf_rows[i]);
			}
			printf("\n");
		}
	}

	return 0;
}
