/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_atom.hpp"

#include <cstddef>
#include <span>
#include <thread>
#include <vector>
#include <barrier>
#include <mutex>
#include <iostream>
#include <array>
#include <fstream>

using MmaOp = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using MmaOpTraits = cute::MMA_Traits<MmaOp>;
using MmaAtom = cute::MMA_Atom<MmaOp>;
using half_t = cute::half_t;

constexpr size_t QK_DIM = 192;
constexpr size_t V_DIM = 128;

constexpr size_t Q_PER_BLOCK = 128;
constexpr size_t Q_PER_WARP = 8;
constexpr size_t KV_PER_STEP = 16;
constexpr size_t FEATURE_TILE = 16;
constexpr size_t KV_PRELOAD = 8;

using ICount = std::common_type_t<
	std::make_signed_t<std::ptrdiff_t>,
	std::make_signed_t<std::size_t>
>;
using UCount = std::make_unsigned_t<ICount>;

using f16 = __nv_bfloat16;
using f32 = float;

template<const ICount V>
struct ConstExtent {
	inline constexpr ICount value() const noexcept {
		return V;
	}
};

struct Extent {
	ICount v;
	inline constexpr ICount value() const noexcept {
		return v;
	}
};

enum StrideType {
	RowMajor,
	ColumnMajor
};

template<typename T>
struct DataPtr {
	std::span<T> __data;
	size_t __offset;

	DataPtr(std::span<T> data_)
		: __data{data_}
		, __offset{0}
	{}

	DataPtr(std::span<T> data_, size_t offset_)
		: __data{data_}
		, __offset{offset_}
	{}

	std::span<T> data() const {
		return __data.subspan(__offset);
	}

	T get(size_t index) const {
		assert(is_valid_index(index));
		return __data[__offset + index];
	}

	void set(size_t index, T value) const {
		assert(is_valid_index(index));
		__data[__offset + index] = value;
	}

	bool is_valid_index(size_t index) const {
		return __offset + index < __data.size();
	}
};

template<typename T>
struct GPtr: DataPtr<T> {
	using DataPtr<T>::DataPtr;

	GPtr with_offset(size_t offset) const {
		return {
			this->__data,
			this->__offset + offset
		};
	}
};

template<typename T>
struct SPtr: DataPtr<T> {
	using DataPtr<T>::DataPtr;

	SPtr with_offset(size_t offset) const {
		return {
			this->__data,
			this->__offset + offset
		};
	}
};

template<typename T, const UCount N>
struct RData {
	RData(RData const &other) = delete;
	RData &operator=(RData const &other) = delete;
	RData(RData &&other) = delete;
	RData &operator=(RData &&other) = delete;

	std::array<T, N> __data;

	RData()
		: __data{}
	{}

	std::span<T> data() const {
		return std::span<T>(__data.data(), N);
	}

	T get(size_t index) const {
		assert(is_valid_index(index));
		return __data[index];
	}

	void set(size_t index, T value) {
		assert(is_valid_index(index));
		__data[index] = value;
	}

	bool is_valid_index(size_t index) const {
		return index < __data.size();
	}
};

template<
	typename Data,
	const ICount M, // number of rows
	const ICount N, // number of columns
	const StrideType S = RowMajor
>
struct Matrix {
	Data data;

	using MExtent = std::conditional_t<
		(M >= 0),
		ConstExtent<M>,
		Extent
	>;
	using NExtent = std::conditional_t<
		(N >= 0),
		ConstExtent<N>,
		Extent
	>;

	std::tuple<MExtent, NExtent, ICount> layout;

	Matrix()
	requires(M >= 0 && N >= 0):
		data{{}, 0},
		layout(
			MExtent{},
			NExtent{},
			(S == RowMajor) ? N : M
		)
	{}

	size_t __max_index() const {
		if (stride_type() == RowMajor) {
			return static_cast<size_t>((m_rows() - 1) * stride() + n_cols() - 1);
		} else {
			return static_cast<size_t>((n_cols() - 1) * stride() + m_rows() - 1);
		}
	}

	Matrix(Data data_, ICount row_stride)
	requires(M >= 0 && N >= 0 && S == RowMajor):
		data(data_),
		layout(
			MExtent{},
			NExtent{},
			row_stride
		)
	{
		assert(row_stride >= N);
		assert(data_.is_valid_index(__max_index()));
	}

	Matrix(Data data_, ICount col_stride)
	requires(M >= 0 && N >= 0 && S == ColumnMajor):
		data(data_),
		layout(
			MExtent{},
			NExtent{},
			col_stride
		)
	{
		assert(col_stride >= M);
		assert(data_.is_valid_index(__max_index()));
	}

	Matrix(ICount stride)
	requires(M >= 0 && N >= 0):
		data(),
		layout(
			MExtent{},
			NExtent{},
			stride
		)
	{
		if (S == RowMajor) {
			assert(stride >= N);
		} else {
			assert(stride >= M);
		}
		assert(data.is_valid_index(__max_index()));
	}

	Matrix(Data data_, ICount m_rows, ICount row_stride)
	requires(M < 0 && N >= 0 && S == RowMajor):
		data(data_),
		layout(
			MExtent{m_rows},
			NExtent{},
			row_stride
		)
	{
		assert(row_stride >= N);
		assert(data_.is_valid_index(__max_index()));
	}

	Matrix(Data data_, ICount n_cols, ICount col_stride)
	requires(M >= 0 && N < 0 && S == ColumnMajor):
		data(data_),
		layout(
			MExtent{},
			NExtent{n_cols},
			col_stride
		)
	{
		assert(col_stride >= M);
		assert(data_.is_valid_index(__max_index()));
	}

	inline ICount m_rows() const {
		return std::get<0>(layout).value();
	}

	inline ICount n_cols() const {
		return std::get<1>(layout).value();
	}

	inline ICount stride() const {
		return std::get<2>(layout);
	}

	inline StrideType stride_type() const {
		return S;
	}

	Matrix<Data, N, M, ColumnMajor> transpose() const
	requires(M >= 0 && N >= 0 && S == RowMajor) {
		return Matrix<Data, N, M, ColumnMajor>{
			data,
			stride()
		};
	}

	template<const ICount M_TILE, const ICount N_TILE>
	Matrix<Data, M_TILE, N_TILE, S> tile(ICount m_tile_idx, ICount n_tile_idx) const
	requires(M_TILE >= 0 && N_TILE >= 0 && M >= 0 && N >= 0 && M % M_TILE == 0 && N % N_TILE == 0) {
		assert(m_tile_idx >= 0 && m_tile_idx < (m_rows() / M_TILE));
		assert(n_tile_idx >= 0 && n_tile_idx < (n_cols() / N_TILE));
		return Matrix<Data, M_TILE, N_TILE, S>{
			data.with_offset(S == RowMajor
				? m_tile_idx * M_TILE * stride() + n_tile_idx * N_TILE
				: n_tile_idx * N_TILE * stride() + m_tile_idx * M_TILE
			),
			stride()
		};
	}

	template<const ICount M_TILE>
	Matrix<Data, M_TILE, N, S> tile_m(ICount m_tile_idx) const
	requires(M_TILE >= 0 && M >= 0 && N >= 0 && M % M_TILE == 0) {
		assert(m_tile_idx >= 0 && m_tile_idx < (m_rows() / M_TILE));
		return Matrix<Data, M_TILE, N, S>{
			data.with_offset(S == RowMajor
				? m_tile_idx * M_TILE * stride()
				: m_tile_idx * M_TILE
			),
			stride()
		};
	}

	template<const ICount M_TILE>
	Matrix<Data, M_TILE, N, S> tile_m(ICount m_tile_idx) const
	requires(M_TILE >= 0 && M < 0 && N >= 0) {
		assert(m_tile_idx >= 0 && m_tile_idx < (m_rows() / M_TILE));
		assert(m_rows() % M_TILE == 0);
		return Matrix<Data, M_TILE, N, S>{
			data.with_offset(S == RowMajor
				? m_tile_idx * M_TILE * stride()
				: m_tile_idx * M_TILE
			),
			stride()
		};
	}

	template<const ICount M_TILE, const ICount N_TILE>
	Matrix<Data, M_TILE, N_TILE, RowMajor> tile(ICount m_tile_idx, ICount n_tile_idx) const
	requires(M_TILE >= 0 && N_TILE >= 0 && M < 0 && N >= 0 && S == RowMajor && N % N_TILE == 0) {
		assert(m_rows() % M_TILE == 0);
		assert(m_tile_idx >= 0 && m_tile_idx < (m_rows() / M_TILE));
		assert(n_tile_idx >= 0 && n_tile_idx < (n_cols() / N_TILE));
		return Matrix<Data, M_TILE, N_TILE, RowMajor>{
			data.with_offset(m_tile_idx * M_TILE * stride() + n_tile_idx * N_TILE),
			stride()
		};
	}

	template<const ICount M_TILE>
	Matrix<Data, M_TILE, N, RowMajor> tile(ICount m_tile_idx) const
	requires(M_TILE >= 0 && M < 0 && N >= 0 && S == RowMajor) {
		assert(m_rows() % M_TILE == 0);
		assert(m_tile_idx >= 0 && m_tile_idx < (m_rows() / M_TILE));
		return Matrix<Data, M_TILE, N, RowMajor>{
			data.with_offset(m_tile_idx * M_TILE * stride()),
			stride()
		};
	}

	template<const ICount N_TILE>
	Matrix<Data, M, N_TILE, S> tile_n(ICount n_tile_idx) const
	requires(N_TILE >= 0 && M >= 0 && N >= 0 && N % N_TILE == 0) {
		assert(n_tile_idx >= 0 && n_tile_idx < (n_cols() / N_TILE));
		return Matrix<Data, M, N_TILE, S>{
			data.with_offset(n_tile_idx * N_TILE),
			stride()
		};
	}

	auto get(ICount row, ICount col) const -> decltype(data.get(0)) {
		assert(row >= 0 && row < m_rows());
		assert(col >= 0 && col < n_cols());
		if (stride_type() == RowMajor) {
			return data.get(row * stride() + col);
		} else {
			return data.get(col * stride() + row);
		}
	}

	void set(ICount row, ICount col, decltype(data.get(0)) value) const {
		assert(row >= 0 && row < m_rows());
		assert(col >= 0 && col < n_cols());
		if (stride_type() == RowMajor) {
			data.set(row * stride() + col, value);
		} else {
			data.set(col * stride() + row, value);
		}
	}

	void fill_(decltype(data.get(0)) value) {
		for (ICount m = 0; m < m_rows(); ++m) {
			for (ICount n = 0; n < n_cols(); ++n) {
				set(m, n, value);
			}
		}
	}

	void zero_() {
		fill_(decltype(data.get(0))());
	}
};

template<
	typename T,
	const ICount M, // number of rows
	const ICount N, // number of columns
	const StrideType S = RowMajor
>
struct RMatrix {
	RData<T, M*N> data;

	RMatrix() {}

	auto get(ICount row, ICount col) const -> decltype(data.get(0)) {
		assert(row >= 0 && row < M);
		assert(col >= 0 && col < N);
		if (S == RowMajor) {
			return data.get(row * N + col);
		} else {
			return data.get(col * M + row);
		}
	}

	void set(ICount row, ICount col, decltype(data.get(0)) value) {
		assert(row >= 0 && row < M);
		assert(col >= 0 && col < N);
		if (S == RowMajor) {
			data.set(row * N + col, value);
		} else {
			data.set(col * M + row, value);
		}
	}

	void zero_() {
		for (ICount m = 0; m < M; ++m) {
			for (ICount n = 0; n < N; ++n) {
				set(m, n, T());
			}
		}
	}

	void read_from_sram(Matrix<SPtr<T>, M, N, S> const &src) {
		for (ICount m = 0; m < M; ++m) {
			for (ICount n = 0; n < N; ++n) {
				set(m, n, src.get(m, n));
			}
		}
	}
};

template<typename T, typename U, const ICount M, const ICount N, const StrideType S>
void downcast(RMatrix<T, M, N, S> const &src, RMatrix<U, M, N, S> &dst) {
	for (ICount m = 0; m < M; ++m) {
		for (ICount n = 0; n < N; ++n) {
			dst.set(m, n, static_cast<U>(src.get(m, n)));
		}
	}
}

template<typename T, typename U, const ICount M, const ICount N>
void downcast_store(
	RMatrix<T, N, M, ColumnMajor> const &src,
	Matrix<GPtr<U>, M, N, RowMajor> const &dst
) {
	for (ICount m = 0; m < M; ++m) {
		for (ICount n = 0; n < N; ++n) {
			dst.set(m, n, static_cast<U>(src.get(n, m)));
		}
	}
}

// implement things  manually on the CPU just for testing
namespace cpu_test {
	template<const StrideType S>
	void tiny_gemm(
		Matrix<SPtr<f16>, 16, 16, S> const &a,
		RMatrix<f16, 16, 8, ColumnMajor> const &b,
		RMatrix<f32, 16, 8, ColumnMajor> &c,
		bool debug = false
	) {
		for (size_t m = 0; m < 16; ++m) {
			for (size_t n = 0; n < 8; ++n) {
				f32 sum = 0.0f;
				for (size_t k = 0; k < 16; ++k) {
					sum += static_cast<f32>(a.get(m, k)) * static_cast<f32>(b.get(k, n));
					if (m == 0 && n == 0 && debug) {
						std::cout << "a[" << m << "," << k << "] = " << f32(a.get(m, k)) << "\n";
						std::cout << "b[" << k << "," << n << "] = " << f32(b.get(k, n)) << "\n";
						std::cout << "partial sum = " << (c.get(m, n) + sum) << "\n";
					}
				}
				c.set(m, n, c.get(m, n) + sum);
			}
		}
	}

	struct BlockShared {
		std::vector<f16> q_sram;
		std::vector<std::vector<f16>> kv_sram;
		std::barrier<> barrier;
		std::mutex *log_mutex;

		BlockShared(
			size_t q_sram_count,
			size_t kv_sram_count,
			size_t kv_preload,
			size_t warp_count,
			std::mutex *log_mutex
		)
			: q_sram(q_sram_count)
			, kv_sram()
			, barrier(warp_count)
			, log_mutex(log_mutex)
		{
			kv_sram.reserve(kv_preload);
			for (size_t i = 0; i < kv_preload; ++i) {
				kv_sram.emplace_back(kv_sram_count);
			}
			std::cerr << "Allocated BlockShared with "
					  << (q_sram_count + kv_sram_count * kv_preload) * sizeof(f16) / 1024
					  << " KB SRAM, for " << warp_count << " warps ("
					  << (warp_count * 32) << " threads)." << std::endl;
		}

		void syncthreads() {
			barrier.arrive_and_wait();
		}

		void cp_async_wait_all() {
			// no-op on CPU
		}

		void cp_async_wait_group(size_t count) {
			// no-op on CPU
		}

		void cp_async_commit() {
			// no-op on CPU
		}

		std::lock_guard<std::mutex> lock_log() {
			return std::lock_guard<std::mutex>(*log_mutex);
		}
	};

	#define X17_UNROLL

	struct Dim3X {
		size_t x;
		constexpr Dim3X(size_t x_) : x{x_} {}
	};

	constexpr Dim3X block_warps_dim{Q_PER_BLOCK / Q_PER_WARP};
	thread_local Dim3X block_idx{size_t(-1)};
	thread_local Dim3X warp_idx{size_t(-1)};
	thread_local BlockShared *shared = nullptr;

	bool is_first_warp() {
		return warp_idx.x == 0;
	}

	Matrix<SPtr<f16>, Q_PER_WARP, QK_DIM> copy_q_to_sram_async(
		Matrix<GPtr<f16>, -1, QK_DIM> const &gQ
	) {
		// Part of Q for this block
		Matrix<GPtr<f16>, Q_PER_BLOCK, QK_DIM> gQ_tile = gQ.tile_m<Q_PER_BLOCK>(block_idx.x);
		// Part of Q for this warp
		Matrix<GPtr<f16>, Q_PER_WARP, QK_DIM> gQ_warp_tile = gQ_tile.tile_m<Q_PER_WARP>(warp_idx.x);
		// SRAM storage for Q tile for this warp
		Matrix<SPtr<f16>, Q_PER_BLOCK, QK_DIM> sQ{{shared->q_sram}, QK_DIM};
		// View of SRAM for this warp
		Matrix<SPtr<f16>, Q_PER_WARP, QK_DIM> sQ_warp_tile = sQ.tile_m<Q_PER_WARP>(warp_idx.x);

		// **Important**: We call cp.async to copy Qs from GMEM into SMEM.
		// It is important that each warp copies its own Qs - the ones it will use later.
		// This way when we later use `cp.async.wait_group`/`cp.async.wait_all`, all threads
		// in a warp wait in a lockstep and so we know the inputs are ready without __syncthreads().
		for (ICount m = 0; m < Q_PER_WARP; ++m) {
			for (ICount n = 0; n < QK_DIM; ++n) {
				sQ_warp_tile.set(m, n, gQ_warp_tile.get(m, n));
			}
		}

		return sQ_warp_tile;
	}

	void copy_kv_to_sram_async(
		Matrix<GPtr<f16>, -1, QK_DIM> gKV,
		size_t kv_step
	) {
		if (kv_step * KV_PER_STEP < gKV.m_rows()) {
			size_t p = kv_step % KV_PRELOAD;
			// SRAM storage for KV tile for this step
			Matrix<SPtr<f16>, KV_PER_STEP, QK_DIM> sKV_tile{{shared->kv_sram[p]}, QK_DIM};
			// View of KV tile for this step
			Matrix<GPtr<f16>, KV_PER_STEP, QK_DIM> gKV_tile = gKV.tile_m<KV_PER_STEP>(kv_step);
			// On GPU, warps will need to cooperate to collectively load the KV tile into SRAM
			if (is_first_warp()) {
				for (ICount m = 0; m < KV_PER_STEP; ++m) {
					for (ICount n = 0; n < QK_DIM; ++n) {
						sKV_tile.set(m, n, gKV_tile.get(m, n));
					}
				}
			}
		}
	}

	Matrix<SPtr<f16>, KV_PER_STEP, QK_DIM> get_kv_in_sram(
		size_t kv_step
	) {
		size_t p = kv_step % KV_PRELOAD;
		// SRAM storage for KV tile for this step
		Matrix<SPtr<f16>, KV_PER_STEP, QK_DIM> sKV_tile{{shared->kv_sram[p]}, QK_DIM};
		return sKV_tile;
	}

	// this function corresponds to one warp on GPU
	void attn_kernel(
		Matrix<GPtr<f16>, -1, QK_DIM> const &gQ,
		Matrix<GPtr<f16>, -1, QK_DIM> const &gKV,
		Matrix<GPtr<f16>, -1, V_DIM> const &gOut
	) {
		// Q tile for this warp
		auto sQ_tile = copy_q_to_sram_async(gQ);

		// Registers
		std::array<f32, Q_PER_WARP> max_score;
		max_score.fill(-std::numeric_limits<f32>::infinity());

		std::array<f32, Q_PER_WARP> score_sum;
		score_sum.fill(0.0);

		RMatrix<f16, KV_PER_STEP, Q_PER_WARP, ColumnMajor> rScores;

		std::array<f32, Q_PER_WARP> rRescale;

		std::array<
			RMatrix<f32, FEATURE_TILE, Q_PER_WARP, ColumnMajor>,
			V_DIM / FEATURE_TILE
		> rO;
		X17_UNROLL for (auto &rO_tile: rO) {
			rO_tile.zero_();
		}

		std::array<RMatrix<f16, FEATURE_TILE, Q_PER_WARP, ColumnMajor>, 2> rQ;

		for (size_t p = 0; p < KV_PRELOAD - 1; ++p) {
			copy_kv_to_sram_async(gKV, p);
			shared->cp_async_commit();
		}

		shared->cp_async_wait_group(KV_PRELOAD - 2);
		rQ[0].read_from_sram(sQ_tile.tile_n<FEATURE_TILE>(0).transpose());
		shared->syncthreads();

		// Iterate over KV
		size_t kv_len = gKV.m_rows();
		for (size_t kv_step = 0; kv_step < kv_len / KV_PER_STEP; ++kv_step) {
			// Preload next KV tile
			copy_kv_to_sram_async(gKV, kv_step + KV_PRELOAD - 1);
			shared->cp_async_commit();

			// KV tile for this step
			auto sKV_tile = get_kv_in_sram(kv_step);

			// Tile both `K` and `Q` along the feature dimension and accumulate gemm.
			// This will result in `rScores = K * Q^T`
			RMatrix<f32, KV_PER_STEP, Q_PER_WARP, ColumnMajor> rScores_f32;
			rScores_f32.zero_();
			X17_UNROLL for (size_t f_step = 0; f_step < QK_DIM / FEATURE_TILE; ++f_step) {
				rQ[(f_step + 1) % 2].read_from_sram(
					sQ_tile.tile_n<FEATURE_TILE>((f_step + 1) % (QK_DIM / FEATURE_TILE)).transpose()
				);
				tiny_gemm(
					sKV_tile.tile_n<FEATURE_TILE>(f_step),
					rQ[f_step % 2],
					rScores_f32
				);
			}
			downcast(rScores_f32, rScores);

			// Update max and sum_exp
			for (size_t i = 0; i < Q_PER_WARP; ++i) {
				// find max
				f32 old_max = max_score[i];
				f32 new_max = old_max;
				for (size_t j = 0; j < KV_PER_STEP; ++j) {
					new_max = std::max(new_max, f32(rScores.get(j, i)));
				}
				max_score[i] = new_max;

				// coefficient for sum_exp scaling
				rRescale[i] = std::exp(old_max - new_max);

				// compute sum_exp
				f32 sum_exp = 0.0f;
				for (size_t j = 0; j < KV_PER_STEP; ++j) {
					f16 score = static_cast<f16>(
						std::exp(f32(rScores.get(j, i)) - new_max)
					);
					rScores.set(j, i, score);
					sum_exp += f32(score);
				}
				f32 sum_exp_old = score_sum[i];
				score_sum[i] *= rRescale[i];
				score_sum[i] += sum_exp;
			}

			// rescale output accumulators
			X17_UNROLL for (auto &rO_tile: rO) {
				for (size_t j = 0; j < Q_PER_WARP; ++j) {
					for (size_t i = 0; i < FEATURE_TILE; ++i) {
						rO_tile.set(i, j, rO_tile.get(i, j) * rRescale[j]);
					}
				}
			}

			// compute `rO += KV_tile * rScores`
			size_t v_tile = 0;
			X17_UNROLL for (auto &rO_tile: rO) {
				if (v_tile < rO.size() - 1) {
					// TODO - preload kv.T to regs
				} else {
					shared->cp_async_wait_group(KV_PRELOAD - 2);
					shared->syncthreads();
					// TODO - preload kv to regs
				}
				tiny_gemm(
					sKV_tile.tile_n<FEATURE_TILE>(v_tile).transpose(),
					rScores,
					rO_tile
				);
				++v_tile;
			}
		}

		// finalize output by normalizing with sum_exp
		X17_UNROLL for (auto &rO_tile: rO) {
			for (size_t j = 0; j < Q_PER_WARP; ++j) {
				f32 inv_sum_exp = 1.0f / score_sum[j];
				for (size_t i = 0; i < FEATURE_TILE; ++i) {
					rO_tile.set(i, j, rO_tile.get(i, j) * inv_sum_exp);
				}
			}
		}

		// Part of O for this block
		Matrix<GPtr<f16>, Q_PER_BLOCK, V_DIM> gOut_tile = gOut.tile_m<Q_PER_BLOCK>(block_idx.x);
		// View rows of O tile for this warp
		Matrix<GPtr<f16>, Q_PER_WARP, V_DIM> gOut_warp_tile = gOut_tile.tile_m<Q_PER_WARP>(warp_idx.x);
		// Write output
		size_t v_tile = 0;
		X17_UNROLL for (auto &rO_tile: rO) {
			downcast_store(rO_tile, gOut_warp_tile.tile_n<FEATURE_TILE>(v_tile));
			++v_tile;
		}
	}

	void start_attn_kernel(
		Dim3X grid_dim,
		size_t q_sram_count,
		size_t kv_sram_count,
		size_t kv_preload,
		Matrix<GPtr<f16>, -1, QK_DIM> gQ,
		Matrix<GPtr<f16>, -1, QK_DIM> gKV,
		Matrix<GPtr<f16>, -1, V_DIM> gO
	) {
		std::mutex log_mutex;

		std::vector<std::unique_ptr<BlockShared>> all_shared;
		for (size_t bX = 0; bX < grid_dim.x; ++bX) {
			all_shared.push_back(
				std::make_unique<BlockShared>(
					q_sram_count,
					kv_sram_count,
					kv_preload,
					block_warps_dim.x,
					&log_mutex
				)
			);
		}

		std::vector<std::thread> threads;
		for (size_t bX = 0; bX < grid_dim.x; ++bX) {
			BlockShared *block_shared = all_shared[bX].get();
			for (uint wX = 0; wX < block_warps_dim.x; ++wX) {
				threads.emplace_back(
					[block_shared, bX, wX, gQ, gKV, gO] () {
						block_idx.x = bX;
						warp_idx.x = wX;
						shared = block_shared;
						attn_kernel(gQ, gKV, gO);
					}
				);
			}
		}

		std::cerr << "Launched " << threads.size() << " CPU warps for attention kernel simulation." << std::endl;

		for (auto& t : threads) {
			t.join();
		}
	}

	void test_attn() {
		constexpr size_t Q_LEN = 4096;
		constexpr size_t KV_LEN = 4096;

		// allocate q: f16 [Q_LEN, QK_DIM]
		std::vector<f16> q_data(Q_LEN * QK_DIM);
		{
			std::ifstream in("q.bin", std::ios::binary);
			in.read(
				reinterpret_cast<char*>(q_data.data()),
				static_cast<std::streamsize>(q_data.size() * sizeof(*q_data.data()))
			);
		}
		Matrix<GPtr<f16>, -1, QK_DIM, RowMajor> q{
			GPtr<f16>{q_data},
			Q_LEN,
			QK_DIM
		};

		// allocate kv: f16 [KV_LEN, QK_DIM]
		std::vector<f16> kv_data(KV_LEN * QK_DIM);
		{
			std::ifstream in("kv.bin", std::ios::binary);
			in.read(
				reinterpret_cast<char*>(kv_data.data()),
				static_cast<std::streamsize>(kv_data.size() * sizeof(*kv_data.data()))
			);
		}
		Matrix<GPtr<f16>, -1, QK_DIM, RowMajor> kv{
			GPtr<f16>{kv_data},
			KV_LEN,
			QK_DIM
		};

		// allocate output: f16 [Q_LEN, V_DIM]
		std::vector<f16> out_data(Q_LEN * V_DIM);
		Matrix<GPtr<f16>, -1, V_DIM, RowMajor> out{
			GPtr<f16>{out_data},
			Q_LEN,
			V_DIM
		};

		std::cerr << "Starting CPU attention kernel test..." << std::endl;

		Dim3X gridDim(Q_LEN / Q_PER_BLOCK);
		start_attn_kernel(
			gridDim,
			/*q_sram_count=*/ Q_PER_BLOCK * QK_DIM,
			/*kv_sram_count=*/ KV_PER_STEP * QK_DIM,
			/*kv_preload=*/ KV_PRELOAD,
			q, kv, out
		);

		// write output to file
		{
			std::ofstream out_file("out_cpu.bin", std::ios::binary);
			out_file.write(
				reinterpret_cast<char*>(out_data.data()),
				static_cast<std::streamsize>(out_data.size() * sizeof(*out_data.data()))
			);
		}
	}
}

int main(int argc, char** argv) {
	cpu_test::test_attn();
	return 0;
}

__global__ static void attn_device() {
	using namespace cute;
	MmaAtom mma_atom{};

	/*Tensor rA = make_tensor<typename MmaOpTraits::ValTypeA,
                          typename MmaOpTraits::ALayout>{};*/
	/*Tensor rA = mma_atom.make_fragment_A();
	Tensor rB = make_fragment_B(mma_atom);
	Tensor rC = make_fragment_C(mma_atom);*/

/*	clear(rC);

	gemm(mma_atom, rA, rB, rC);*/
}

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));              // MMA_N

  // Clear the accumulators
  clear(tCrC);

  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
    print("tCrB : "); print(tCrB); print("\n");
    print("tCrC : "); print(tCrC); print("\n");

    print("tXsA : "); print(tXsA); print("\n");
    print("tXrA : "); print(tXrA); print("\n");
    print("tXsB : "); print(tXsB); print("\n");
    print("tXrB : "); print(tXrB); print("\n");
  }
#endif

#if 1

  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX-1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
  //           and explicit pipelines in shared memory.
  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
  //   Data is computed on registers(b_block).
  //
  //   This allows all copies and compute to overlap:
  //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
  //     Copy from smem->rmem can overlap with compute on rmem.
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

template <class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        cute::half_t const* A, int ldA,
        cute::half_t const* B, int ldB,
        Beta beta,
        cute::half_t      * C, int ldC,
        cudaStream_t stream = 0)
{
  assert(false && "Not implemented");
}

// Setup params for a TN HGEMM
template <class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        cute::half_t const* A, int ldA,
        cute::half_t const* B, int ldB,
        Beta beta,
        cute::half_t      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,Shape <_8, _8>>,
                                         Stride<_8,Stride<_1,_64>>>{});

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
  auto sC = make_layout(make_shape(bM, bN));

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 n-major

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                 Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM

  //Copy_Atom<DefaultCopy, half_t> s2r_atom_A;
  //Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_A;
  //Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_A;
  //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_A;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;

  //Copy_Atom<DefaultCopy, half_t> s2r_atom_B;
  //Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_B;
  //Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_B;
  //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_B;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  auto kernel_fptr = gemm_device<
    decltype(prob_shape), decltype(cta_tiler),
    cute::half_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
    cute::half_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
    cute::half_t, decltype(dC), decltype(sC), decltype(mmaC),
    decltype(alpha), decltype(beta)>;

  // Set L1 to be SMEM only
  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, s2r_atom_A,
       B, dB, sB, copyB, s2r_atom_B,
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK, bP));             // (m,k,p) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK, bP));             // (n,k,p) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 n-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 n-major

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, Copy_Atom<AutoVectorizingCopy, TA>{},
       B, dB, sB, copyB, Copy_Atom<AutoVectorizingCopy, TB>{},
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA_atom                  = make_layout(make_shape (      bM,          bK),
                                              make_stride(Int<1>{}, bM+Int<1>{})); // (m,k) -> smem_idx; padded m-major
  [[maybe_unused]] auto sB_atom = make_layout(make_shape (      bN,          bK),
                                              make_stride(Int<1>{}, bN+Int<1>{})); // (n,k) -> smem_idx; padded n-major
  auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(sA_atom, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TB>, TB>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, Copy_Atom<AutoVectorizingCopy, TA>{},
       B, dB, sB, copyB, Copy_Atom<AutoVectorizingCopy, TB>{},
       C, dC, sC, mmaC,
       alpha, beta);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}


int main1(int argc, char** argv)
{
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major < 8) {
    std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  std::cout << "Using device 0: " << props.name
            << " (SM" << props.major * 10 + props.minor
            << ", " << props.multiProcessorCount
            << ")" << std::endl;

  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'T';
  if (argc >= 5)
    sscanf(argv[4], "%c", &transA);

  char transB = 'N';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transB);

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;
  using TI = cute::half_t;

  TI alpha = static_cast<TI>(1.0f);
  TI beta  = static_cast<TI>(0.0f);

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  // Run once
  d_C = h_C;
  gemm(transA, transB, m, n, k,
       alpha,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k,
         alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         beta,
         d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  return 0;
}
