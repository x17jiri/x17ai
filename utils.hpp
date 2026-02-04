
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdint.h>

#define X17_UNROLL
#define X17_NO_UNROLL

using f16 = __half;
using bf16 = __nv_bfloat16;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using isize = i32;
using usize = u32;

template<const isize V>
struct ConstExtent {
	inline constexpr usize value() const noexcept {
		return V;
	}
};

struct Extent {
	usize v;
	inline constexpr usize value() const noexcept {
		return v;
	}
};

template<typename T>
struct GPtr: DataPtr<T> {
	T *ptr;

	GPtr(T *p): ptr(p) {}

	GPtr with_offset(usize offset) const {
		return GPtr(ptr + offset);
	}
};

template<typename T>
struct SPtr: DataPtr<T> {
	T *ptr;

	SPtr(T *p): ptr(p) {}

	SPtr with_offset(usize offset) const {
		return SPtr(ptr + offset);
	}
};

enum StrideType {
	RowMajor,
	ColumnMajor
};

template<
	typename Data,
	const isize M, // number of rows
	const isize N, // number of columns
	const StrideType S = RowMajor,
	const usize STRIDE = (S == RowMajor ? N : M)
>
requires(
	(M >= 0 || N >= 0) // at least one dimension must be known
	&& ( !(S == RowMajor) || N >= 0 ) // if row-major, N must be known
	&& ( !(S == ColumnMajor) || M >= 0 ) // if column-major, M must be known
	&& ( !(S == RowMajor) || STRIDE >= N ) // if row-major, stride must be >= N
	&& ( !(S == ColumnMajor) || STRIDE >= M ) // if column-major, stride must be >= M
)
struct Matrix {
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

	std::tuple<Data, MExtent, NExtent> members;

	Matrix(Data d)
	requires(M > 0 && N > 0 && STRIDE > 0):
		members(
			d,
			MExtent{},
			NExtent{},
		)
	{}

	Matrix(Data data_, ICount m_rows, ICount row_stride)
	requires(M < 0 && N > 0 && S == RowMajor):
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
