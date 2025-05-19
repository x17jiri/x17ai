// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::num::NonZeroUsize;

pub trait Savable {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` tensor.
	fn save_to(&self, to: &Tensor);
}

pub trait MatrixSavable {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` matrix.
	fn save_to(self, to: Matrix);
}

pub trait Accumulable {
	/// Calculate the result of the operation represented by `self`
	/// and accumulate it into the `to` tensor.
	///
	///    to = to_weight * to + expr_weight * self
	fn acc_to(&self, to: &Tensor, to_weight: f64, expr_weight: f64);
}

pub trait MatrixAccumulable {
	/// Calculate the result of the operation represented by `self`
	/// and accumulate it into the `to` matrix.
	///
	///    to = to_weight * to + expr_weight * self
	fn acc_to(self, to: Matrix, to_weight: f64, expr_weight: f64);
}

//--------------------------------------------------------------------------------------------------

/// Broadcast is disabled for tensors[0] and enabled for tensors[1..].
fn __elem_wise<'a, const N: usize, F: FnMut([SliceSet; N])>(tensors: [&Tensor; N], mut f: F) {
	let merger = DimMerger::new(tensors.map(|t| t.dims.as_slice()));
	let smallest = merger.smallest_dim();
	let batch_dims = merger.dims_increasing_without_smallest();
	let batch_iter = batch_dims.iter();

	let dtypes = tensors.map(|t| t.dtype());
	let buffers = tensors.map(|t| t.buffer.as_ref());

	let lengths: [TensorSize; N] = std::array::from_fn(|i| {
		let dim = smallest.size_and_stride(i);
		if dim.is_contiguous() {
			smallest.size
		} else {
			assert!(
				dim.is_broadcasted(),
				"last dimension of each tensor needs to be either contiguous or broadcasted. It cannot be strided"
			);
			assert!(i != 0, "broadcast is disabled for this tensor");
			1
		}
	});

	batch::run(
		batch_iter,
		tensors.map(|t| t.offset),
		|batch_size: TensorSize, batch_strides: [TensorSize; N], offsets: [TensorSize; N]| {
			f(std::array::from_fn(|i| SliceSet {
				buffer: buffers[i],
				dtype: dtypes[i],
				len: lengths[i],
				batch_size,
				batch_stride: batch_strides[i],
				offset: offsets[i],
			}));
		},
	);
}

/// Data dimension broadcast is disabled for all tensors.
/// This could be improved.
///
/// Batch dimensions broadcast is disabled for tensors[0] and enabled for tensors[1..].
/// This is by design.
fn __vec_wise<'a, const N: usize, F: Fn([SliceSet; N])>(tensors: [&Tensor; N], f: F) {
	assert!(tensors.iter().all(|t| t.ndim() >= 1));
	assert!(tensors.iter().all(|t| t.dim(-1).is_contiguous()));

	let lengths = tensors.map(|t| t.dim(-1).size);
	let dtypes = tensors.map(|t| t.dtype());
	let buffers = tensors.map(|t| t.buffer.as_ref());

	let merger = DimMerger::new(tensors.map(|t| &t.dims[..t.ndim() - 1]));
	let batch_dims = merger.dims_increasing();
	let batch_iter = batch_dims.iter();

	batch::run(
		batch_iter,
		tensors.map(|t| t.offset),
		|batch_size: TensorSize, batch_strides: [TensorSize; N], offsets: [TensorSize; N]| {
			f(std::array::from_fn(|i| SliceSet {
				buffer: buffers[i],
				dtype: dtypes[i],
				len: lengths[i],
				batch_size,
				batch_stride: batch_strides[i],
				offset: offsets[i],
			}));
		},
	);
}

/// At least one of the matrix dimensions should be contiguous.
///
/// Batch dimensions broadcast is disabled for matrices[0] and enabled for matrices[1..].
/// This is by design.
fn __mat_wise<'a, const N: usize, F: Fn([MatrixSet; N])>(
	matrices: [&Matrix; N], batch_dims: MergedDimIter<N>, f: F,
) {
	assert!(matrices.iter().all(|m| {
		let row_dim = SizeAndStride { size: m.rows.get(), stride: m.row_stride };
		let col_dim = SizeAndStride { size: m.cols.get(), stride: m.col_stride };
		row_dim.is_contiguous() || col_dim.is_contiguous()
	}));
	batch::run(
		batch_dims,
		matrices.map(|m| m.tensor.offset),
		|batch_size: TensorSize, batch_strides: [TensorSize; N], offsets: [TensorSize; N]| {
			f(std::array::from_fn(|i| MatrixSet {
				slice_set: SliceSet {
					buffer: matrices[i].tensor.buffer.as_ref(),
					dtype: matrices[i].tensor.dtype(),
					offset: offsets[i],
					len: MatrixSet::slice_len(
						matrices[i].rows,
						matrices[i].cols,
						matrices[i].row_stride,
						matrices[i].col_stride,
					),
					batch_size,
					batch_stride: batch_strides[i],
				},

				rows: matrices[i].rows,
				cols: matrices[i].cols,
				row_stride: matrices[i].row_stride,
				col_stride: matrices[i].col_stride,
			}));
		},
	);
}

//--------------------------------------------------------------------------------------------------

pub struct Zeros();

pub fn zeros() -> Zeros {
	Zeros()
}

impl Savable for Zeros {
	fn save_to(&self, to: &Tensor) {
		__elem_wise([to], |[to]| {
			to.buffer.zeros(&to);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Randn();

pub fn randn() -> Randn {
	Randn()
}

impl Savable for Randn {
	fn save_to(&self, to: &Tensor) {
		__elem_wise([to], |[to]| {
			to.buffer.randn(&to);
		});
	}
}

//--------------------------------------------------------------------------------------------------

impl Savable for Tensor {
	fn save_to(&self, to: &Tensor) {
		__elem_wise([to, self], |[to, input]| {
			to.buffer.copy(&to, &input);
		});
	}
}

impl Accumulable for Tensor {
	fn acc_to(&self, to: &Tensor, to_weight: f64, expr_weight: f64) {
		__elem_wise([to, self], |[to, input]| {
			to.buffer.acc(&to, to_weight, &input, expr_weight);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Mul<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
}

pub fn mul<'a>(a: &'a Tensor, b: &'a Tensor) -> Mul<'a> {
	Mul { a, b }
}

impl<'a> Savable for Mul<'a> {
	fn save_to(&self, to: &Tensor) {
		__elem_wise([to, self.a, self.b], |[to, a, b]| {
			to.buffer.mul(&to, &a, &b);
		});
	}
}

impl<'a> Accumulable for Mul<'a> {
	fn acc_to(&self, to: &Tensor, to_weight: f64, expr_weight: f64) {
		__elem_wise([to, self.a, self.b], |[to, a, b]| {
			to.buffer.mul_acc(&to, to_weight, &a, &b, expr_weight);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Sub<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
}

pub fn sub<'a>(a: &'a Tensor, b: &'a Tensor) -> Sub<'a> {
	Sub { a, b }
}

impl<'a> Savable for Sub<'a> {
	fn save_to(&self, to: &Tensor) {
		__elem_wise([to, self.a, self.b], |[to, a, b]| {
			to.buffer.sub(&to, &a, &b);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RSqrt<'a> {
	pub tensor: &'a Tensor,
	pub eps: f64,
}

pub fn rsqrt(tensor: &Tensor, eps: f64) -> RSqrt {
	RSqrt { tensor, eps }
}

impl<'a> Savable for RSqrt<'a> {
	fn save_to(&self, to: &Tensor) {
		__elem_wise([to, self.tensor], |[to, input]| {
			to.buffer.rsqrt(&to, &input, self.eps);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct LogClamped<'a> {
	pub tensor: &'a Tensor,
}

/// Calculates:
///
///     low_bound = max(-1000, DType.MAX_NEGATIVE);
///     dst = max(log(a), low_bound);
///
/// So the output is defined even for a <= 0.
pub fn log_clamped(tensor: &Tensor) -> LogClamped {
	LogClamped { tensor }
}

impl<'a> Savable for LogClamped<'a> {
	fn save_to(&self, to: &Tensor) {
		__elem_wise([to, self.tensor], |[to, input]| {
			to.buffer.log_clamped(&to, &input);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct VecMul<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
	pub scale: f64,
}

impl<'a> VecMul<'a> {
	pub fn scale(self, scale: f64) -> VecMul<'a> {
		VecMul { scale: self.scale * scale, ..self }
	}
}

pub fn dot<'a>(a: &'a Tensor, b: &'a Tensor) -> VecMul<'a> {
	VecMul { a, b, scale: 1.0 }
}

impl<'a> Savable for VecMul<'a> {
	fn save_to(&self, to: &Tensor) {
		__vec_wise([to, self.a, self.b], |[to, a, b]| {
			to.buffer.dot(&to, &a, &b, self.scale);
		});
	}
}

impl Accumulable for VecMul<'_> {
	fn acc_to(&self, to: &Tensor, to_weight: f64, expr_weight: f64) {
		__vec_wise([to, self.a, self.b], |[to, a, b]| {
			to.buffer.dot_acc(&to, to_weight, &a, &b, expr_weight * self.scale);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub fn sum_all(tensor: &Tensor) -> f64 {
	let mut sum = 0.0;
	// TODO - `__elem_wise()` disables broadcast for tensor at position 0.
	// In the case of a `sum_all()`, it would make sense to enable it,
	// but it would require some refactoring. Not sure if it is worth it.
	__elem_wise([tensor], |[a]| {
		sum += a.buffer.sum_all(&a);
	});
	sum
}

//--------------------------------------------------------------------------------------------------

pub struct Softmax<'a> {
	pub tensor: &'a Tensor,
}

pub fn softmax<'a>(tensor: &'a Tensor) -> Softmax<'a> {
	Softmax { tensor }
}

impl<'a> Savable for Softmax<'a> {
	fn save_to(&self, to: &Tensor) {
		__vec_wise([to, self.tensor], |[to, input]| {
			to.buffer.softmax(&to, &input);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RMSNorm<'a> {
	pub tensor: &'a Tensor,
	pub eps: f64,
}

pub fn rms_norm<'a>(tensor: &'a Tensor, eps: f64) -> RMSNorm<'a> {
	RMSNorm { tensor, eps }
}

impl<'a> Savable for RMSNorm<'a> {
	fn save_to(&self, to: &Tensor) {
		__vec_wise([to, self.tensor], |[to, input]| {
			to.buffer.rms_norm(&to, &input, self.eps);
		});
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Matrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],

	pub rows: NonZeroTensorSize,
	pub cols: NonZeroTensorSize,
	pub row_stride: TensorSize,
	pub col_stride: TensorSize,
}

impl<'a> Matrix<'a> {
	pub fn T(self) -> Matrix<'a> {
		Matrix {
			rows: self.cols,
			cols: self.rows,
			row_stride: self.col_stride,
			col_stride: self.row_stride,
			..self
		}
	}
}

pub fn matrix<'a>(tensor: &'a Tensor) -> Matrix<'a> {
	assert!(tensor.ndim() >= 2);
	let rows = tensor.dim_from_end(2);
	let cols = tensor.dim_from_end(1);
	Matrix {
		tensor,
		batch_dims: &tensor.dims[..tensor.ndim() - 2],
		rows: NonZeroTensorSize::new(rows.size).unwrap(),
		cols: NonZeroTensorSize::new(cols.size).unwrap(),
		row_stride: rows.stride,
		col_stride: cols.stride,
	}
}

pub fn row_matrix<'a>(tensor: &'a Tensor) -> Matrix<'a> {
	assert!(tensor.ndim() >= 1);
	let rows = SizeAndStride { size: 1, stride: 0 };
	let cols = tensor.dim_from_end(1);
	Matrix {
		tensor,
		batch_dims: &tensor.dims[..tensor.ndim() - 1],
		rows: NonZeroTensorSize::new(rows.size).unwrap(),
		cols: NonZeroTensorSize::new(cols.size).unwrap(),
		row_stride: rows.stride,
		col_stride: cols.stride,
	}
}

pub fn col_matrix<'a>(tensor: &'a Tensor) -> Matrix<'a> {
	assert!(tensor.ndim() >= 1);
	let rows = tensor.dim_from_end(1);
	let cols = SizeAndStride { size: 1, stride: 0 };
	Matrix {
		tensor,
		batch_dims: &tensor.dims[..tensor.ndim() - 1],
		rows: NonZeroTensorSize::new(rows.size).unwrap(),
		cols: NonZeroTensorSize::new(cols.size).unwrap(),
		row_stride: rows.stride,
		col_stride: cols.stride,
	}
}

//--------------------------------------------------------------------------------------------------

pub struct MatMul<'a> {
	pub a: Matrix<'a>,
	pub b: Matrix<'a>,
	pub scale: f64,
}

pub fn mm<'a>(a: Matrix<'a>, b: Matrix<'a>) -> MatMul<'a> {
	MatMul { a, b, scale: 1.0 }
}

impl<'a> MatMul<'a> {
	pub fn scale(self, scale: f64) -> MatMul<'a> {
		MatMul { scale: self.scale * scale, ..self }
	}
}

struct MatMulPrep<'a> {
	to: Matrix<'a>,
	a: Matrix<'a>,
	b: Matrix<'a>,
	batch_dims: MergedDimList<3>,
}

impl<'a> MatMulPrep<'a> {
	fn new(mm: MatMul<'a>, mut to: Matrix<'a>) -> MatMulPrep<'a> {
		let mut a = mm.a;
		let mut b = mm.b;

		const TO: usize = 0;
		const A: usize = 1;
		const B: usize = 2;
		// TODO - the output tensor could be broadcasted
		let merger = DimMerger::new([to.batch_dims, a.batch_dims, b.batch_dims]);

		// Is this actually a vector dot product?
		if to.rows.get() == 1 && to.cols.get() == 1 {
			todo!("implement vector dot product");
		}

		if to.rows.get() == 1 {
			// `to` has single row, so `a` also has to have just one row
			assert!(a.rows.get() == 1);

			// Do we have a batch of matrix multiplications
			// where the matrix `b` is the same for all items in the batch?
			let batch_dim = merger.smallest_dim();
			if batch_dim.size > 1 && batch_dim.strides[B] == 0 {
				a.rows = NonZeroTensorSize::new(batch_dim.size).unwrap();
				a.row_stride = batch_dim.strides[A];

				to.rows = NonZeroTensorSize::new(batch_dim.size).unwrap();
				to.row_stride = batch_dim.strides[TO];

				return MatMulPrep {
					to,
					a,
					b,
					batch_dims: merger.dims_increasing_without_smallest(),
				};
			}
		} else if to.cols.get() == 1 {
			// `to` has single column, so `b` also has to have just one column
			assert!(b.cols.get() == 1);

			// Do we have a batch of matrix multiplications
			// where the matrix `a` is the same for all items in the batch?
			let batch_dim = merger.smallest_dim();
			if batch_dim.size > 1 && batch_dim.strides[A] == 0 {
				b.cols = NonZeroTensorSize::new(batch_dim.size).unwrap();
				b.col_stride = batch_dim.strides[B];

				to.cols = NonZeroTensorSize::new(batch_dim.size).unwrap();
				to.col_stride = batch_dim.strides[TO];

				return MatMulPrep {
					to,
					a,
					b,
					batch_dims: merger.dims_increasing_without_smallest(),
				};
			}
		}

		MatMulPrep {
			to,
			a,
			b,
			batch_dims: merger.dims_increasing(),
		}
	}
}

impl<'a> MatrixSavable for MatMul<'a> {
	fn save_to(self, to: Matrix) {
		let scale = self.scale;
		let prep = MatMulPrep::new(self, to);
		__mat_wise([&prep.to, &prep.a, &prep.b], prep.batch_dims.iter(), |[to, a, b]| {
			to.slice_set.buffer.gemm(&to, 0.0, &a, &b, scale);
		});
	}
}

impl<'a> MatrixAccumulable for MatMul<'a> {
	fn acc_to(self, to: Matrix, to_weight: f64, expr_weight: f64) {
		let scale = self.scale * expr_weight;
		let prep = MatMulPrep::new(self, to);
		__mat_wise([&prep.to, &prep.a, &prep.b], prep.batch_dims.iter(), |[to, a, b]| {
			to.slice_set.buffer.gemm(&to, to_weight, &a, &b, scale);
		});
	}
}

//--------------------------------------------------------------------------------------------------
