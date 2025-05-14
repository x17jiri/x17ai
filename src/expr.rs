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

fn __elem_wise<'a, const N: usize, F: Fn([SliceSet; N])>(a: [&Tensor; N], f: F) {
	let merger = DimMerger::new(a.map(|t| t.dims.as_slice()), 1);
	let smallest = merger.smallest_dim();
	let batch_dims = merger.dims_increasing_without_smallest();
	let batch_iter = batch_dims.iter();

	// TODO - should allow stride == 0?
	assert!(smallest.strides.iter().copied().all(|stride| stride == 1));

	let dtypes = a.map(|a| a.dtype());
	let buffers = a.map(|a| a.buffer.as_ref());

	batch::run(
		batch_iter,
		a.map(|t| t.offset),
		|batch_size: TensorSize, batch_strides: [TensorSize; N], offsets: [TensorSize; N]| {
			f(std::array::from_fn(|i| SliceSet {
				buffer: buffers[i],
				dtype: dtypes[i],
				len: smallest.size,
				batch_size,
				batch_stride: batch_strides[i],
				offset: offsets[i],
			}));
		},
	);
}

fn __vec_wise<'a, const N: usize, F: Fn([SliceSet; N])>(a: [&Tensor; N], f: F) {
	assert!(a.iter().all(|a| a.ndim() >= 1));
	assert!(a.iter().all(|a| a.dim(-1).is_contiguous()));

	let lengths = a.map(|a| a.dim(-1).size);
	let dtypes = a.map(|a| a.dtype());
	let buffers = a.map(|a| a.buffer.as_ref());

	let merger = DimMerger::new(a.map(|a| &a.dims[..a.ndim() - 1]), 1);
	let batch_dims = merger.dims_increasing();
	let batch_iter = batch_dims.iter();

	batch::run(
		batch_iter,
		a.map(|a| a.offset),
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

fn __mat_wise<'a, const N: usize, F: Fn([MatrixSet; N])>(
	a: [&Matrix; N], batch_dims: MergedDimIter<N>, f: F,
) {
	batch::run(
		batch_dims,
		a.map(|a| a.tensor.offset),
		|batch_size: TensorSize, batch_strides: [TensorSize; N], offsets: [TensorSize; N]| {
			f(std::array::from_fn(|i| MatrixSet {
				slice_set: SliceSet {
					buffer: a[i].tensor.buffer.as_ref(),
					dtype: a[i].tensor.dtype(),
					offset: offsets[i],
					len: MatrixSet::slice_len(
						a[i].rows,
						a[i].cols,
						a[i].row_stride,
						a[i].col_stride,
					),
					batch_size,
					batch_stride: batch_strides[i],
				},

				rows: a[i].rows,
				cols: a[i].cols,
				row_stride: a[i].row_stride,
				col_stride: a[i].col_stride,
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

pub struct VecMul<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
}

pub fn dot<'a>(a: &'a Tensor, b: &'a Tensor) -> VecMul<'a> {
	VecMul { a, b }
}

impl<'a> Savable for VecMul<'a> {
	fn save_to(&self, to: &Tensor) {
		__vec_wise([to, self.a, self.b], |[to, a, b]| {
			to.buffer.dot(&to, &a, &b);
		});
	}
}

impl Accumulable for VecMul<'_> {
	fn acc_to(&self, to: &Tensor, to_weight: f64, expr_weight: f64) {
		__vec_wise([to, self.a, self.b], |[to, a, b]| {
			to.buffer.dot_acc(&to, to_weight, &a, &b, expr_weight);
		});
	}
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
		let merger = DimMerger::new([to.batch_dims, a.batch_dims, b.batch_dims], 1);

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
