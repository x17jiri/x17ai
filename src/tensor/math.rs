// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::num::NonZeroUsize;

use crate::tensor::device::AttentionParams;

use super::buffer::{MatrixSet, SliceSet};
use super::dim_merger::{DimMerger, MergedDimIter, MergedDimList};
use super::{SizeAndStride, Tensor, batch};

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
pub(crate) fn __elem_wise<'a, const N: usize, F: FnMut([SliceSet; N])>(
	tensors: [&Tensor; N], mut f: F,
) {
	let merger = DimMerger::new(tensors.map(|t| t.dim_slice(..)));
	let smallest = merger.smallest_dim();
	let batch_dims = merger.dims_increasing_without_smallest();
	let batch_iter = batch_dims.iter();

	let dtypes = tensors.map(|t| t.dtype());
	let buffers = tensors.map(|t| t.buffer.as_ref());

	let lengths: [usize; N] = std::array::from_fn(|i| {
		if smallest.size_and_stride(i).is_contiguous() {
			smallest.size
		} else {
			assert!(
				smallest.size_and_stride(i).is_broadcasted(),
				"last dimension of each tensor needs to be either contiguous or broadcasted. It cannot be strided"
			);
			assert!(i != 0, "broadcast is disabled for this tensor");
			1
		}
	});

	batch::run(
		batch_iter,
		tensors.map(|t| t.offset),
		|batch_size: usize, batch_strides: [usize; N], offsets: [usize; N]| {
			f(std::array::from_fn(|i| SliceSet {
				buffer: buffers[i],
				dtype: dtypes[i],
				offset: offsets[i],
				len: lengths[i],
				count: batch_size,
				stride: batch_strides[i],
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

	let merger = DimMerger::new(tensors.map(|t| t.dim_slice(..t.ndim() - 1)));
	let batch_dims = merger.dims_increasing();
	let batch_iter = batch_dims.iter();

	batch::run(
		batch_iter,
		tensors.map(|t| t.offset),
		|batch_size: usize, batch_strides: [usize; N], offsets: [usize; N]| {
			f(std::array::from_fn(|i| SliceSet {
				buffer: buffers[i],
				dtype: dtypes[i],
				len: lengths[i],
				offset: offsets[i],
				count: batch_size,
				stride: batch_strides[i],
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
		|batch_size: usize, batch_strides: [usize; N], offsets: [usize; N]| {
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
					count: batch_size,
					stride: batch_strides[i],
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
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to], |[to]| {
			executor.zeros(&to);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Randn();

pub fn randn() -> Randn {
	Randn()
}

impl Savable for Randn {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to], |[to]| {
			executor.randn(&to);
		});
	}
}

//--------------------------------------------------------------------------------------------------

impl Savable for Tensor {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to, self], |[to, input]| {
			executor.copy(&to, &input);
		});
	}
}

impl Accumulable for Tensor {
	#[inline(never)]
	fn acc_to(&self, to: &Tensor, to_weight: f64, expr_weight: f64) {
		let executor = to.buffer.executor();
		__elem_wise([to, self], |[to, input]| {
			executor.acc(&to, to_weight, &input, expr_weight);
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
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to, self.a, self.b], |[to, a, b]| {
			executor.mul(&to, &a, &b);
		});
	}
}

impl<'a> Accumulable for Mul<'a> {
	#[inline(never)]
	fn acc_to(&self, to: &Tensor, to_weight: f64, expr_weight: f64) {
		let executor = to.buffer.executor();
		__elem_wise([to, self.a, self.b], |[to, a, b]| {
			executor.mul_acc(&to, to_weight, &a, &b, expr_weight);
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
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to, self.a, self.b], |[to, a, b]| {
			executor.sub(&to, &a, &b);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Add<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
}

pub fn add<'a>(a: &'a Tensor, b: &'a Tensor) -> Add<'a> {
	Add { a, b }
}

impl<'a> Savable for Add<'a> {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to, self.a, self.b], |[to, a, b]| {
			executor.add(&to, &a, &b);
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
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to, self.tensor], |[to, input]| {
			executor.rsqrt(&to, &input, self.eps);
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
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to, self.tensor], |[to, input]| {
			executor.log_clamped(&to, &input);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SwiGLU<'a> {
	pub lin: &'a Tensor,
	pub gate: &'a Tensor,
}

pub fn swiglu<'a>(lin: &'a Tensor, gate: &'a Tensor) -> SwiGLU<'a> {
	SwiGLU { lin, gate }
}

impl<'a> Savable for SwiGLU<'a> {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__elem_wise([to, self.lin, self.gate], |[to, lin, gate]| {
			executor.swiglu(&to, &lin, &gate);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SwiGLUBackward<'a> {
	pub d_out: &'a Tensor,
	pub lin: &'a Tensor,
	pub gate: &'a Tensor,
}

pub fn swiglu_backward<'a>(
	d_out: &'a Tensor, lin: &'a Tensor, gate: &'a Tensor,
) -> SwiGLUBackward<'a> {
	SwiGLUBackward { d_out, lin, gate }
}

impl<'a> SwiGLUBackward<'a> {
	#[inline(never)]
	pub fn save_to(&self, d_lin: &Tensor, d_gate: &Tensor) {
		let executor = d_lin.buffer.executor();
		__elem_wise(
			[d_lin, d_gate, self.lin, self.gate, self.d_out],
			|[d_lin, d_gate, lin, gate, d_out]| {
				executor.swiglu_backward(&d_lin, &d_gate, &lin, &gate, &d_out);
			},
		);
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
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__vec_wise([to, self.a, self.b], |[to, a, b]| {
			executor.dot(&to, &a, &b, self.scale);
		});
	}
}

impl Accumulable for VecMul<'_> {
	#[inline(never)]
	fn acc_to(&self, to: &Tensor, to_weight: f64, expr_weight: f64) {
		let executor = to.buffer.executor();
		__vec_wise([to, self.a, self.b], |[to, a, b]| {
			executor.dot_acc(&to, to_weight, &a, &b, expr_weight * self.scale);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub fn sum_all(tensor: &Tensor) -> f64 {
	let executor = tensor.buffer.executor();
	let mut sum = 0.0;
	// TODO - `__elem_wise()` disables broadcast for tensor at position 0.
	// In the case of a `sum_all()`, it would make sense to enable it,
	// but it would require some refactoring. Not sure if it is worth it.
	__elem_wise([tensor], |[a]| {
		sum += executor.sum_all(&a);
	});
	sum
}

pub fn approx_eq(a: &Tensor, b: &Tensor, eps: f64) -> bool {
	let executor = a.buffer.executor();
	let mut result = true;
	__elem_wise([a, b], |[a, b]| {
		result &= executor.approx_eq(&a, &b, eps);
	});
	result
}

//--------------------------------------------------------------------------------------------------

pub struct Softmax<'a> {
	pub tensor: &'a Tensor,
}

pub fn softmax<'a>(tensor: &'a Tensor) -> Softmax<'a> {
	Softmax { tensor }
}

impl<'a> Savable for Softmax<'a> {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		__vec_wise([to, self.tensor], |[to, input]| {
			executor.softmax(&to, &input);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RMSNorm<'a> {
	pub tensor: &'a Tensor,
	pub eps: f64,
	pub scale_storage: Option<&'a Tensor>,
}

pub fn rms_norm<'a>(tensor: &'a Tensor, eps: f64) -> RMSNorm<'a> {
	RMSNorm { tensor, eps, scale_storage: None }
}

impl<'a> RMSNorm<'a> {
	pub fn scale_storage(self, scale_storage: &'a Tensor) -> RMSNorm<'a> {
		RMSNorm {
			scale_storage: Some(scale_storage),
			..self
		}
	}
}

impl<'a> Savable for RMSNorm<'a> {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		if let Some(scale_storage) = self.scale_storage {
			// TODO - could this broadcast the `scale_storage` tensor?
			__vec_wise([to, self.tensor, scale_storage], |[to, input, scale_storage]| {
				executor.rms_norm(&to, &input, self.eps, Some(&scale_storage));
			});
		} else {
			__vec_wise([to, self.tensor], |[to, input]| {
				executor.rms_norm(&to, &input, self.eps, None);
			});
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Matrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],

	pub rows: NonZeroUsize,
	pub cols: NonZeroUsize,
	pub row_stride: usize,
	pub col_stride: usize,
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
	let rows = tensor.dim(-2);
	let cols = tensor.dim(-1);
	Matrix {
		tensor,
		batch_dims: tensor.dim_slice(..tensor.ndim() - 2),
		rows: NonZeroUsize::new(rows.size).unwrap(),
		cols: NonZeroUsize::new(cols.size).unwrap(),
		row_stride: rows.stride,
		col_stride: cols.stride,
	}
}

pub fn row_matrix<'a>(tensor: &'a Tensor) -> Matrix<'a> {
	assert!(tensor.ndim() >= 1);
	let rows = SizeAndStride { size: 1, stride: 0 };
	let cols = tensor.dim(-1);
	Matrix {
		tensor,
		batch_dims: tensor.dim_slice(..tensor.ndim() - 1),
		rows: NonZeroUsize::new(rows.size).unwrap(),
		cols: NonZeroUsize::new(cols.size).unwrap(),
		row_stride: rows.stride,
		col_stride: cols.stride,
	}
}

pub fn col_matrix<'a>(tensor: &'a Tensor) -> Matrix<'a> {
	assert!(tensor.ndim() >= 1);
	let rows = tensor.dim(-1);
	let cols = SizeAndStride { size: 1, stride: 0 };
	Matrix {
		tensor,
		batch_dims: tensor.dim_slice(..tensor.ndim() - 1),
		rows: NonZeroUsize::new(rows.size).unwrap(),
		cols: NonZeroUsize::new(cols.size).unwrap(),
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
				a.rows = NonZeroUsize::new(batch_dim.size).unwrap();
				a.row_stride = batch_dim.strides[A];

				to.rows = NonZeroUsize::new(batch_dim.size).unwrap();
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
				b.cols = NonZeroUsize::new(batch_dim.size).unwrap();
				b.col_stride = batch_dim.strides[B];

				to.cols = NonZeroUsize::new(batch_dim.size).unwrap();
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
	#[inline(never)]
	fn save_to(self, to: Matrix) {
		let executor = to.tensor.buffer.executor();
		let scale = self.scale;
		let prep = MatMulPrep::new(self, to);
		__mat_wise([&prep.to, &prep.a, &prep.b], prep.batch_dims.iter(), |[to, a, b]| {
			executor.gemm(&to, 0.0, &a, &b, scale);
		});
	}
}

impl<'a> MatrixAccumulable for MatMul<'a> {
	#[inline(never)]
	fn acc_to(self, to: Matrix, to_weight: f64, expr_weight: f64) {
		let executor = to.tensor.buffer.executor();
		let scale = self.scale * expr_weight;
		let prep = MatMulPrep::new(self, to);
		__mat_wise([&prep.to, &prep.a, &prep.b], prep.batch_dims.iter(), |[to, a, b]| {
			executor.gemm(&to, to_weight, &a, &b, scale);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Attention<'a> {
	pub q: &'a Tensor,
	pub k: &'a Tensor,
	pub v: &'a Tensor,
}

/// Requirements:
///
///    q.shape = [..., inputs, q_heads, qk_features]
///    k.shape = [..., inputs, k_heads, qk_features]
///    v.shape = [..., inputs, v_heads, v_features]
///
///    q_heads >= k_heads && q_heads % k_heads == 0 && (q_heads / k_heads).is_power_of_two()
///    q_heads >= v_heads && q_heads % v_heads == 0 && (q_heads / v_heads).is_power_of_two()
///
/// The output shape is:
///
///    [..., inputs, q_heads, v_features]
pub fn attention<'a>(q: &'a Tensor, k: &'a Tensor, v: &'a Tensor) -> Attention<'a> {
	Attention { q, k, v }
}

impl<'a> Savable for Attention<'a> {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let tensors = [self.q, self.k, self.v, to];
		assert!(tensors.iter().all(|t| t.ndim() >= 3));

		let q_input_dim = self.q.dim(-3);
		let q_head_dim = self.q.dim(-2);
		let q_feature_dim = self.q.dim(-1);

		let k_input_dim = self.k.dim(-3);
		let k_head_dim = self.k.dim(-2);
		let k_feature_dim = self.k.dim(-1);

		let v_input_dim = self.v.dim(-3);
		let v_head_dim = self.v.dim(-2);
		let v_feature_dim = self.v.dim(-1);

		let to_output_dim = to.dim(-3);
		let to_head_dim = to.dim(-2);
		let to_feature_dim = to.dim(-1);

		let qk_features = q_feature_dim.size;
		assert!(q_feature_dim.is_contiguous());
		assert!(k_feature_dim.size == qk_features);
		assert!(k_feature_dim.is_contiguous());

		let v_features = v_feature_dim.size;
		assert!(v_feature_dim.is_contiguous());

		let q_heads = q_head_dim.size;
		assert!(
			q_head_dim.stride == qk_features,
			"TODO: If this were useful, we'd need to capture the stride in AttentionParams"
		);

		let k_heads = k_head_dim.size;
		assert!(
			k_head_dim.stride == qk_features,
			"TODO: If this were useful, we'd need to capture the stride in AttentionParams"
		);
		assert!(q_heads >= k_heads);
		assert!(q_heads % k_heads == 0);
		assert!((q_heads / k_heads).is_power_of_two());
		let k_shift = (q_heads / k_heads).trailing_zeros() as usize;

		let v_heads = v_head_dim.size;
		assert!(
			v_head_dim.stride == v_features,
			"TODO: If this were useful, we'd need to capture the stride in AttentionParams"
		);
		assert!(q_heads >= v_heads);
		assert!(q_heads % v_heads == 0);
		assert!((q_heads / v_heads).is_power_of_two());
		let v_shift = (q_heads / v_heads).trailing_zeros() as usize;

		let inputs = q_input_dim.size;
		assert!(k_input_dim.size == inputs);
		assert!(v_input_dim.size == inputs);

		assert!(to_output_dim.size == inputs);
		assert!(to_head_dim.size == q_heads);
		assert!(to_feature_dim.size == v_features);
		assert!(to_feature_dim.is_contiguous());
		assert!(
			to_head_dim.stride == v_features,
			"TODO: If this were useful, we'd need to capture the stride in AttentionParams"
		);

		let q_input_stride = q_input_dim.stride;
		let k_input_stride = k_input_dim.stride;
		let v_input_stride = v_input_dim.stride;
		let to_output_stride = to_output_dim.stride;

		let params = AttentionParams {
			heads: q_heads,
			qk_features,
			v_features,
			k_shift,
			v_shift,
		};

		let merger = DimMerger::new(tensors.map(|t| t.dim_slice(..t.ndim() - 3)));
		let batch_dims = merger.dims_increasing();
		let batch_iter = batch_dims.iter();

		let mut q = SliceSet {
			buffer: self.q.buffer.as_ref(),
			dtype: self.q.dtype(),
			offset: 0,
			len: q_heads * qk_features,
			count: inputs,
			stride: q_input_stride,
		};

		let mut k = SliceSet {
			buffer: self.k.buffer.as_ref(),
			dtype: self.k.dtype(),
			offset: 0,
			len: k_heads * qk_features,
			count: inputs,
			stride: k_input_stride,
		};

		let mut v = SliceSet {
			buffer: self.v.buffer.as_ref(),
			dtype: self.v.dtype(),
			offset: 0,
			len: v_heads * v_features,
			count: inputs,
			stride: v_input_stride,
		};

		let mut to = SliceSet {
			buffer: to.buffer.as_ref(),
			dtype: to.dtype(),
			offset: 0,
			len: q_heads * v_features,
			count: inputs,
			stride: to_output_stride,
		};

		let executor = to.buffer.executor();

		batch::run(
			batch_iter,
			tensors.map(|t| t.offset),
			|batch_size: usize, batch_strides: [usize; 4], offsets: [usize; 4]| {
				for i in 0..batch_size {
					q.offset = offsets[0] + i * batch_strides[0];
					k.offset = offsets[1] + i * batch_strides[1];
					v.offset = offsets[2] + i * batch_strides[2];
					to.offset = offsets[3] + i * batch_strides[3];
					executor.attention(&to, &q, &k, &v, &params);
				}
			},
		);
	}
}

//--------------------------------------------------------------------------------------------------
