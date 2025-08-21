//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::ErrPack;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut, check_borrows};
use crate::tensor::dim_merger::{DimMerger, MergedDim};
use crate::tensor::generic::map::{ND, NotEnoughDimensionsError, SizeAndStride};
use crate::tensor::{Tensor, TensorOpError, generic};
use crate::util::array;

//--------------------------------------------------------------------------------------------------

pub trait EvaluatesToTensor {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` tensor.
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>>;
}

pub trait ClearAccToMatrix {
	fn clear_acc_to_matrix(self, to: &Matrix) -> Result<(), ErrPack<TensorOpError>>;
}

pub trait EvaluatesToColMatrix {
	fn eval_to_col_matrix(self, to: &ColMatrix) -> Result<(), ErrPack<TensorOpError>>;
}

//--------------------------------------------------------------------------------------------------

// TODO - disable broadcast for M tensors.

pub struct ElemWise<'a, const M: usize, const C: usize>
where
	[(); M + C]:,
{
	m: [&'a Tensor; M],
	c: [&'a Tensor; C],
	dims: [MergedDim<{ M + C }>; 2],
}

impl<'a, const M: usize, const C: usize> ElemWise<'a, M, C>
where
	[(); M + C]:,
{
	pub fn new(m: [&'a Tensor; M], c: [&'a Tensor; C]) -> Result<Self, ErrPack<TensorOpError>> {
		let m_dims = m.map(|t| t.map().dims.as_slice());
		let c_dims = c.map(|t| t.map().dims.as_slice());
		let dims = DimMerger::merge(array::concat_arrays(m_dims, c_dims))?;
		Ok(Self { m, c, dims })
	}

	/// Note that we use 'K' instead of 'C' and it is allowed that 'K <= C'.
	///
	/// So we don't have to use all the `c` tensors processed during `::new()`.
	pub fn run<const O: usize, const K: usize>(
		&self,
		mut f: impl FnMut(
			&mut [generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>; O],
			&[generic::Tensor<ND<2>, DeviceBufferRef<'a>>; K],
		) -> Result<(), ErrPack<TensorOpError>>,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); M - O]:,
		[(); C - K]:,
	{
		unsafe {
			let mut c_fail = 0;
			let c_tensors = std::array::from_fn(|i| {
				generic::Tensor::new_unchecked(
					ND {
						dims: [self.dims[0].get(M + i), self.dims[1].get(M + i)],
						offset: self.c[i].map().offset,
					},
					DeviceBufferRef::new_unsafe(self.c[i].buf().as_ref(), &mut c_fail),
				)
			});
			let mut m_fail = 0;
			let mut m_tensors = std::array::from_fn(|i| {
				generic::Tensor::new_unchecked(
					ND {
						dims: [self.dims[0].get(i), self.dims[1].get(i)],
						offset: self.m[i].map().offset,
					},
					DeviceBufferRefMut::new_unsafe(self.m[i].buf().as_ref(), &mut m_fail),
				)
			});
			check_borrows(c_fail, m_fail)?;

			f(&mut m_tensors, &c_tensors)
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RandnClampedExpr();

pub fn randn_clamped() -> RandnClampedExpr {
	RandnClampedExpr()
}

impl EvaluatesToTensor for RandnClampedExpr {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let vmt = to.vmt();
		ElemWise::new([to], [])?.run(|[to], []| {
			vmt.randn_clamped(to)?;
			Ok(())
		})
	}
}

//--------------------------------------------------------------------------------------------------
/*
pub fn sum_all(tensor: &Tensor) -> Result<f64, ErrPack<TensorOpError>> {
	let vmt = tensor.vmt();
	let mut sum = 0.0;
	// TODO - `__elem_wise()` disables broadcast for tensor at position 0.
	// In the case of a `sum_all()`, it would make sense to enable it,
	// but it would require some refactoring. Not sure if it is worth it.
	ElemWise::new([], [tensor])?.run(|[], [a]| {
		sum += vmt.sum_all(a)?;
		Ok(())
	})?;
	Ok(sum)
}

//--------------------------------------------------------------------------------------------------

pub fn approx_eq(a: &Tensor, b: &Tensor, eps: f64) -> Result<bool, ErrPack<TensorOpError>> {
	let executor = a.executor();
	let mut result = true;
	ElemWise::new([], [a, b])?.run(|[], [a, b]| {
		result &= executor.approx_eq(a, b, eps)?;
		Ok(())
	})?;
	Ok(result)
}
*/
//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Matrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],
	pub rows: SizeAndStride,
	pub cols: SizeAndStride,
}

impl<'a> Matrix<'a> {
	pub fn T(self) -> Self {
		Matrix { rows: self.cols, cols: self.rows, ..self }
	}

	/// First clears the matrix `self` (i.e. sets all elements to zero),
	/// then accumulates the result of a batch of matrix multiplications into it.
	pub fn clear_acc<Expr: ClearAccToMatrix>(
		&self,
		expr: Expr,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.clear_acc_to_matrix(self)
	}
}

pub fn mat<'a>(tensor: &'a Tensor) -> Result<Matrix<'a>, NotEnoughDimensionsError> {
	let dims = tensor.map().dims.as_slice();
	if dims.len() < 2 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		Ok(Matrix {
			tensor,
			batch_dims: &dims[..dims.len() - 2],
			rows: dims[dims.len() - 2],
			cols: dims[dims.len() - 1],
		})
	}
}

#[derive(Clone, Copy)]
pub struct RowMatrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],
	pub cols: SizeAndStride,
}

impl<'a> RowMatrix<'a> {
	pub fn T(self) -> ColMatrix<'a> {
		ColMatrix {
			tensor: self.tensor,
			batch_dims: self.batch_dims,
			rows: self.cols,
		}
	}
}

pub fn row<'a>(tensor: &'a Tensor) -> Result<RowMatrix<'a>, NotEnoughDimensionsError> {
	let dims = tensor.map().dims.as_slice();
	#[allow(clippy::len_zero)]
	if dims.len() < 1 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		Ok(RowMatrix {
			tensor,
			batch_dims: &dims[..dims.len() - 1],
			cols: dims[dims.len() - 1],
		})
	}
}

#[derive(Clone, Copy)]
pub struct ColMatrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],
	pub rows: SizeAndStride,
}

impl<'a> ColMatrix<'a> {
	pub fn T(self) -> RowMatrix<'a> {
		RowMatrix {
			tensor: self.tensor,
			batch_dims: self.batch_dims,
			cols: self.rows,
		}
	}

	pub fn assign<Expr: EvaluatesToColMatrix>(
		&self,
		expr: Expr,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.eval_to_col_matrix(self)
	}
}

pub fn col<'a>(tensor: &'a Tensor) -> Result<ColMatrix<'a>, NotEnoughDimensionsError> {
	let dims = tensor.map().dims.as_slice();
	#[allow(clippy::len_zero)]
	if dims.len() < 1 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		Ok(ColMatrix {
			tensor,
			batch_dims: &dims[..dims.len() - 1],
			rows: dims[dims.len() - 1],
		})
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct ColTimesRow<'a> {
	pub col: ColMatrix<'a>,
	pub row: RowMatrix<'a>,
	pub scale: f64,
}

impl<'a> ColTimesRow<'a> {
	pub fn scale(self, scale: f64) -> Self {
		Self {
			col: self.col,
			row: self.row,
			scale: self.scale * scale,
		}
	}
}

#[derive(Clone, Copy)]
pub struct MatTimesCol<'a> {
	pub mat: Matrix<'a>,
	pub col: ColMatrix<'a>,
	pub scale: f64,
}

impl<'a> MatTimesCol<'a> {
	pub fn scale(self, scale: f64) -> Self {
		Self {
			mat: self.mat,
			col: self.col,
			scale: self.scale * scale,
		}
	}
}

impl<'a> std::ops::Mul<RowMatrix<'a>> for ColMatrix<'a> {
	type Output = ColTimesRow<'a>;

	fn mul(self, row: RowMatrix<'a>) -> ColTimesRow<'a> {
		ColTimesRow { col: self, row, scale: 1.0 }
	}
}

impl<'a> std::ops::Mul<ColMatrix<'a>> for Matrix<'a> {
	type Output = MatTimesCol<'a>;

	fn mul(self, col: ColMatrix<'a>) -> MatTimesCol<'a> {
		MatTimesCol { mat: self, col, scale: 1.0 }
	}
}

impl<'a> ClearAccToMatrix for ColTimesRow<'a> {
	#[allow(clippy::panic_in_result_fn)]
	#[inline(never)]
	fn clear_acc_to_matrix(self, to: &Matrix) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			const COL: usize = 0;
			const ROW: usize = 1;

			let Self { col, row, scale } = self;

			assert!(to.batch_dims.is_empty());

			let dims = DimMerger::merge::<1>([col.batch_dims, row.batch_dims])?;

			let mut c_fail = 0;
			let col = generic::Tensor::new_unchecked(
				ND {
					dims: [col.rows, dims[0].get(COL)],
					offset: col.tensor.map().offset,
				},
				DeviceBufferRef::new_unsafe(col.tensor.buf().as_ref(), &mut c_fail),
			);
			let row = generic::Tensor::new_unchecked(
				ND {
					dims: [dims[0].get(ROW), row.cols],
					offset: row.tensor.map().offset,
				},
				DeviceBufferRef::new_unsafe(row.tensor.buf().as_ref(), &mut c_fail),
			);
			let mut m_fail = 0;
			let mut to = generic::Tensor::new_unchecked(
				ND {
					dims: [to.rows, to.cols],
					offset: to.tensor.map().offset,
				},
				DeviceBufferRefMut::new_unsafe(to.tensor.buf().as_ref(), &mut m_fail),
			);
			check_borrows(c_fail, m_fail)?;

			let vmt = col.buf().vmt();
			vmt.mm(&mut to, &col, &row, scale)?;
			Ok(())
		}
	}
}

impl<'a> EvaluatesToColMatrix for MatTimesCol<'a> {
	#[allow(clippy::panic_in_result_fn)]
	#[inline(never)]
	fn eval_to_col_matrix(self, to: &ColMatrix) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			const TO: usize = 0;
			const COL: usize = 1;

			let Self { mat, col, scale } = self;

			assert!(mat.batch_dims.is_empty());

			let dims = DimMerger::merge::<1>([to.batch_dims, col.batch_dims])?;
			let mut c_fail = 0;
			let mat = generic::Tensor::new_unchecked(
				ND {
					dims: [mat.rows, mat.cols],
					offset: mat.tensor.map().offset,
				},
				DeviceBufferRef::new_unsafe(mat.tensor.buf().as_ref(), &mut c_fail),
			);
			let col = generic::Tensor::new_unchecked(
				ND {
					dims: [col.rows, dims[0].get(COL)],
					offset: col.tensor.map().offset,
				},
				DeviceBufferRef::new_unsafe(col.tensor.buf().as_ref(), &mut c_fail),
			);
			let mut m_fail = 0;
			let mut to = generic::Tensor::new_unchecked(
				ND {
					dims: [to.rows, dims[0].get(TO)],
					offset: to.tensor.map().offset,
				},
				DeviceBufferRefMut::new_unsafe(to.tensor.buf().as_ref(), &mut m_fail),
			);
			check_borrows(c_fail, m_fail)?;

			let vmt = mat.buf().vmt();
			vmt.mm(&mut to, &mat, &col, scale)?;
			Ok(())
		}
	}
}

//--------------------------------------------------------------------------------------------------

/*
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

impl<'a> EvaluatesToTensor for Attention<'a> {
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

*/
//--------------------------------------------------------------------------------------------------
