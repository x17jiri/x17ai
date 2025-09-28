//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::tensor::device::MatMulArgs;
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::dim_merger::DimMerger;
use crate::tensor::generic::map::{NotEnoughDimensionsError, SizeAndStride};
use crate::tensor::{DType, HasDType, Tensor, TensorOpError};
use crate::util::mycell::UnsafeBorrowFailFlag;
use crate::{ErrPack, custom_kernel};

//--------------------------------------------------------------------------------------------------

pub trait ClearAccToMatrix {
	fn clear_acc_to_matrix(
		self,
		to: &Matrix,
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>>;
}

pub trait EvaluatesToColMatrix {
	fn eval_to_col_matrix(
		self,
		to: &ColMatrix,
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>>;
}

//--------------------------------------------------------------------------------------------------

pub fn approx_eq(a: &Tensor, b: &Tensor, eps: f64) -> Result<bool, ErrPack<TensorOpError>> {
	let diff_dtype = common_dtype(common_dtype(a.dtype(), b.dtype())?, f32::dtype)?;
	// TODO - `a` may be broadcasted in which case the `diff` shape is wrong
	let diff = a.new_empty_like(diff_dtype)?;
	diff.assign(custom_kernel!(
		[a: a, b: b], (), {
			(a - b).abs()
		}
	))?;
	let diff = diff.merge_all_dims()?;
	let max = a.new_empty(&[1], diff.dtype())?;
	max.assign(custom_kernel!(
		[diff: &diff], (), {
			diff.max()
		}
	))?;
	Ok(max.scalar()? <= eps)
}

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
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.clear_acc_to_matrix(self, internal_dtype)
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
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.eval_to_col_matrix(self, internal_dtype)
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
	fn clear_acc_to_matrix(
		self,
		to: &Matrix,
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { col, row, scale } = self;

		if col.rows.size != to.rows.size
			|| row.cols.size != to.cols.size
			|| !to.batch_dims.is_empty()
		{
			cold_path();
			return Err(TensorOpError::shape_mismatch());
		}
		if to.tensor.dtype() != row.tensor.dtype() || to.tensor.dtype() != col.tensor.dtype() {
			cold_path();
			return Err(TensorOpError::dtype_mismatch());
		}
		debug_assert!(col.tensor.ensure_safe().is_ok());
		debug_assert!(row.tensor.ensure_safe().is_ok());
		debug_assert!(to.tensor.ensure_safe().is_ok());

		let dims = DimMerger::merge::<1>([col.batch_dims, row.batch_dims])?;
		let col_cols = dims[0].get(0);
		let row_rows = dims[0].get(1);

		let mut borrow_fail = UnsafeBorrowFailFlag::new();
		let col_borrow = unsafe { col.tensor.buf().unsafe_borrow(&mut borrow_fail) };
		let row_borrow = unsafe { row.tensor.buf().unsafe_borrow(&mut borrow_fail) };
		borrow_fail.check()?;
		let to_borrow = to.tensor.buf().try_borrow_mut()?;

		let args = MatMulArgs {
			o_row_stride: to.rows.stride,
			o_col_stride: to.cols.stride,
			o_rows: to.rows.size,
			o_cols: to.cols.size,
			o_offset: to.tensor.map().offset,
			o_buf: to_borrow.memory(),

			a_row_stride: col.rows.stride,
			a_col_stride: col_cols.stride,
			// a_rows == o_rows
			a_cols: col_cols.size,
			a_offset: col.tensor.map().offset,
			a_buf: col_borrow.memory(),

			b_row_stride: row_rows.stride,
			b_col_stride: row.cols.stride,
			// b_rows == a_cols - this condition is ensured by DimMerger
			// b_cols == o_cols
			b_offset: row.tensor.map().offset,
			b_buf: row_borrow.memory(),

			o_dtype: to.tensor.dtype(),
			a_dtype: col.tensor.dtype(),
			b_dtype: row.tensor.dtype(),
			internal_dtype,

			o_buf_elems: to_borrow.elems(),
			a_buf_elems: col_borrow.elems(),
			b_buf_elems: row_borrow.elems(),
		};

		let device = to_borrow.device();
		unsafe { device.mm(&args, scale) }?;
		Ok(())
	}
}

impl<'a> EvaluatesToColMatrix for MatTimesCol<'a> {
	#[allow(clippy::panic_in_result_fn)]
	#[inline(never)]
	fn eval_to_col_matrix(
		self,
		to: &ColMatrix,
		internal_dtype: DType,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { mat, col, scale } = self;

		#[allow(clippy::suspicious_operation_groupings)]
		if mat.rows.size != to.rows.size
			|| mat.cols.size != col.rows.size
			|| !mat.batch_dims.is_empty()
		{
			cold_path();
			return Err(TensorOpError::shape_mismatch());
		}
		if to.tensor.dtype() != mat.tensor.dtype() || to.tensor.dtype() != col.tensor.dtype() {
			cold_path();
			return Err(TensorOpError::dtype_mismatch());
		}
		debug_assert!(mat.tensor.ensure_safe().is_ok());
		debug_assert!(col.tensor.ensure_safe().is_ok());
		debug_assert!(to.tensor.ensure_safe().is_ok());

		let dims = DimMerger::merge::<1>([to.batch_dims, col.batch_dims])?;
		let to_cols = dims[0].get(0);
		let col_cols = dims[0].get(1);

		let mut borrow_fail = UnsafeBorrowFailFlag::new();
		let mat_borrow = unsafe { mat.tensor.buf().unsafe_borrow(&mut borrow_fail) };
		let col_borrow = unsafe { col.tensor.buf().unsafe_borrow(&mut borrow_fail) };
		borrow_fail.check()?;
		let to_borrow = to.tensor.buf().try_borrow_mut()?;

		let args = MatMulArgs {
			o_row_stride: to.rows.stride,
			o_col_stride: to_cols.stride,
			o_rows: to.rows.size,
			o_cols: to_cols.size,
			o_offset: to.tensor.map().offset,
			o_buf: to_borrow.memory(),

			a_row_stride: mat.rows.stride,
			a_col_stride: mat.cols.stride,
			// a_rows == o_rows
			a_cols: mat.cols.size,
			a_offset: mat.tensor.map().offset,
			a_buf: mat_borrow.memory(),

			b_row_stride: col.rows.stride,
			b_col_stride: col_cols.stride,
			// b_rows == a_cols
			// b_cols == o_cols
			b_offset: col.tensor.map().offset,
			b_buf: col_borrow.memory(),

			o_dtype: to.tensor.dtype(),
			a_dtype: mat.tensor.dtype(),
			b_dtype: col.tensor.dtype(),
			internal_dtype,

			o_buf_elems: to_borrow.elems(),
			a_buf_elems: mat_borrow.elems(),
			b_buf_elems: col_borrow.elems(),
		};

		let device = to_borrow.device();
		unsafe { device.mm(&args, scale) }?;
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
