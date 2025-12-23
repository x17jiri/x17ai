//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::tensor::device::MatMulArgs;
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::error::{ShapeMismatchError, UnsupportedDTypeError};
use crate::tensor::map::{NotEnoughDimensionsError, SizeAndStride};
use crate::tensor::shape::DimMerger;
use crate::tensor::{DType, HasDType, Tensor, TensorOpError};
use crate::util::intrusive_ref_cell::{BorrowFailFlag, IntrusiveRefCellTrait};
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
	let diff_dtype = common_dtype(common_dtype(a.dtype(), b.dtype()), f32::dtype);
	// TODO - `a` may be broadcasted in which case the `diff` shape is wrong
	let diff = a.new_empty_like(diff_dtype)?;
	diff.assign(custom_kernel!(
		diff_dtype,
		[a: a, b: b], (), {
			(a - b).abs()
		}
	))?;
	let diff = diff.merge_all_dims()?;
	let cpu = CPUDevice::new();
	let diff_cpu = diff.to_device(cpu)?;
	println!("diff = {}", &diff_cpu);
	let max = a.new_empty(&[1], diff.dtype())?;
	max.assign(custom_kernel!(
		diff_dtype,
		[diff: &diff], (), {
			diff.max()
		}
	))?;
	let value = max.scalar()?;
	Ok(value <= eps)
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
	let dims = tensor.map().dims();
	if dims.len() < 2 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		#[allow(clippy::indexing_slicing)]
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
	let dims = tensor.map().dims();
	#[allow(clippy::len_zero)]
	if dims.len() < 1 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		#[allow(clippy::indexing_slicing)]
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
	let dims = tensor.map().dims();
	#[allow(clippy::len_zero)]
	if dims.len() < 1 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		#[allow(clippy::indexing_slicing)]
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
			return Err(ShapeMismatchError.into());
		}

		let [col_cols, row_rows] =
			DimMerger::<2>::merge::<1>(&[col.batch_dims, row.batch_dims], 0)?[0].expand();

		let mut borrow_fail = BorrowFailFlag::new();
		let _col_borrow = col.tensor.buf().borrow(&mut borrow_fail);
		let _row_borrow = row.tensor.buf().borrow(&mut borrow_fail);
		borrow_fail.check()?;
		let _to_borrow = to.tensor.buf().try_borrow_mut(0)?;

		let frac = to.tensor.dtype().is_fractional()
			| col.tensor.dtype().is_fractional()
			| row.tensor.dtype().is_fractional();
		if frac != 0 {
			cold_path();
			return Err(UnsupportedDTypeError.into());
		}
		let to_dtype_bytes = to.tensor.dtype().floor_bytes();
		let col_dtype_bytes = col.tensor.dtype().floor_bytes();
		let row_dtype_bytes = row.tensor.dtype().floor_bytes();

		let args = MatMulArgs {
			o_row_stride_bytes: to.rows.stride * to_dtype_bytes,
			o_col_stride_bytes: to.cols.stride * to_dtype_bytes,
			o_rows: to.rows.size,
			o_cols: to.cols.size,
			o_offset_bytes: to.tensor.map().offset() * to_dtype_bytes,
			o_buf: to.tensor.buf().device_ptr(),

			a_row_stride_bytes: col.rows.stride * col_dtype_bytes,
			a_col_stride_bytes: col_cols.stride * col_dtype_bytes,
			// a_rows == o_rows
			a_cols: col_cols.size,
			a_offset_bytes: col.tensor.map().offset() * col_dtype_bytes,
			a_buf: col.tensor.buf().device_ptr(),

			b_row_stride_bytes: row_rows.stride * row_dtype_bytes,
			b_col_stride_bytes: row.cols.stride * row_dtype_bytes,
			// b_rows == a_cols - this condition is ensured by DimMerger
			// b_cols == o_cols
			b_offset_bytes: row.tensor.map().offset() * row_dtype_bytes,
			b_buf: row.tensor.buf().device_ptr(),

			o_dtype: to.tensor.dtype(),
			a_dtype: col.tensor.dtype(),
			b_dtype: row.tensor.dtype(),
			internal_dtype,

			o_buf_bytes: to.tensor.buf().byte_len(),
			a_buf_bytes: col.tensor.buf().byte_len(),
			b_buf_bytes: row.tensor.buf().byte_len(),
		};

		let device = to.tensor.device();
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
			return Err(ShapeMismatchError.into());
		}

		let [to_cols, col_cols] =
			DimMerger::<2>::merge::<1>(&[to.batch_dims, col.batch_dims], 0)?[0].expand();

		let mut borrow_fail = BorrowFailFlag::new();
		let _mat_borrow = mat.tensor.buf().borrow(&mut borrow_fail);
		let _col_borrow = col.tensor.buf().borrow(&mut borrow_fail);
		borrow_fail.check()?;
		let _to_borrow = to.tensor.buf().try_borrow_mut(0)?;

		let frac = to.tensor.dtype().is_fractional()
			| mat.tensor.dtype().is_fractional()
			| col.tensor.dtype().is_fractional();
		if frac != 0 {
			cold_path();
			return Err(UnsupportedDTypeError.into());
		}
		let to_dtype_bytes = to.tensor.dtype().floor_bytes();
		let mat_dtype_bytes = mat.tensor.dtype().floor_bytes();
		let col_dtype_bytes = col.tensor.dtype().floor_bytes();

		let args = MatMulArgs {
			o_row_stride_bytes: to.rows.stride * to_dtype_bytes,
			o_col_stride_bytes: to_cols.stride * to_dtype_bytes,
			o_rows: to.rows.size,
			o_cols: to_cols.size,
			o_offset_bytes: to.tensor.map().offset() * to_dtype_bytes,
			o_buf: to.tensor.buf().device_ptr(),

			a_row_stride_bytes: mat.rows.stride * mat_dtype_bytes,
			a_col_stride_bytes: mat.cols.stride * mat_dtype_bytes,
			// a_rows == o_rows
			a_cols: mat.cols.size,
			a_offset_bytes: mat.tensor.map().offset() * mat_dtype_bytes,
			a_buf: mat.tensor.buf().device_ptr(),

			b_row_stride_bytes: col.rows.stride * col_dtype_bytes,
			b_col_stride_bytes: col_cols.stride * col_dtype_bytes,
			// b_rows == a_cols
			// b_cols == o_cols
			b_offset_bytes: col.tensor.map().offset() * col_dtype_bytes,
			b_buf: col.tensor.buf().device_ptr(),

			o_dtype: to.tensor.dtype(),
			a_dtype: mat.tensor.dtype(),
			b_dtype: col.tensor.dtype(),
			internal_dtype,

			o_buf_bytes: to.tensor.buf().byte_len(),
			a_buf_bytes: mat.tensor.buf().byte_len(),
			b_buf_bytes: col.tensor.buf().byte_len(),
		};

		let device = to.tensor.device();
		unsafe { device.mm(&args, scale) }?;
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
