// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;

pub struct BufferBase {
	pub device: Rc<dyn Device>,
	pub size_bytes: usize,
}

pub trait ToBufferBase {
	fn to_buffer_base(&self) -> &BufferBase;
}

pub struct SliceSet<'a> {
	pub buffer: &'a dyn Buffer,
	pub dtype: DType,
	pub offset: TensorSize,

	pub len: TensorSize,
	pub batch_size: TensorSize,
	pub batch_stride: TensorSize,
}

impl<'a> SliceSet<'a> {
	pub fn to_typed_slice_set<T: ToBufferBase, F: FnOnce(&'a dyn Buffer) -> &'a T>(
		&'a self, f: F,
	) -> TypedSliceSet<'a, T> {
		TypedSliceSet {
			buffer: f(self.buffer),
			dtype: self.dtype,
			offset: self.offset,

			len: self.len,
			batch_size: self.batch_size,
			batch_stride: self.batch_stride,
		}
	}
}

pub struct TypedSliceSet<'a, BufType: ToBufferBase> {
	pub buffer: &'a BufType,
	pub dtype: DType,
	pub offset: TensorSize,

	pub len: TensorSize,
	pub batch_size: TensorSize,
	pub batch_stride: TensorSize,
}

/// All matrices are stored in row-major order.
/// Example:
/// 	[	1 2 3
/// 		4 5 6	->	[ 1 2 3 4 5 6 7 8 9 ]
/// 		7 8 9 ]
pub struct MatrixSet<'a> {
	pub slice_set: SliceSet<'a>,

	pub rows: NonZeroTensorSize,
	pub cols: NonZeroTensorSize,
	pub row_stride: TensorSize,
	pub col_stride: TensorSize,
}

impl<'a> MatrixSet<'a> {
	pub fn slice_len(
		rows: NonZeroTensorSize, cols: NonZeroTensorSize, row_stride: TensorSize,
		col_stride: TensorSize,
	) -> TensorSize {
		(rows.get() - 1) * row_stride + (cols.get() - 1) * col_stride + 1
	}

	pub fn to_typed_matrix_set<T: ToBufferBase, F: FnOnce(&'a dyn Buffer) -> &'a T>(
		&'a self, f: F,
	) -> TypedMatrixSet<'a, T> {
		TypedMatrixSet {
			slice_set: self.slice_set.to_typed_slice_set(f),
			rows: self.rows,
			cols: self.cols,
			row_stride: self.row_stride,
			col_stride: self.col_stride,
		}
	}
}

pub struct TypedMatrixSet<'a, BufType: ToBufferBase> {
	pub slice_set: TypedSliceSet<'a, BufType>,

	pub rows: NonZeroTensorSize,
	pub cols: NonZeroTensorSize,
	pub row_stride: TensorSize,
	pub col_stride: TensorSize,
}

pub trait Buffer {
	// If any of the slices represented by a SliceSet are not in bounds,
	// these functions will panic.

	fn zeros(&self, dst: &SliceSet);

	fn randn(&self, dst: &SliceSet);

	fn copy(&self, dst: &SliceSet, src: &SliceSet);

	fn acc(&self, dst: &SliceSet, dst_weight: f64, b: &SliceSet, b_weight: f64);

	fn mul(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet);

	fn mul_acc(&self, dst: &SliceSet, dst_weight: f64, a: &SliceSet, b: &SliceSet, ab_weight: f64);

	fn sub(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet);

	fn dot(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet);

	fn dot_acc(&self, dst: &SliceSet, dst_weight: f64, a: &SliceSet, b: &SliceSet, ab_weight: f64);

	fn rsqrt(&self, dst: &SliceSet, a: &SliceSet, eps: f64);

	/// Calculates:
	///
	///    dst = max(log(a), -1000, DType.MAX_NEGATIVE);
	///
	/// So the output is defined even for a <= 0.
	fn log_clamped(&self, dst: &SliceSet, a: &SliceSet);

	fn softmax(&self, dst: &SliceSet, a: &SliceSet);

	fn rms_norm(&self, dst: &SliceSet, a: &SliceSet, eps: f64);

	fn gemm(&self, dst: &MatrixSet, dst_weight: f64, a: &MatrixSet, b: &MatrixSet, ab_weight: f64);

	fn format(
		&self, f: &mut fmt::Formatter, dtype: DType, offset: TensorSize, len: TensorSize,
		stride: TensorSize,
	) -> fmt::Result;
}

impl BufferBase {
	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let my_dev = self.device.as_ref();
		let my_dev = my_dev as *const dyn Device;
		let my_dev = my_dev as *const u8;

		let dev = device as *const dyn Device;
		let dev = dev as *const u8;

		my_dev == dev
	}

	#[inline]
	pub fn has_same_device(&self, other: &BufferBase) -> bool {
		other.is_on_device(self.device.as_ref())
	}

	#[inline]
	pub fn from_dyn_buf(buf: &dyn Buffer) -> &BufferBase {
		let buf = buf as *const dyn Buffer;
		let buf = buf as *const BufferBase;
		unsafe { &*buf }
	}

	pub fn is_in_bounds(&self, dtype: DType, offset: TensorSize, len: TensorSize) -> bool {
		let elems = offset + len;
		let bytes = dtype.array_bytes(elems);
		bytes.is_some_and(|b| b <= self.size_bytes)
	}

	pub fn are_slices_in_bounds(slice_set: &SliceSet) -> bool {
		if slice_set.batch_size == 0 {
			return true;
		}
		let max_batch = slice_set.batch_size - 1;

		BufferBase::from_dyn_buf(slice_set.buffer).is_in_bounds(
			slice_set.dtype,
			slice_set.offset,
			slice_set.batch_stride * max_batch + slice_set.len,
		)
	}

	pub fn are_matrices_in_bounds(matrix_set: &MatrixSet) -> bool {
		let slice_len = MatrixSet::slice_len(
			matrix_set.rows,
			matrix_set.cols,
			matrix_set.row_stride,
			matrix_set.col_stride,
		);
		matrix_set.slice_set.len >= slice_len && Self::are_slices_in_bounds(&matrix_set.slice_set)
	}

	pub fn is_in_bounds_T<T>(&self, offset: TensorSize, len: TensorSize) -> bool {
		let elems = tensor_size_to_usize(offset + len);
		let bytes = std::mem::size_of::<T>().checked_mul(elems);
		bytes.is_some_and(|b| b <= self.size_bytes)
	}

	pub fn are_slices_in_bounds_T<T>(&self, slice_set: &SliceSet) -> bool {
		if slice_set.batch_size == 0 {
			return true;
		}
		let max_batch = slice_set.batch_size - 1;

		self.is_in_bounds_T::<T>(
			slice_set.offset,
			slice_set.batch_stride * max_batch + slice_set.len,
		)
	}
}
