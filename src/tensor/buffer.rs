// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::rc::Rc;

use super::device::Device;
use super::dtype::DType;
use super::{NonZeroTensorSize, TensorSize};

pub struct Buffer {
	pub device: ManuallyDrop<Rc<dyn Device>>,
	pub device_buffer: NonNull<u8>,
	pub size_bytes: usize,
}

impl Drop for Buffer {
	fn drop(&mut self) {
		self.device.drop_buffer(self);
	}
}

pub struct SliceSet<'a> {
	pub buffer: &'a Buffer,
	pub dtype: DType,
	pub offset: TensorSize,

	pub len: TensorSize,
	pub count: TensorSize,
	pub stride: TensorSize,
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
}

impl Buffer {
	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let my_dev = self.device.as_ref();
		let my_dev = my_dev as *const dyn Device;
		let my_dev = my_dev as *const u8;

		let dev = device as *const dyn Device;
		let dev = dev as *const u8;

		my_dev == dev
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

		slice_set.buffer.is_in_bounds(
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

	/*
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
	*/
}
