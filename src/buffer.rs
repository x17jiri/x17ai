// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;

pub struct BufferBase {
	pub device: Rc<dyn Device>,
	pub size_bytes: usize,
}

pub struct SliceSet {
	pub len: usize,
	pub batch_size: usize,
	pub offset: usize,
	pub batch_stride: usize,
}

pub trait Buffer {
	// If any of the slices represented by a SliceSet are not in bounds,
	// these functions will panic.

	fn zeros(&self, dtype: DType, dst_slices: &SliceSet);
	fn randn(&self, dtype: DType, dst_slices: &SliceSet);

	fn copy(&self, dtype: DType, dst_slices: &SliceSet, src: &BufferBase, src_slices: &SliceSet);
	fn acc(
		&self, dtype: DType, dst_slices: &SliceSet, dst_weight: f64, b: &BufferBase,
		b_slices: &SliceSet, b_weight: f64,
	);

	fn vec_mul(
		&self, dtype: DType, dst_slices: &SliceSet, a: &BufferBase, a_slices: &SliceSet,
		b: &BufferBase, b_slices: &SliceSet,
	);
	fn vec_mul_acc(
		&self, dtype: DType, dst_slices: &SliceSet, dst_weight: f64, a: &BufferBase,
		a_slices: &SliceSet, b: &BufferBase, b_slices: &SliceSet, ab_weight: f64,
	);

	fn rsqrt(
		&self, dtype: DType, dst_slices: &SliceSet, a: &BufferBase, a_slices: &SliceSet, eps: f64,
	);

	fn softmax(&self, dtype: DType, dst_slices: &SliceSet, a: &BufferBase, a_slices: &SliceSet);

	fn rms_norm(&self, slices: &SliceSet<1>, eps: f64);

	// All matrices are stored in row-major order.
	// Example:
	// 	[	1 2 3
	// 		4 5 6	->	[ 1 2 3 4 5 6 7 8 9 ]
	// 		7 8 9 ]
	#[rustfmt::skip]
	#[allow(clippy::too_many_arguments)]
	unsafe fn gemm(
		&self, dtype: DType, c_offset: usize, ldc: usize, c_batch_stride: usize,
		m: usize, n: usize, k: usize,
		a: &BufferBase, a_offset: usize, lda: usize, transa: bool, a_batch_stride: usize,
		b: &BufferBase, b_offset: usize, ldb: usize, transb: bool, b_batch_stride: usize,
		alpha: f64, beta: f64,
		batch_size: usize,
	);

	#[rustfmt::skip]
	unsafe fn format(
		&self, f: &mut fmt::Formatter, dtype: DType,
		offset: usize, count: usize, stride: usize,
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

	pub fn is_in_bounds(&self, dtype: DType, offset: usize, len: usize) -> bool {
		let elems = offset + len;
		let bytes = dtype.array_bytes(elems);
		bytes.is_some_and(|b| b <= self.size_bytes)
	}

	pub fn are_slices_in_bounds(&self, dtype: DType, slice_set: &SliceSet) -> bool {
		if slice_set.batch_size == 0 {
			return true;
		}
		let max_batch = slice_set.batch_size - 1;

		self.is_in_bounds(
			dtype,
			slice_set.offset,
			slice_set.batch_stride * max_batch + slice_set.len,
		)
	}

	pub fn is_in_bounds_T<T>(&self, offset: usize, len: usize) -> bool {
		let elems = offset + len;
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
