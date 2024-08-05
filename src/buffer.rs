// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;

/// This struct represents a batch of contiguous slices.
///
/// The slices are stored in a `buffer` at given `offset`.
/// To get to the next slice, we need to add `batch_stride` to the offset.
///
/// Length of the slice and size of the batch are the same for all arguments.
/// We don't want to repeat them and so they are only passed once in `SelfArg`.
pub struct BufOff<Buf> {
	pub buffer: Buf,
	pub offset: usize,
	pub batch_stride: usize,
}

impl<Buf> BufOff<Buf> {
	pub fn without_buf(&self) -> BufOff<()> {
		BufOff {
			buffer: (),
			offset: self.offset,
			batch_stride: self.batch_stride,
		}
	}
}

pub struct CommonArgs1D {
	pub dtype: DType,
	pub len: usize,
	pub batch_size: usize,
}

pub trait Buffer {
	fn zeros_(&self, tensor: &Tensor);
	fn randn_(&self, tensor: &Tensor);

	unsafe fn rms_norm(
		&self,
		o: BufOff<()>,
		a: BufOff<&BufferBase>,
		common: CommonArgs1D,
		eps: f64,
	);

	unsafe fn softmax(
		&self,
		o: BufOff<()>,
		a: BufOff<&BufferBase>,
		common: CommonArgs1D, // rustfmt::newline
	);

	unsafe fn acc(
		&self,
		o: BufOff<()>,
		a: BufOff<&BufferBase>,
		common: CommonArgs1D,
		alpha: f64,
		beta: f64,
	);

	unsafe fn acc_sum(
		&self,
		o: BufOff<()>,
		a: BufOff<&BufferBase>,
		common: CommonArgs1D,
		alpha: f64,
		beta: f64,
	);

	// All matrices are stored in row-major order.
	// Example:
	//     [ 1 2 3
	// A =   4 5 6   ->  [ 1 2 3 4 5 6 7 8 9 ]
	//       7 8 9 ]
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

pub struct BufferBase {
	pub device: Rc<dyn Device>,
	pub capacity: usize,
}

impl BufferBase {
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let my_dev = self.device.as_ref();
		let my_dev = my_dev as *const dyn Device;
		let my_dev = my_dev as *const u8;

		let dev = device as *const dyn Device;
		let dev = dev as *const u8;

		my_dev == dev
	}
}

pub fn buf_to_base(buf: &dyn Buffer) -> &BufferBase {
	let buf = buf as *const dyn Buffer;
	let buf = buf as *const BufferBase;
	unsafe { &*buf }
}

pub fn is_buf_owned_by_device(buf: &dyn Buffer, device: &dyn Device) -> bool {
	let buffer_device = buf_to_base(buf).device.as_ref();
	let buffer_device = buffer_device as *const dyn Device;
	let buffer_device = buffer_device as *const u8;

	let expected_device = device as *const dyn Device;
	let expected_device = expected_device as *const u8;

	buffer_device == expected_device
}

pub fn are_bufs_on_the_same_device(buf1: &dyn Buffer, buf2: &dyn Buffer) -> bool {
	let device1 = buf_to_base(buf1).device.as_ref();
	let device1 = device1 as *const dyn Device;
	let device1 = device1 as *const u8;

	let device2 = buf_to_base(buf2).device.as_ref();
	let device2 = device2 as *const dyn Device;
	let device2 = device2 as *const u8;

	device1 == device2
}
