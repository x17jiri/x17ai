// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;

pub trait Buffer {
	fn zeros_(&self, tensor: &Tensor);
	fn randn_(&self, tensor: &Tensor);

	unsafe fn rms_norm(
		&self,
		out: &Tensor,
		a: &Tensor,
		dim_size: usize,
		eps: f64,
		batch: &[BatchDim<1>],
	);

	// All matrices are stored in row-major order.
	// Example:
	//     [ 1 2 3
	// A =   4 5 6   ->  [ 1 2 3 4 5 6 7 8 9 ]
	//       7 8 9 ]
	#[rustfmt::skip]
	unsafe fn gemm(
		&self, dtype: DType, c_offset: usize, ldc: usize,
		m: usize, n: usize, k: usize,
		a: &dyn Buffer, a_offset: usize, lda: usize, transa: bool,
		b: &dyn Buffer, b_offset: usize, ldb: usize, transb: bool,
		alpha: f64, beta: f64,
		batch: BatchDim<2>,
	);

	unsafe fn format(
		&self,
		f: &mut fmt::Formatter,
		dtype: DType,
		offset: usize,
		count: usize,
		stride: usize,
	) -> fmt::Result;
}

pub struct BufferBase {
	pub device: Rc<dyn Device>,
	pub capacity: usize,
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
