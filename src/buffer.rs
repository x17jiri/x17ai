// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;

pub trait Buffer {
	fn zeros_(&self, tensor: &Tensor);
	fn randn_(&self, tensor: &Tensor);

	fn rms_norm(&self, a: &Tensor, out: &Tensor, params: &ReduceParams);

	fn new_buffer(&self, byte_size: usize) -> Rc<dyn Buffer>;

	//	fn mm(&self, a: &Tensor, b: &Tensor, c: &Tensor);

	fn format(
		&self,
		byte_offset: usize,
		dtype: DType,
		f: &mut fmt::Formatter,
		count: usize,
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
