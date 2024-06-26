// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::buffer::Buffer;
use crate::shape::Shape;

#[repr(u16)]
#[derive(Debug, Clone, Copy)]
pub enum DType {
	Float,
	Int,
	Uint,
}

pub struct Device {
	pub drop_buffer: fn(buf: *mut Buffer),

	pub name: String,
}

pub struct VMT {
	pub dtype: DType,
	pub type_bits: usize,
	pub dev: *const Device,

	pub zero_: fn(self_: &Self, buf: &mut Buffer, shape: &Shape),
}
