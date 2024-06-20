// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::buffer::Buffer;

#[repr(u16)]
#[derive(Debug, Clone, Copy)]
pub enum DType {
	Float,
	Int,
	Uint,
}

pub struct MasterDevice;

pub struct Device {
	dtype: DType,
	type_bits: usize,
	type_shift: usize,
	master: *const MasterDevice,

	zero_: fn(self_: &Self, buf: *mut Buffer, shape: &Shape),
}
