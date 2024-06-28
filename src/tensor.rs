// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::shape::Shape;

use std::fmt;
use std::ops::Index;
use std::rc::Rc;

//--------------------------------------------------------------------------------------------------
// DType

#[derive(Debug, Clone, Copy)]
pub enum DType {
	Float(u8),
	Int(u8),
	Uint(u8),
}

//--------------------------------------------------------------------------------------------------
// Device

pub trait Device {
	fn name(&self) -> &str;

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: usize) -> Option<Rc<dyn Buffer>>;
}

//--------------------------------------------------------------------------------------------------
// Buffer

pub trait Buffer {
	fn device(&self) -> Rc<dyn Device>;

	fn dtype(&self) -> DType;

	fn zero_(&self, shape: &Shape);

	// requirement:
	//     index.len() == shape.ndim() - 1
	// This takes indexes for all except the last dimension and
	// prints values in the last dimension.
	fn __format(
		&self,
		f: &mut fmt::Formatter,
		off: isize,
		len: usize,
		stride: isize,
	) -> fmt::Result;
}

//--------------------------------------------------------------------------------------------------
// Tensor

#[derive(Clone)]
pub struct Tensor {
	pub buf: Rc<dyn Buffer>,
	pub shape: Shape,
}

impl Tensor {
	pub fn zero_(&self) {
		self.buf.zero_(&self.shape);
	}
}

//--------------------------------------------------------------------------------------------------
