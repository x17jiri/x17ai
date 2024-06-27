// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::shape::Shape;

use std::rc::Rc;

//--------------------------------------------------------------------------------------------------
// DType

#[derive(Debug, Clone, Copy)]
pub enum DType {
	Float,
	Int,
	Uint,
}

//--------------------------------------------------------------------------------------------------
// Device

pub trait Device {
	fn name(&self) -> &str;
	fn dtype(&self) -> (DType, usize); // (dtype, type_bits)

	fn new_buffer(self: Rc<dyn Device>, elems: usize) -> Rc<dyn Buffer>;
}

//--------------------------------------------------------------------------------------------------
// Buffer

pub trait Buffer {
	fn device(&self) -> Rc<dyn Device>;
	fn dtype(&self) -> (DType, usize); // (dtype, type_bits)

	fn zero_(&self, shape: &Shape);
}

//--------------------------------------------------------------------------------------------------
// Tensor

#[derive(Clone, Debug)]
pub struct Tensor {
	buf: Rc<dyn Buffer>,
	shape: Shape,
}

impl Tensor {
	pub fn zero_(&self) {
		self.buf.zero_(&self.shape);
	}
}
