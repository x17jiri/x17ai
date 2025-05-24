// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

pub mod cpu;

use std::fmt;
use std::rc::Rc;

use super::TensorSize;
use super::buffer::Buffer;
use super::dtype::DType;

pub trait Device {
	fn name(&self) -> &str;

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: TensorSize) -> Rc<dyn Buffer>;
}
