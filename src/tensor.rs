// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::{DType, Device, Shape};
use std::fmt;
use std::rc::Rc;

pub trait TensorData {}

pub struct Tensor<Data: ?Sized = dyn TensorData> {
	pub shape: Rc<Shape>,
	pub dtype: DType,
	pub device: Rc<dyn Device>,
	pub data: Data,
}

fn fmt_0d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	tensor.device.format(f, tensor, off, 1, 1)
}

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	write!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 1];
	tensor.device.format(f, tensor, off, len, 1)?;
	write!(f, "]")
}

fn fmt_2d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	writeln!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 2];
	let stride = tensor.shape.dims()[ndim - 1];
	for i in 0..len {
		write!(f, "\t")?;
		fmt_1d(tensor, f, off + i * stride)?;
		writeln!(f, ",")?;
	}
	write!(f, "]")
}

impl fmt::Display for Tensor {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let ndim = self.shape.ndim();
		match ndim {
			0 => fmt_0d(self, f, 0),
			1 => fmt_1d(self, f, 0),
			2 => fmt_2d(self, f, 0),
			_ => {
				unimplemented!("Tensor with {} dimensions", ndim);
			},
		}
	}
}
