// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::shape::Shape;
use crate::tensor::*;

use std::fmt;

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, off: isize) -> fmt::Result {
	write!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 1].len;
	let stride = tensor.shape.dims()[ndim - 1].stride;
	tensor.buf.__format(f, off, len, stride)?;
	write!(f, "]")
}

fn fmt_2d(tensor: &Tensor, f: &mut fmt::Formatter, off: isize) -> fmt::Result {
	writeln!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 2].len;
	let stride = tensor.shape.dims()[ndim - 2].stride;
	for i in 0..len {
		write!(f, "\t")?;
		fmt_1d(tensor, f, off + (i as isize) * stride)?;
		writeln!(f, ",")?;
	}
	write!(f, "]")
}

impl fmt::Display for Tensor {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let ndim = self.shape.ndim();
		match ndim {
			0 => {
				// Scalar
				unimplemented!("Scalar");
			},
			1 => fmt_1d(self, f, self.shape.off()),
			2 => fmt_2d(self, f, self.shape.off()),
			_ => {
				unimplemented!("Tensor with {} dimensions", ndim);
			},
		}
	}
}
