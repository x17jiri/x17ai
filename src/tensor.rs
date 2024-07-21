// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;
use std::rc::{Rc, Weak};

#[derive(Clone)]
pub struct Tensor {
	pub shape: Rc<Shape>,
	pub dtype: DType,
	pub buffer: Rc<dyn Buffer>,
	pub byte_offset: usize,
}

impl Tensor {
	pub fn as_expr(&self) -> Rc<Expr> {
		Expr::new_input(self.clone())
	}

	pub fn zeros_(&self) {
		self.buffer.zeros_(self);
	}

	pub fn randn_(&self) {
		self.buffer.randn_(self);
	}

	pub fn reshape_last_n(&self, n: usize, replacement: &[usize]) -> Tensor {
		// TODO
	}

	pub fn transpose(&self, dim1: isize, dim2: isize) -> Tensor {
		// TODO
	}

	pub fn broadcast(&self, dim: isize, size: usize) -> Tensor {
		// TODO
	}
}

pub fn gemm(alpha: f64, a: &Tensor, b: &Tensor, beta: f64, c: &Tensor) {
	// TODO
}

pub fn rms_norm(a: &Tensor, out: &Tensor) {
	// TODO
}

/*
pub struct PrepMM {
	pub batch_size: usize,
	pub a_rows: usize,
	pub a_cols: usize,
	pub b_cols: usize,
	pub a_transpose: bool,
	pub b_transpose: bool,
	pub dtype: DType,
}

pub fn prep_mm(a: &Tensor, b: &Tensor, c: &Tensor) -> PrepMM {
	let (a_batch_dims, a_dims) = a.shape.split(-2);
	let (b_batch_dims, b_dims) = b.shape.split(-2);
	let (c_batch_dims, c_dims) = c.shape.split(-2);

	if a_batch_dims != b_batch_dims || a_batch_dims != c_batch_dims {
		panic!("batch dimensions do not match");
	}
	let batch_size = a_batch_dims.iter().product();

	if a_dims[1] != b_dims[0] || a_dims[0] != c_dims[0] || b_dims[1] != c_dims[1] {
		panic!("matrix dimensions do not match");
	}

	if a.dtype != b.dtype || a.dtype != c.dtype {
		panic!("dtype mismatch");
	}

	PrepMM {
		batch_size,
		a_rows: a_dims[0],
		a_cols: a_dims[1],
		b_cols: b_dims[1],
		a_transpose: false,
		b_transpose: false,
		dtype: a.dtype,
	}
}

// matrix-matrix multiplication
// c = a * b
pub fn m_dot_m(m1: &Tensor, m2: &Tensor, out: &Tensor) {
	a.buffer.mm(m1, m2, out);
}

// vector-matrix multiplication
pub fn v_dot_m(v: &Tensor, m: &Tensor, out: &Tensor) {
	// TODO
}

// matrix-vector multiplication
pub fn m_dot_v(m: &Tensor, v: &Tensor, out: &Tensor) {
	// TODO
}
*/

fn fmt_0d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	let byte_offset = tensor.byte_offset + off * tensor.dtype.bytes();
	tensor.buffer.format(byte_offset, tensor.dtype, f, 1)
}

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	let byte_offset = tensor.byte_offset + off * tensor.dtype.bytes();
	write!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 1];
	tensor.buffer.format(byte_offset, tensor.dtype, f, len)?;
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
		write!(f, "Tensor(")?;
		let ndim = self.shape.ndim();
		match ndim {
			0 => fmt_0d(self, f, 0)?,
			1 => fmt_1d(self, f, 0)?,
			2 => fmt_2d(self, f, 0)?,
			_ => {
				unimplemented!("Tensor with {} dimensions", ndim);
			},
		};
		write!(f, ")")
	}
}
