// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;
use std::rc::{Rc, Weak};
use thin_vec::ThinVec;

#[derive(Clone)]
pub struct Tensor {
	pub shape: Shape,
	pub strides: ThinVec<usize>,
	pub dtype: DType,
	pub buffer: Rc<dyn Buffer>,
	pub byte_offset: usize,
}

impl Tensor {
	// Allocate a new tensor on the same device
	pub fn new_tensor(&self, shape: Shape, dtype: DType) -> Tensor {
		// TODO
	}

	pub fn zeros_(&self) {
		self.buffer.zeros_(self);
	}

	pub fn randn_(&self) {
		self.buffer.randn_(self);
	}

	pub fn reshape_last_n(&self, n: usize, replacement: &[usize]) -> Tensor {
		if !self.strides.is_empty() {
			todo!("reshape_last_n() for strided tensors");
		}
		let mut new_shape = self.shape.clone();
		new_shape.replace_last_n(n, replacement);
		if new_shape.elems() != self.shape.elems() {
			panic!("reshape_last_n() must preserve the number of elements");
		}
		Tensor {
			shape: new_shape,
			strides: self.strides.clone(),
			dtype: self.dtype,
			buffer: self.buffer.clone(),
			byte_offset: self.byte_offset,
		}
	}

	// Reshape the tensor to have the new number of dimensions
	//
	// if current ndim <= new_ndim, prepend 1s to the shape to make it the required size
	//
	// if current ndim > new_ndim:
	// - the last (new_ndim - 1) dimensions are unchanged
	// - the dimensions before that are combined into a single big dimension
	//
	// Limitation: Tensors with strides are not supported at the moment
	pub fn as_ndim(&self, new_ndim: usize) -> Tensor {
		if !self.strides.is_empty() {
			todo!("as_ndim() for strided tensors");
		}
		let ndim = self.shape.ndim();
		let mut new_shape = self.shape.clone();
		if ndim <= new_ndim {
			new_shape.prepend_dims(new_ndim - ndim);
		} else {
			new_shape.merge_dims(ndim - new_ndim + 1);
		}
		Tensor {
			shape: new_shape,
			strides: self.strides.clone(),
			dtype: self.dtype,
			buffer: self.buffer.clone(),
			byte_offset: self.byte_offset,
		}
	}

	fn __default_strides(&self) -> ThinVec<usize> {
		let mut strides = ThinVec::new();
		strides.resize(self.shape.ndim(), 0);
		let mut elems = 1;
		let stride_iter = strides.iter_mut().rev();
		let dim_iter = self.shape.iter().rev();
		for (stride, dim) in stride_iter.zip(dim_iter) {
			*stride = elems;
			elems *= *dim;
		}
		strides
	}

	fn __are_default_strides(&self, strides: &[usize]) -> bool {
		let mut elems = 1;
		let stride_iter = strides.iter().rev();
		let dim_iter = self.shape.iter().rev();
		for (stride, dim) in stride_iter.zip(dim_iter) {
			if *stride != elems {
				return false;
			}
			elems *= *dim;
		}
		true
	}

	pub fn transpose(&self, dim1: isize, dim2: isize) -> Tensor {
		let dim1 = self.shape.dim_to_usize(dim1).unwrap();
		let dim2 = self.shape.dim_to_usize(dim2).unwrap();

		let mut new_shape = self.shape.clone();
		new_shape.swap(dim1, dim2);

		let new_strides = if self.strides.is_empty() {
			// create default strides
			let mut s = self.__default_strides();

			// swap
			s.swap(dim1, dim2);

			s
		} else {
			// copy strides
			let mut s = self.strides.clone();
			// swap
			s.swap(dim1, dim2);

			// if the strides are equal to default, we can get rid of them
			if self.__are_default_strides(&new_strides) {
				s = ThinVec::new();
			}

			s
		};

		Tensor {
			shape: new_shape,
			strides: new_strides,
			dtype: self.dtype,
			buffer: self.buffer.clone(),
			byte_offset: self.byte_offset,
		}
	}

	pub fn broadcast(&self, dim: isize, size: usize) -> Tensor {
		let dim = self.shape.dim_to_usize(dim).unwrap();

		if self.shape.__dims[dim] != 1 {
			panic!("broadcasting dimension must have size 1");
		}

		let mut new_shape = self.shape.clone();
		new_shape.__dims[dim] = size;

		let mut new_strides =
			if self.strides.is_empty() { self.__default_strides() } else { self.strides.clone() };
		new_strides[dim] = 0;

		Tensor {
			shape: new_shape,
			strides: new_strides,
			dtype: self.dtype,
			buffer: self.buffer.clone(),
			byte_offset: self.byte_offset,
		}
	}
}

pub fn matmul(m1: &Tensor, m2: &Tensor) -> Tensor {
	// TODO
}

// c = alpha * (a dot b) + beta * c
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
				todo!("Tensor with {} dimensions", ndim);
			},
		};
		write!(f, ")")
	}
}
