// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::panic;
use smallvec::SmallVec;
use std::fmt;
use std::intrinsics::likely;
use std::rc::{Rc, Weak};
use thin_vec::ThinVec;

pub const MAX_DIM: usize = 5;

#[derive(Clone, Copy)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: isize,
}

#[derive(Clone)]
pub struct Tensor {
	ndim: u8,
	dims: [SizeAndStride; MAX_DIM],
	byte_offset: isize,
	dtype: DType,
	elems: usize,
	buffer: Rc<dyn Buffer>,
}

impl Tensor {
	// Allocate a new tensor on the same device
	pub fn new_tensor(&self, shape: &[usize], dtype: DType) -> Tensor {
		let ndim = shape.len();
		if ndim > MAX_DIM {
			panic!("too many dimensions");
		}

		let mut dims = [SizeAndStride { size: 0, stride: 0 }; MAX_DIM];

		// Total number of elements in the dimensions processed so far
		let mut elems = 1;

		// Total number of elements in the dimensions processed so far,
		// ignoring zero length dimensions.
		let mut nonzero_elems: isize = 1;

		for i in (0..ndim).rev() {
			// dimension len must be a positive number that fits into isize
			let len: Option<isize> = shape[i].try_into().ok();
			let Some(len) = len else {
				panic!("dimension length does not fit into isize");
			};

			// Check that if we ignore zero length dimensions, the number of elements
			// does not overflow. This is done to make sure our calculations would not overflow
			// even if we had the same dimensions but in different order.
			if likely(len > 0) {
				let Some(mul) = nonzero_elems.checked_mul(len) else {
					panic!("too many elements");
				};
				nonzero_elems = mul;
			}

			// Initialize the dimension
			dims[i] = SizeAndStride { size: len as usize, stride: elems };
			elems *= len;
		}

		Tensor {
			ndim: ndim as u8,
			dims,
			byte_offset: 0,
			dtype,
			elems: elems as usize,
			buffer: self.buffer.new_buffer(elems, dtype),
		}
	}

	pub fn zeros_(&self) {
		self.buffer.zeros_(self);
	}

	pub fn randn_(&self) {
		self.buffer.randn_(self);
	}

	pub fn __merge_dims(&self, dims: &[SizeAndStride]) -> SmallVec<[SizeAndStride; MAX_DIMS]> {
		let mut result = SmallVec::new();
		if dims.is_empty() {
			result.push(SizeAndStride { size: 1, stride: 1 });
			return result;
		}
		result.push(dims[0]);
		for dim in dims[1..].iter().rev() {
			let last = result.last_mut().unwrap();
			if last.stride == (dim.size as isize) * dim.stride {
				last.size *= dim.size;
				last.stride = dim.stride;
			} else {
				result.push(*dim);
			}
		}
		result
	}

	// Reshape `dims` to the new `shape` and return new dims
	fn __reshape(
		&self,
		dims: &[SizeAndStride],
		shape: &[usize],
	) -> SmallVec<[SizeAndStride; MAX_DIMS]> {
		let merged = self.__merge_dims(dims).iter();
		let mut merged_dim = *merged.next().unwrap();
		let mut t = 1;

		let mut result = SmallVec::with_capacity(shape.len());
		unsafe { result.set_len(shape.len()) };

		for (o, i) in result.iter_mut().zip(shape.iter()) {
			/*			t *= *i;
			if t < merged_dim.size {
			} else if t == merged_dim.size {
				*o = merged_dim;
				merged_dim = *merged.next().unwrap();
				t = 1;
			} else {
				*o = SizeAndStride { size: *i, stride: 0 };
			}
			*o = SizeAndStride { size: *i, stride: 0 };*/
		}
		0 // TOOD
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
