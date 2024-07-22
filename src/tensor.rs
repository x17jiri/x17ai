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

	fn __merge_dims(dims: &[SizeAndStride]) -> SmallVec<[SizeAndStride; MAX_DIMS]> {
		if dims.is_empty() {
			let mut result = SmallVec::new();
			result.push(SizeAndStride { size: 1, stride: 1 });
			return result;
		}

		let mut result = SmallVec::with_capacity(dims.len());
		result.push(dims[0]);
		let mut last = result.last_mut().unwrap();

		for dim in dims[1..].iter().rev() {
			if dim.size == 1 {
				continue;
			}

			if last.stride == (dim.size as isize) * dim.stride {
				last.size *= dim.size;
				last.stride = dim.stride;
			} else {
				result.push(*dim);
				last = result.last_mut().unwrap();
			}
		}
		result
	}

	// Reshape `dims` to the new `shape` and return new dims
	fn __try_reshape(
		dims: &[SizeAndStride],
		shape: &[usize],
	) -> Option<SmallVec<[SizeAndStride; MAX_DIMS]>> {
		let merged = Self::__merge_dims(dims).iter();
		let merged_dim = *merged.next()?;
		let mut acc: isize = merged_dim.stride;
		let mut target: isize = (merged_dim.size as isize) * merged_dim.stride;
		let mut range = -target.abs()..=target.abs();

		let mut result = SmallVec::with_capacity(shape.len());
		unsafe { result.set_len(shape.len()) };

		for (o, i) in result.iter_mut().zip(shape.iter()).rev() {
			let i: isize = i.try_into().ok()?;

			if !range.contains(&(acc * i)) {
				if acc == target {
					merged_dim = *merged.next()?;
					acc = merged_dim.stride;
					target = merged_dim.size * merged_dim.stride;
					range = -target.abs()..=target.abs();
					if !range.contains(&(acc * i)) {
						return None;
					}
				} else {
					return None;
				}
			}
			acc *= i;

			*o = SizeAndStride { size: i as usize, stride: acc };
		}

		Some(result)
	}

	pub fn clone_shape(&self) -> SmallVec<[usize; MAX_DIMS]> {
		let dims = unsafe { self.dims.get_unchecked(..self.ndim as usize) };
		let mut result = SmallVec::new();
		unsafe { result.set_len(self.ndim as usize) };
		for (o, i) in result.iter_mut().zip(dims) {
			*o = i.size;
		}
		result
	}

	pub fn reshape_last_n(&self, n: usize, replacement: &[usize]) -> Tensor {
		let ndim = self.ndim as usize;
		if n > ndim {
			panic!("cannot reshape more dimensions than the tensor has");
		}
		let last_n = unsafe { self.dims.get_unchecked(ndim - n..) };
		let reshape: Option<_> = Self::__try_reshape(last_n, replacement);
		let Some(new_dims) = reshape else {
			panic!("incompatible shape");
		};

		let mut dims = self.dims;
		let b = ndim - n;
		let e = b + new_dims.len();
		if e > MAX_DIMS {
			panic!("too many dimensions");
		}
		let mut slice = unsafe { dims.get_unchecked(b..e) };
		slice.copy_from_slice(&new_dims);

		Tensor {
			ndim: e as u8,
			dims,
			byte_offset: self.byte_offset,
			dtype: self.dtype,
			elems: self.elems,
			buffer: self.buffer.clone(),
		}
	}

	fn __dim_to_usize(&self, dim: isize) -> usize {
		let dim = if dim >= 0 { dim as usize } else { self.ndim as usize - ((-dim) as usize) };
		if likely(dim < self.ndim as usize) {
			dim
		} else {
			panic!("dimension out of range");
		}
	}

	pub fn transposed(&self, dim1: isize, dim2: isize) -> Tensor {
		let dim1 = self.__dim_to_usize(dim1);
		let dim2 = self.__dim_to_usize(dim2);

		let mut new_dims = self.dims;
		new_dims.swap(dim1, dim2);

		Tensor {
			ndim: self.ndim,
			dims: new_dims,
			byte_offset: self.byte_offset,
			dtype: self.dtype,
			elems: self.elems,
			buffer: self.buffer.clone(),
		}
	}

	pub fn t(&self) -> Tensor {
		self.transposed(-2, -1)
	}
}

// result = a * b
pub fn matmul(m1: &Tensor, m2: &Tensor) -> Tensor {
	scaled_matmul(m1, m2, 1.0)
}

// result = (a * b) * scale
pub fn scaled_matmul(m1: &Tensor, m2: &Tensor, scale: f64) -> Tensor {
	// TODO
}

// t = v reinterpretted as a column matrix
// result = (m * t) * scale
pub fn scaled_mat_vec_mul(m: &Tensor, v: &Tensor, scale: f64) -> Tensor {
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
