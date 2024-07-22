// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::panic;
use smallvec::{smallvec, SmallVec};
use std::fmt;
use std::intrinsics::{likely, unlikely};
use std::mem::MaybeUninit;
use std::ops::Index;
use std::rc::{Rc, Weak};
use thin_vec::ThinVec;

pub const INLINE_DIMS: usize = 5;

#[derive(Clone, Copy)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

pub struct ShapeView<'a> {
	tensor: &'a Tensor,
}

impl<'a> ShapeView<'a> {
	pub fn len(&self) -> usize {
		self.tensor.dims.len()
	}

	pub fn to_vec(&self) -> SmallVec<[usize; INLINE_DIMS]> {
		self.tensor.dims.iter().map(|x| x.size).collect()
	}
}

impl Index<isize> for ShapeView<'_> {
	type Output = usize;

	fn index(&self, index: isize) -> &Self::Output {
		let index = self.tensor.__dim_to_usize(index);
		unsafe { &self.tensor.dims.get_unchecked(index).size }
	}
}

#[derive(Clone)]
pub struct Tensor {
	dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,
	byte_offset: isize,
	dtype: DType,
	elems: usize,
	buffer: Rc<dyn Buffer>,
}

impl Tensor {
	// Allocate a new tensor on the same device
	pub fn new_tensor(&self, shape: &[usize], dtype: DType) -> Tensor {
		let ndim = shape.len();

		// Total number of elements in the dimensions processed so far
		let mut elems = 1;

		// Total number of elements in the dimensions processed so far,
		// ignoring zero length dimensions.
		let mut nonzero_elems: usize = 1;

		let mut dims = SmallVec::with_capacity(ndim);
		unsafe { dims.set_len(ndim) };

		for (dim, size) in dims.iter_mut().zip(shape.iter().copied()).rev() {
			// Check that if we ignore zero length dimensions, the number of elements
			// does not overflow. This is done to make sure our calculations would not overflow
			// even if we had the same dimensions but in different order.
			if likely(size > 0) {
				if let Some(mul) = nonzero_elems.checked_mul(size) {
					nonzero_elems = mul;
				} else {
					panic!("too many elements");
				};
			}

			// Initialize the dimension
			*dim = SizeAndStride { size, stride: elems };
			elems *= size;
		}

		let check: Option<isize> = nonzero_elems.try_into().ok();
		if check.is_none() {
			panic!("too many elements");
		}

		Tensor {
			dims,
			byte_offset: 0,
			dtype,
			elems,
			buffer: self.buffer.new_buffer(dtype.array_bytes(elems).unwrap()),
		}
	}

	pub fn zeros_(&self) {
		self.buffer.zeros_(self);
	}

	pub fn randn_(&self) {
		self.buffer.randn_(self);
	}

	fn __merge_dims(dims: &[SizeAndStride]) -> SmallVec<[SizeAndStride; INLINE_DIMS]> {
		if dims.is_empty() {
			return smallvec![SizeAndStride { size: 1, stride: 1 }];
		}

		let mut result = smallvec![dims[0]];
		// last = the last dimension pushed to result
		let mut last: &mut SizeAndStride = result.last_mut().unwrap();

		for dim in dims[1..].iter().rev().copied() {
			if dim.size == 1 {
				continue;
			}

			if last.stride == dim.size * dim.stride {
				last.size *= dim.size;
				last.stride = dim.stride;
			} else {
				result.push(dim);
				last = result.last_mut().unwrap();
			}
		}
		result
	}

	// Reshape `dims` to the new `shape` and return new dims
	fn __try_reshape(
		dims: &[SizeAndStride],
		shape: &[usize],
	) -> Option<SmallVec<[SizeAndStride; INLINE_DIMS]>> {
		let merged = Self::__merge_dims(dims);
		let mut merged = merged.iter();

		let merged_dim = *merged.next()?;
		let mut acc = merged_dim.stride;
		let mut target = merged_dim.size * merged_dim.stride;

		let mut result = SmallVec::with_capacity(shape.len());
		unsafe { result.set_len(shape.len()) };

		for (o, size) in result.iter_mut().zip(shape.iter().copied()).rev() {
			if acc * size >= target {
				if acc == target {
					let merged_dim = *merged.next()?;
					acc = merged_dim.stride;
					target = merged_dim.size * merged_dim.stride;

					if acc * size > target {
						cold_path();
						return None;
					}
				} else {
					cold_path();
					return None;
				}
			}

			acc *= size;

			*o = SizeAndStride { size, stride: acc };
		}

		Some(result)
	}

	pub fn shape(&self) -> ShapeView {
		ShapeView { tensor: self }
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn reshape_last_n(&self, n: usize, replacement: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		if n > ndim {
			panic!("cannot reshape more dimensions than the tensor has");
		}
		let last_n = unsafe { self.dims.get_unchecked(ndim - n..) };
		let reshape: Option<_> = Self::__try_reshape(last_n, replacement);
		let Some(new_dims) = reshape else {
			panic!("incompatible shape");
		};

		let new_ndim = ndim - n + new_dims.len();
		let mut dims = SmallVec::with_capacity(new_ndim);
		unsafe { dims.set_len(new_ndim) };

		let mid = ndim - n;
		dims[..mid].copy_from_slice(&self.dims[..mid]);
		dims[mid..].copy_from_slice(&new_dims);

		Tensor {
			dims,
			byte_offset: self.byte_offset,
			dtype: self.dtype,
			elems: self.elems,
			buffer: self.buffer.clone(),
		}
	}

	fn __dim_to_usize(&self, dim: isize) -> usize {
		let ndim = self.dims.len();
		let dim = if dim >= 0 { dim as usize } else { ndim - ((-dim) as usize) };
		if likely(dim < ndim as usize) {
			dim
		} else {
			panic!("dimension out of range");
		}
	}

	pub fn transposed(&self, dim1: isize, dim2: isize) -> Tensor {
		let dim1 = self.__dim_to_usize(dim1);
		let dim2 = self.__dim_to_usize(dim2);

		let mut new_dims = self.dims.clone();
		new_dims.swap(dim1, dim2);

		Tensor {
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
	let mut output_shape = input.shape().to_vec();
	*output_shape.last_mut().unwrap() = self.outputs;
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

//--------------------------------------------------------------------------------------------------

pub fn prep_op<const N: usize>(inputs: [&Shape; N]) -> Option<(Traversal<N>, Shape)> {
	assert!(N > 0);
	let ndim = inputs[0].ndim();
	let perm = inputs[0].perm();

	// Check if the number of dimensions is the same for all inputs
	for n in 1..N {
		if unlikely(inputs[n].ndim() != ndim) {
			return None;
		}
	}

	let mut traversal = Traversal::new(inputs);
	let mut out_shape = inputs[0].clone();

	for perm_index in 0..ndim {
		let i = perm[perm_index] as usize;
		let dim = inputs[0].dims()[i];

		// Set the stride of the output shape so that it is contiguous
		out_shape.dims_mut()[i].stride = traversal.elems as isize;
		if unlikely(dim.stride < 0) {
			out_shape.dims_mut()[i].stride *= -1;
		}

		// Collect the strides of the input shapes
		let mut strides = [0; N];
		for n in 0..N {
			if unlikely(inputs[n].dims()[i].len != dim.len) {
				return None;
			}
			strides[n] = inputs[n].dims()[i].stride;
		}

		// Push the dimension to the traversal
		traversal.push_dim(dim.len, strides);
	}

	traversal.finalize();
	out_shape.__off = traversal.out_off;
	Some((traversal, out_shape))
}

pub fn prep_op_1(input: &Shape) -> (Traversal<1>, Shape) {
	unsafe {
		// SAFETY: prep_op() will fail if inputs are not compatible
		// and we are passing only one input
		prep_op([&input]).unwrap_unchecked()
	}
}

pub struct TraversalDim<const N: usize> {
	pub size: usize,
	pub out_stride: usize,
	pub in_strides: [usize; N],
}

#[derive(Clone)]
pub struct Traversal<const N: usize> {
	pub out_off: usize,
	pub in_off: [isize; N],
	pub dims: SmallVec<[TraversalDim<N>; INLINE_DIMS]>,
	pub elems: usize,
}

impl<const N: usize> Traversal<N> {
	pub fn new(byte_offsets: [isize; N]) -> Traversal<N> {
		Traversal {
			out_off: 0,
			in_off: byte_offsets,
			dims: SmallVec::new(),
			elems: 1,
		}
	}

	pub fn push_dim(&mut self, size: usize, in_strides: [usize; N]) {
		if size == 1 {
			return;
		}

		let out_stride = self.elems;
		self.elems *= size;

		// If there already are dimensions, try to merge the new dimension with the last one
		if !self.dims.is_empty() {
			let prev = self.dims.last_mut().unwrap();

			let mut can_merge = true;
			#[allow(unused_parens)]
			for i in 0..N {
				can_merge &= (in_strides[i] == prev.in_strides[i] * prev.size);
			}

			if can_merge {
				prev.size *= size;
				return;
			}
		}

		// Can't merge, push the new dimension
		self.dims.push(TraversalDim { size, out_stride, in_strides });
	}
}
