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
		let index = self.tensor.__dim_to_internal(index);
		unsafe { &self.tensor.dims.get_unchecked(index).size }
	}
}

impl std::cmp::PartialEq<ShapeView<'_>> for ShapeView<'_> {
	fn eq(&self, other: &ShapeView) -> bool {
		self.tensor.dims == other.tensor.dims
	}
}

// This struct is used mainly to ensure that the total number of elements does not overflow
struct DimsConstructor {
	// dimensions in reverse order
	dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,

	remaining_dims: usize,

	// Total number of elements in the dimensions processed so far
	elems: usize,

	// Total number of elements in the dimensions processed so far,
	// ignoring zero length dimensions.
	nonzero_elems: usize,
}

impl DimsConstructor {
	fn from_shape(shape: &[usize]) -> DimsConstructor {
		let mut t = DimsConstructor::new(shape.len());
		for dim in shape.iter().copied().rev() {
			t.push(dim);
		}
		t.final_check();
		t
	}

	fn new(ndim: usize) -> DimsConstructor {
		let mut dims = SmallVec::with_capacity(ndim);
		unsafe { dims.set_len(ndim) };

		DimsConstructor {
			dims,
			remaining_dims: ndim,
			elems: 1,
			nonzero_elems: 1,
		}
	}

	// Push a new dimension with the given size
	// Returns the stride of the new dimension
	fn push(&mut self, size: usize) -> usize {
		debug_assert!(self.remaining_dims > 0);

		// Check that if we ignore zero length dimensions, the number of elements
		// does not overflow. This is done to make sure our calculations would not overflow
		// even if we had the same dimensions but in different order.
		if likely(size != 0) {
			if let Some(mul) = self.nonzero_elems.checked_mul(size) {
				self.nonzero_elems = mul;
			} else {
				panic!("too many elements");
			};
		}

		let stride = self.elems;
		self.elems *= size;

		self.remaining_dims -= 1;
		unsafe {
			*self.dims.get_unchecked_mut(self.remaining_dims) = SizeAndStride { size, stride };
		}

		stride
	}

	fn final_check(&self) {
		debug_assert!(self.remaining_dims == 0);

		// Check that the total number of elements does not overflow isize
		let check: Option<isize> = self.elems.try_into().ok();
		if check.is_none() {
			panic!("too many elements");
		}
	}
}

#[derive(Clone)]
pub struct Tensor {
	// dims in reverse order
	dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,
	pub(crate) offset: usize,
	dtype: DType,
	elems: usize,
	pub(crate) buffer: Rc<dyn Buffer>,
}

impl Tensor {
	pub fn new(shape: &[usize], dtype: DType, device: Rc<dyn Device>) -> Tensor {
		let t = DimsConstructor::from_shape(shape);
		Tensor {
			dims: t.dims,
			offset: 0,
			dtype,
			elems: t.elems,
			buffer: device.new_buffer(dtype.array_bytes(t.elems).unwrap()),
		}
	}

	// Allocate a new tensor on the same device
	pub fn new_tensor(&self, shape: &[usize], dtype: DType) -> Tensor {
		let t = DimsConstructor::from_shape(shape);
		self.__new(t.elems, t.dims, dtype)
	}

	fn __new(
		&self,
		elems: usize,
		dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,
		dtype: DType,
	) -> Tensor {
		Tensor {
			dims,
			offset: 0,
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

	pub fn shape(&self) -> ShapeView {
		ShapeView { tensor: self }
	}

	pub fn ndim(&self) -> usize {
		self.dims.len()
	}

	pub fn elems(&self) -> usize {
		self.elems
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn reshape_last_n(mut self, n: usize, new_shape: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		if n > ndim {
			panic!("cannot reshape more dimensions than the tensor has");
		}

		// use `Batch` to merge the last `n` dimensions
		let last_n = unsafe { self.dims.get_unchecked(ndim - n..) };
		let mut merged = Batch::new(n);
		let mut elems = 1;
		for dim in last_n.iter().rev() {
			merged.push_dim(dim.size, elems, [dim.stride]);
			elems *= dim.size;
		}
		let mut merged = merged.get_rev_dims().iter();

		let merged_dim = merged.next().unwrap();
		let mut prev_stride = merged_dim.in_strides[0];
		let mut target_stride = merged_dim.size * merged_dim.in_strides[0];

		unsafe { self.dims.set_len(ndim - n) };
		self.dims.reserve(new_shape.len());
		let result = unsafe { self.dims.get_unchecked_mut(ndim - n..) };

		for (o, size) in result.iter_mut().zip(new_shape.iter().copied()).rev() {
			if prev_stride * size > target_stride {
				if prev_stride == target_stride {
					let merged_dim = merged.next().unwrap();
					prev_stride = merged_dim.in_strides[0];
					target_stride = merged_dim.size * merged_dim.in_strides[0];

					if prev_stride * size > target_stride {
						panic!("incompatible reshape");
					}
				} else {
					panic!("incompatible reshape");
				}
			}

			prev_stride *= size;

			*o = SizeAndStride { size, stride: prev_stride };
		}

		assert!(merged.next().is_none());
		assert!(prev_stride == target_stride);

		self
	}

	fn __dim_to_internal(&self, dim: isize) -> usize {
		let ndim = self.dims.len();
		let dim = if dim >= 0 { dim as usize } else { ndim - ((-dim) as usize) };
		if likely(dim < ndim as usize) {
			dim
		} else {
			panic!("dimension out of range");
		}
	}

	pub fn transposed(mut self, dim1: isize, dim2: isize) -> Tensor {
		let dim1 = self.__dim_to_internal(dim1);
		let dim2 = self.__dim_to_internal(dim2);

		self.dims.swap(dim1, dim2);
		self
	}

	pub fn t(self) -> Tensor {
		self.transposed(-2, -1)
	}
}

pub fn assert_compatible_types(a: &Tensor, b: &Tensor) {
	if a.dtype != b.dtype {
		panic!("incompatible dtypes");
	}
}

pub fn assert_compatible_devices(a: &Tensor, b: &Tensor) {
	if !are_bufs_on_the_same_device(a.buffer.as_ref(), b.buffer.as_ref()) {
		panic!("incompatible devices");
	}
}

// c = a * b * alpha
pub fn gemm(
	a: &Tensor,
	a_batch_dims: &[SizeAndStride],
	a_dims: [SizeAndStride; 2], // [rows, cols]

	b: &Tensor,
	b_batch_dims: &[SizeAndStride],
	b_dims: [SizeAndStride; 2], // [rows, cols]

	alpha: f64,
) -> Tensor {
	assert_compatible_types(a, b);
	assert_compatible_devices(a, b);

	if a_dims[1].size != b_dims[0].size {
		panic!("incompatible dimensions");
	}
	let c_dims = [
		SizeAndStride {
			size: a_dims[0].size,
			stride: b_dims[1].size,
		},
		SizeAndStride { size: b_dims[1].size, stride: 1 },
	];

	let dtype = a.dtype;
	let batch_ndim = a_batch_dims.len().max(b_batch_dims.len());
	let mut c_dims_ctor = DimsConstructor::new(batch_ndim + 2);
	// Note: dims need to be pushed to ctor in reverse order
	c_dims_ctor.push(c_dims[1].size);
	c_dims_ctor.push(c_dims[0].size);

	let batch = prep_batch(batch_ndim, [&a_dims, &b_dims], &mut c_dims_ctor);
	let batch = batch.get_rev_dims();

	c_dims_ctor.final_check();
	let c = a.__new(c_dims_ctor.elems, c_dims_ctor.dims, dtype);

	let a_rows_contiguous = a_dims[0].stride == 1;
	let a_cols_contiguous = a_dims[1].stride == 1;
	let transa = !a_rows_contiguous;
	let lda = if a_rows_contiguous { a_dims[0].stride } else { a_dims[1].stride };
	assert!(
		a_rows_contiguous || a_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	let b_rows_contiguous = b_dims[0].stride == 1;
	let b_cols_contiguous = b_dims[1].stride == 1;
	let transb = !b_rows_contiguous;
	let ldb = if b_rows_contiguous { b_dims[0].stride } else { b_dims[1].stride };
	assert!(
		b_rows_contiguous || b_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	let m = c_dims[0].size;
	let n = c_dims[1].size;
	let k = a_dims[1].size;

	let c_rows_contiguous = c_dims[0].stride == 1;
	let c_cols_contiguous = c_dims[1].stride == 1;
	let transc = !c_rows_contiguous;
	let ldc = if c_rows_contiguous { c_dims[0].stride } else { c_dims[1].stride };
	assert!(
		c_rows_contiguous || c_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	if !transc {
		unsafe {
			c.buffer.gemm(
				transa, transb, m, n, k, // .
				alpha, a, lda, b, ldb, // .
				0.0, &c, ldc, batch,
			);
		}
	} else {
		// C^T = B^T * A^T
		let (transa, transb) = (!transb, !transa);
		let (m, n) = (n, m);
		let (a, b) = (b, a);
		let (lda, ldb) = (ldb, lda);

		unsafe {
			c.buffer.gemm(
				transa, transb, m, n, k, // .
				alpha, a, lda, b, ldb, // .
				0.0, &c, ldc, batch,
			);
		}
	}

	c
}

pub fn rms_norm(a: &Tensor, out: &Tensor) {
	// TODO
}

fn fmt_0d(tensor: &Tensor, f: &mut fmt::Formatter, offset: usize) -> fmt::Result {
	let offset = tensor.offset + offset;
	unsafe { tensor.buffer.format(f, tensor.dtype, offset, 1, 1) }
}

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, offset: usize) -> fmt::Result {
	let offset = tensor.offset + offset;
	let dim = tensor.dims[tensor.ndim() - 1];
	write!(f, "[")?;
	unsafe { tensor.buffer.format(f, tensor.dtype, offset, dim.size, dim.stride)? };
	write!(f, "]")
}

fn fmt_2d(tensor: &Tensor, f: &mut fmt::Formatter, offset: usize) -> fmt::Result {
	writeln!(f, "[")?;
	let dim = tensor.dims[tensor.ndim() - 2];
	for i in 0..dim.size {
		write!(f, "\t")?;
		fmt_1d(tensor, f, offset + i * dim.stride)?;
		writeln!(f, ",")?;
	}
	write!(f, "]")
}

impl fmt::Display for Tensor {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Tensor(")?;
		match self.ndim() {
			0 => fmt_0d(self, f, 0)?,
			1 => fmt_1d(self, f, 0)?,
			2 => fmt_2d(self, f, 0)?,
			_ => {
				todo!("Tensor with {} dimensions", self.ndim());
			},
		};
		write!(f, ")")
	}
}

//--------------------------------------------------------------------------------------------------

pub fn prep_batch<const N: usize>(
	ndim: usize,
	inputs: [&[SizeAndStride]; N],
	output: &mut DimsConstructor,
) -> Batch<N> {
	assert!(N > 0);

	let mut batch = Batch::new(ndim);

	for d in (0..ndim).rev() {
		// Get sizes and strides for the current dimension from all inputs
		let mut in_sizes = [0; N];
		let mut in_strides = [0; N];
		for i in 0..N {
			// Does the input have enough dimensions,
			// or do we need to extend it with broadcasted dimensions?
			if d < inputs[i].len() {
				in_sizes[i] = inputs[i][d].size;
				in_strides[i] = inputs[i][d].stride;
			} else {
				in_sizes[i] = 1;
				in_strides[i] = 0;
			}
		}

		// TODO - what happens when one of the dimensions has size 0?

		// The dim_size should be the same for all inputs except for broadcasted inputs
		// So max() gets the dim_size of the non-broadcasted inputs.
		let dim_size = in_sizes.iter().copied().max().unwrap();

		let out_stride = output.push(dim_size);

		// Find inputs that need broadcasting and set their strides to 0
		for i in 0..N {
			if in_sizes[i] != dim_size {
				assert!(in_sizes[i] == 1, "cannot broadcast: incompatible dimensions");
				in_strides[i] = 0;
			}
		}

		batch.push_dim(dim_size, out_stride, in_strides);
	}

	batch
}

#[derive(Clone, Copy)]
pub struct BatchDim<const N: usize> {
	pub size: usize,
	pub out_stride: usize,
	pub in_strides: [usize; N],
}

#[derive(Clone)]
pub struct Batch<const N: usize> {
	// dims in batch are in reverse order
	// in other words, from smallest to largest stride
	pub rev_dims: SmallVec<[BatchDim<N>; INLINE_DIMS]>,
}

impl<const N: usize> Batch<N> {
	pub fn new(ndim: usize) -> Batch<N> {
		Batch { rev_dims: SmallVec::with_capacity(ndim) }
	}

	// dimensions should be pushed in the order of increasing strides
	pub fn push_dim(&mut self, size: usize, out_stride: usize, in_strides: [usize; N]) {
		if size == 1 {
			return;
		}

		// If there already are dimensions, try to merge the new dimension with the last one
		if !self.rev_dims.is_empty() {
			let prev = self.rev_dims.last_mut().unwrap();

			let mut can_merge;
			#[allow(unused_parens)]
			{
				can_merge = (out_stride == prev.out_stride * prev.size);
				for i in 0..N {
					can_merge &= (in_strides[i] == prev.in_strides[i] * prev.size);
				}
			}

			if can_merge {
				prev.size *= size;
				return;
			}
		}

		// Can't merge, push the new dimension
		self.rev_dims.push(BatchDim { size, out_stride, in_strides });
	}

	pub fn get_rev_dims(&mut self) -> &[BatchDim<N>] {
		if unlikely(self.rev_dims.is_empty()) {
			self.rev_dims.push(BatchDim {
				size: 1,
				out_stride: 1,
				in_strides: [1; N],
			});
		}
		&self.rev_dims
	}
}
