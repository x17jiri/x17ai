// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::panic;
use smallvec::{SmallVec, smallvec};
use std::alloc::Layout;
use std::fmt;
use std::intrinsics::{cold_path, likely, unlikely};
use std::iter::ExactSizeIterator;
use std::mem::MaybeUninit;
use std::ops::Index;
use std::rc::{Rc, Weak};
use thin_vec::ThinVec;

pub const INLINE_DIMS: usize = 5;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Default)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

impl SizeAndStride {
	pub fn is_contiguous(&self) -> bool {
		self.stride == 1 || self.size <= 1
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ShapeView<'a> {
	dims: &'a [SizeAndStride],
}

impl<'a> ShapeView<'a> {
	pub fn len(&self) -> usize {
		self.dims.len()
	}
}

impl Index<isize> for ShapeView<'_> {
	type Output = usize;

	fn index(&self, index: isize) -> &Self::Output {
		let i = if index < 0 { self.dims.len() as isize + index } else { index };
		&self.dims[i as usize].size
	}
}

impl Index<usize> for ShapeView<'_> {
	type Output = usize;

	fn index(&self, index: usize) -> &Self::Output {
		&self.dims[index].size
	}
}

/*
impl std::cmp::PartialEq<ShapeView<'_>> for ShapeView<'_> {
	fn eq(&self, other: &ShapeView) -> bool {
		if self.tensor.dims.len() != other.tensor.dims.len() {
			return false;
		}
		for (a, b) in self.tensor.dims.iter().zip(other.tensor.dims.iter()) {
			if a.size != b.size {
				return false;
			}
		}
		true
	}
}
*/

impl<'a> IntoIterator for ShapeView<'a> {
	type Item = &'a usize;
	type IntoIter =
		std::iter::Map<std::slice::Iter<'a, SizeAndStride>, fn(&SizeAndStride) -> &usize>;

	fn into_iter(self) -> Self::IntoIter {
		self.dims.iter().map(|x| &x.size)
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tensor {
	// dims in reverse order
	pub(crate) dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,
	pub(crate) offset: usize,
	pub(crate) dtype: DType,
	pub(crate) elems: usize,
	pub(crate) buffer: Rc<dyn Buffer>,
}

impl Tensor {
	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on<'a, Shape>(shape: Shape, dtype: DType, device: Rc<dyn Device>) -> Tensor
	where
		Shape: IntoIterator<Item = &'a usize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a usize> + ExactSizeIterator,
	{
		let shape = shape.into_iter().copied();
		let builder = NewOutputHandler::new(dtype, device);
		let mut builder = builder.init(shape.len());
		for dim in shape.rev() {
			builder.prepend_dim(dim);
		}
		builder.value()
	}

	/// Allocate a new tensor on the same device as `self`.
	pub fn new_empty<'a, Shape>(&'a self, shape: Shape, dtype: DType) -> Tensor
	where
		Shape: IntoIterator<Item = &'a usize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a usize> + ExactSizeIterator,
	{
		Self::new_empty_on(shape, dtype, self.device())
	}

	/// Allocate a new tensor on the same device with the same shape and dtype
	/// as `self`.
	pub fn new_empty_like(&self) -> Tensor {
		Self::new_empty_on(self.shape(), self.dtype, self.device())
	}

	/// Returns the device on which the tensor is allocated.
	pub fn device(&self) -> Rc<dyn Device> {
		BufferBase::from_dyn_buf(self.buffer.as_ref()).device.clone()
	}

	pub fn shape(&self) -> ShapeView {
		ShapeView { dims: &self.dims }
	}

	/// `dim` should be in the range `0..<ndim`.
	pub fn dim_from_start(&self, dim: usize) -> SizeAndStride {
		let ndim = self.dims.len();
		if dim < ndim { self.dims[dim] } else { SizeAndStride { size: 1, stride: 1 } }
	}

	/// `dim` should be in the range `1..=ndim`.
	pub fn dim_from_end(&self, dim: usize) -> SizeAndStride {
		let ndim = self.dims.len();
		let dim = ndim.wrapping_sub(dim);
		if dim < ndim { self.dims[dim] } else { SizeAndStride { size: 1, stride: 0 } }
	}

	pub fn dim(&self, dim: isize) -> SizeAndStride {
		if dim >= 0 {
			self.dim_from_start(dim.unsigned_abs())
		} else {
			self.dim_from_end(dim.unsigned_abs())
		}
	}

	/// Returns the number of dimensions in the tensor.
	///
	/// This is also known as the rank of the tensor.
	pub fn ndim(&self) -> usize {
		self.dims.len()
	}

	/// Returns the total number of elements in the tensor.
	pub fn elems(&self) -> usize {
		self.elems
	}

	/// Returns the data type of the tensor elements.
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	// Reshapes the last `n_dims_to_reshape` dimensions of the tensor
	pub fn reshape(mut self, n_dims_to_reshape: usize, new_shape: &[usize]) -> Tensor {
		let dims = &mut self.dims;
		let ndim = dims.len();
		let n_dims_to_keep = ndim - n_dims_to_reshape.min(ndim);

		// Merge the dimensions that we are going to reshape
		let dims_to_reshape = &dims[n_dims_to_keep..];
		let merged = DimMerger::new([dims_to_reshape]);

		// Resize the dims array. The new dimensions will be initialized in the `for` loop below.
		unsafe { dims.set_len(n_dims_to_keep) };
		dims.reserve(new_shape.len());
		unsafe { dims.set_len(n_dims_to_keep + new_shape.len()) };
		let new_dims = &mut dims[n_dims_to_keep..];

		// Try to match the new shape with the merged dimensions
		let smallest_dim = merged.smallest_dim();
		let mut prev_stride = smallest_dim.strides[0];
		let mut target_stride = smallest_dim.size * smallest_dim.strides[0];

		let merged_dims = merged.dims_increasing_without_smallest();
		let mut merged_dims = merged_dims.iter();

		for (dims_slot, size) in new_dims.iter_mut().zip(new_shape.iter().copied()).rev() {
			// We are about to store `size` into a slot in `dims`,
			// but first we need to calculate the stride

			let mut new_stride = prev_stride * size;

			if new_stride > target_stride {
				cold_path();

				if prev_stride == target_stride {
					let Some(merged_dim) = merged_dims.next() else {
						panic!("incompatible reshape");
					};
					prev_stride = merged_dim.strides[0];
					target_stride = merged_dim.size * merged_dim.strides[0];

					new_stride = prev_stride * size;
					if new_stride > target_stride {
						panic!("incompatible reshape");
					}
				} else {
					panic!("incompatible reshape");
				}
			}

			*dims_slot = SizeAndStride { size, stride: new_stride };
			prev_stride = new_stride;
		}

		assert!(merged_dims.is_empty());
		assert!(prev_stride == target_stride);

		self
	}

	pub fn reshape_all(self, new_shape: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		self.reshape(ndim, new_shape)
	}

	pub fn dim_to_positive(&self, dim: isize) -> usize {
		let ndim = self.dims.len();
		let dim = if dim >= 0 { dim as usize } else { ndim.wrapping_add(dim as usize) };
		if likely(dim < ndim) {
			dim
		} else {
			panic!("dimension out of range");
		}
	}

	pub fn transposed(mut self, dim1: isize, dim2: isize) -> Tensor {
		let dim1 = self.dim_to_positive(dim1);
		let dim2 = self.dim_to_positive(dim2);

		self.dims.swap(dim1, dim2);
		self
	}

	pub fn permuted(mut self, perm: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		assert!(perm.len() == ndim, "number of dimensions does not match");

		let mut new_dims = SmallVec::with_capacity(ndim);
		unsafe { new_dims.set_len(ndim) };

		let mut sum = 0;
		for (new_dim, p) in new_dims.iter_mut().zip(perm.iter()) {
			*new_dim = self.dims[*p];
			sum += *p;
		}

		let expected_sum = ndim * (ndim - 1) / 2;
		assert!(sum == expected_sum, "invalid permutation");

		self.dims = new_dims;
		self
	}
}

fn fmt_0d(tensor: &Tensor, f: &mut fmt::Formatter, offset: usize) -> fmt::Result {
	let offset = tensor.offset + offset;
	tensor.buffer.format(f, tensor.dtype, offset, 1, 1)
}

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, offset: usize) -> fmt::Result {
	let offset = tensor.offset + offset;
	let dim = tensor.dims[tensor.ndim() - 1];
	write!(f, "[")?;
	tensor.buffer.format(f, tensor.dtype, offset, dim.size, dim.stride)?;
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
