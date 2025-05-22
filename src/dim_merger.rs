// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::tensor::INLINE_DIMS;
use crate::*;
use smallvec::{SmallVec, smallvec};
use std::fmt;
use std::intrinsics::{cold_path, likely, unlikely};
/*
//--------------------------------------------------------------------------------------------------
// OutputHandler

pub trait OutputHandler {
	type Impl: OutputHandlerImpl;

	/// `ndim` is the number of dims that will be prepended.
	fn init(self, ndim: usize) -> Self::Impl;
}

pub trait OutputHandlerImpl {
	type Value;

	/// Adds a new dimension with the given size.
	///
	/// Returns the stride of the new dimension.
	fn prepend_dim(&mut self, size: TensorSize);

	/// Checks validity after all dimensions have been added.
	///
	/// No additional dimensions should be added after callig this function.
	fn value(self) -> Self::Value;
}

//--------------------------------------------------------------------------------------------------
// NullOutputHandler

pub struct NullOutputHandler();

impl OutputHandler for NullOutputHandler {
	type Impl = NullOutputHandler;

	fn init(self, _ndim: usize) -> Self::Impl {
		self
	}
}

impl OutputHandlerImpl for NullOutputHandler {
	type Value = ();

	fn prepend_dim(&mut self, _size: TensorSize) {}

	fn value(self) -> Self::Value {}
}

//--------------------------------------------------------------------------------------------------
// NewOutputHandler

pub struct NewOutputHandler {
	dtype: DType,
	device: Rc<dyn Device>,
}

impl NewOutputHandler {
	pub fn new(dtype: DType, device: Rc<dyn Device>) -> NewOutputHandler {
		NewOutputHandler { dtype, device }
	}
}

impl OutputHandler for NewOutputHandler {
	type Impl = NewOutputHandlerImpl;

	fn init(self, ndim: usize) -> Self::Impl {
		let mut dims = SmallVec::with_capacity(ndim);
		unsafe { dims.set_len(ndim) };
		NewOutputHandlerImpl {
			dims,
			remaining_dims: ndim,
			elems: 1,
			nonzero_elems: 1,
			dtype: self.dtype,
			device: self.device,
		}
	}
}

pub struct NewOutputHandlerImpl {
	dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,

	/// How many more dims do we need to prepend.
	remaining_dims: usize,

	elems: TensorSize,

	/// Total number of elements in the dimensions processed so far but ignoring
	/// zero length dimensions.
	nonzero_elems: TensorSize,

	dtype: DType,
	device: Rc<dyn Device>,
}

impl OutputHandlerImpl for NewOutputHandlerImpl {
	type Value = Tensor;

	fn prepend_dim(&mut self, size: TensorSize) {
		assert!(self.remaining_dims > 0);
		// Check that if we ignore zero length dimensions, the number of elements does not
		// overflow. This is done to make sure our calculations would not overflow even if we
		// had the same dimensions but in different order.
		if likely(size != 0) {
			if let Some(mul) = self.nonzero_elems.checked_mul(size) {
				self.nonzero_elems = mul;
			} else {
				panic!("too many elements");
			};
		}

		let stride = self.elems;
		self.elems *= size;

		let dim = SizeAndStride { size, stride };

		self.remaining_dims -= 1;
		*unsafe { self.dims.get_unchecked_mut(self.remaining_dims) } = dim;
	}

	fn value(mut self) -> Self::Value {
		if self.remaining_dims != 0 {
			cold_path();
			for dim in unsafe { self.dims.get_unchecked_mut(0..self.remaining_dims) } {
				*dim = SizeAndStride { size: 1, stride: 0 };
			}
		}

		Tensor {
			dims: DimVec { vec: self.dims }, // TODO - use DimVec for self.dims
			offset: 0,
			dtype: self.dtype,
			elems: self.elems,
			buffer: self.device.new_buffer(self.dtype, self.elems),
		}
	}
}

//--------------------------------------------------------------------------------------------------
*/

#[derive(Clone, Copy)]
pub struct MergedDim<const N: usize> {
	pub size: TensorSize,
	pub strides: [TensorSize; N],
}

impl<const N: usize> MergedDim<N> {
	pub fn size_and_stride(&self, dim: usize) -> SizeAndStride {
		assert!(dim < N);
		SizeAndStride {
			size: self.size,
			stride: self.strides[dim],
		}
	}
}

#[derive(Clone)]
pub struct DimMerger<const N: usize> {
	/// We order dimensions from smallest to largest stride.
	/// This is the reverse order of how they are stored in a Tensor.
	dims_increasing: SmallVec<[MergedDim<N>; INLINE_DIMS]>,
}

pub struct MergedDimList<const N: usize> {
	dims_increasing: SmallVec<[MergedDim<N>; INLINE_DIMS]>,
	start: usize,
}

#[derive(Clone)]
pub struct MergedDimIter<'a, const N: usize> {
	iter: std::slice::Iter<'a, MergedDim<N>>,
}

impl<const N: usize> DimMerger<N> {
	pub fn new(inputs: [&[SizeAndStride]; N]) -> DimMerger<N> {
		let ndim = inputs.iter().fold(1, |acc, input| acc.max(input.len()));

		let dim = inputs.map(|i| i.get(0).copied().unwrap_or(SizeAndStride { size: 1, stride: 0 }));
		let size = dim.iter().fold(1, |acc, d| if acc == 1 { d.size } else { acc });
		let strides = dim.map(|d| {
			assert!(d.size == size || d.size == 1, "cannot broadcast: incompatible dimensions");
			if d.size == size { d.stride } else { 0 }
		});
		let mut dim = MergedDim { size, strides };

		let mut merger = DimMerger { dims_increasing: smallvec![dim] };

		for k in 1..ndim {
			let next_dim =
				inputs.map(|i| i.get(k).copied().unwrap_or(SizeAndStride { size: 1, stride: 0 }));
			let size = next_dim.iter().fold(1, |acc, d| if acc == 1 { d.size } else { acc });
			if size == 1 {
				continue;
			}

			let strides = next_dim.map(|d| {
				assert!(d.size == size || d.size == 1, "cannot broadcast: incompatible dimensions");
				if d.size == size { d.stride } else { 0 }
			});

			// Can we extend the previous dimension?
			// TODO - verify that the `.all(...)` generates good assembly code
			if (0..N).all(|i| strides[i] == dim.strides[i] * dim.size) {
				dim.size *= size;
			} else {
				merger.prepend_dim(dim);
				dim = MergedDim { size, strides };
			}
		}

		unsafe {
			let len = merger.dims_increasing.len();
			let ptr = merger.dims_increasing.as_mut_ptr().add(len);
			ptr.write(dim);
			merger.dims_increasing.set_len(len + 1);
		}

		merger
	}

	#[inline(never)]
	fn prepend_dim(&mut self, dim: MergedDim<N>) {
		self.dims_increasing.reserve(2);
		self.dims_increasing.push(dim);
	}

	pub fn dims_increasing(self) -> MergedDimList<N> {
		MergedDimList {
			dims_increasing: self.dims_increasing,
			start: 0,
		}
	}

	pub fn smallest_dim(&self) -> MergedDim<N> {
		// SAFETY: In `new_empty()`, we make sure `dims_increasing` has at least one element.
		unsafe { *self.dims_increasing.get_unchecked(0) }
	}

	pub fn dims_increasing_without_smallest(self) -> MergedDimList<N> {
		MergedDimList {
			dims_increasing: self.dims_increasing,
			start: 1,
		}
	}
}

impl<const N: usize> MergedDimList<N> {
	pub fn iter(&self) -> MergedDimIter<'_, N> {
		let slice = unsafe { self.dims_increasing.get_unchecked(self.start..) };
		MergedDimIter { iter: slice.iter() }
	}
}

impl<'a, const N: usize> Iterator for MergedDimIter<'a, N> {
	type Item = MergedDim<N>;

	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next().copied()
	}
}

impl<'a, const N: usize> ExactSizeIterator for MergedDimIter<'a, N> {
	fn len(&self) -> usize {
		self.iter.len()
	}
}

impl<'a, const N: usize> DoubleEndedIterator for MergedDimIter<'a, N> {
	fn next_back(&mut self) -> Option<Self::Item> {
		self.iter.next_back().copied()
	}
}

impl<'a, const N: usize> MergedDimIter<'a, N> {
	pub fn is_empty(&self) -> bool {
		self.iter.len() == 0
	}
}
