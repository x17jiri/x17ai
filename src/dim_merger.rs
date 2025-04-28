// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::tensor::INLINE_DIMS;
use crate::*;
use smallvec::SmallVec;
use std::fmt;

#[derive(Clone, Copy)]
pub struct MergedDim<const N: usize> {
	pub size: usize,
	pub out_stride: usize,
	pub in_strides: [usize; N],
}

#[derive(Clone)]
pub struct DimMerger<const N: usize> {
	/// We order dimensions from smallest to largest stride.
	/// This is the reverse order of how they are stored in a Tensor.
	rev_dims: SmallVec<[MergedDim<N>; INLINE_DIMS]>,
}

impl<const N: usize> DimMerger<N> {
	pub fn new<O: OutputHandler>(inputs: [&[SizeAndStride]; N], out: &mut O) -> DimMerger<N> {
		let Some(ndim) = inputs.iter().map(|i| i.len()).max() else {
			// N == 0
			return DimMerger::new_empty(0);
		};
		let mut merger = DimMerger::new_empty(ndim);

		// get iterators over input dimensions
		let mut iterators: [_; N] = inputs.map(|i| i.iter().rev().copied());

		for _ in 0..ndim {
			// Get sizes and strides for the current dimension from all inputs
			let in_dims = std::array::from_fn(|i| {
				iterators[i].next().unwrap_or(SizeAndStride { size: 1, stride: 1 })
			});

			// Only dimensions with size 1 can be broadcasted.
			// Other dimensions cannot be broadcasted and therefore should all be of the same size.
			// Let's get the size of any of the non-broadcastable dimensions.
			// If the non-broadcastable dimensions are not all the same size,
			// we will detect it later.
			let dim_size = in_dims.iter().fold(1, |acc, dim| if acc == 1 { dim.size } else { acc });

			// Get input strides. If needed, try to broadcast
			let in_strides = in_dims.map(|i| {
				if i.size == dim_size {
					i.stride
				} else {
					assert!(i.size == 1, "cannot broadcast: incompatible dimensions");
					0
				}
			});

			let out_stride = out.prepend_dim(dim_size, true);
			merger.prepend_dim(dim_size, out_stride, in_strides);
		}

		merger
	}

	/// The `ndim` parameter is just a hint to preallocate the right amount of space.
	pub fn new_empty(ndim: usize) -> DimMerger<N> {
		let mut rev_dims = SmallVec::with_capacity(ndim);
		rev_dims.push(MergedDim {
			size: 1,
			out_stride: 1,
			in_strides: [1; N],
		});
		DimMerger { rev_dims }
	}

	/// Adds a new dimension and tries to merge it with previous dimensions.
	/// The strides of the new dimension should be >= than any of the previous dimensions.
	///
	/// If the strides are not all >=, merging will not work correctly.
	pub fn prepend_dim(&mut self, size: usize, out_stride: usize, in_strides: [usize; N]) {
		let prev = self.rev_dims.last_mut();
		// SAFETY: In `new_empty()`, we make sure `rev_dims` has at least one element.
		let prev = unsafe { prev.unwrap_unchecked() };

		// Can we extend the previous dimension?
		if size == 1
			|| (out_stride == prev.out_stride * prev.size
				// TODO - verify that the `.all(...)` generates good assembly code
				&& (0..N).all(|i| in_strides[i] == prev.in_strides[i] * prev.size))
		{
			prev.size *= size;
			return;
		}

		// Can we entirely replace the previous dimension?
		if prev.size == 1 {
			prev.size = size;
			prev.out_stride = out_stride;
			prev.in_strides = in_strides;
			return;
		}

		// We can neither extend nor replace the previous dimension. Add the new dimension.
		self.rev_dims.push(MergedDim { size, out_stride, in_strides });
	}

	/// Returns an iterator over the dimensions in the order of increasing strides.
	/// This is the reverse of the order used internally by a Tensor.
	pub fn dims_increasing(&self) -> impl ExactSizeIterator<Item = &MergedDim<N>> {
		self.rev_dims.iter()
	}

	pub fn smallest_dim(&self) -> &MergedDim<N> {
		// SAFETY: In `new_empty()`, we make sure `rev_dims` has at least one element.
		unsafe { self.rev_dims.get_unchecked(0) }
	}

	pub fn dims_increasing_without_smallest(&self) -> impl ExactSizeIterator<Item = &MergedDim<N>> {
		// SAFETY: In `new_empty()`, we make sure `rev_dims` has at least one element.
		unsafe { self.rev_dims.get_unchecked(1..).iter() }
	}

	/// Returns an iterator over the dimensions in the order of decreasing strides.
	/// This is the same order used internally by a Tensor.
	pub fn dims_decreasing(&self) -> impl ExactSizeIterator<Item = &MergedDim<N>> {
		self.rev_dims.iter().rev()
	}
}
