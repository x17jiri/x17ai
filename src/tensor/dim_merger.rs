// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use log::warn;
use smallvec::{SmallVec, smallvec};
use std::intrinsics::cold_path;

use super::TensorSize;
use super::dim_vec::{INLINE_DIMS, SizeAndStride};

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
	#[inline(never)]
	pub fn new(inputs: [&[SizeAndStride]; N]) -> DimMerger<N> {
		// Get the max len of the input slices, or 0 if N == 0.
		let ndim = inputs.iter().map(|inp| inp.len()).max().unwrap_or(0);

		// Initialize the first dimension with strides == 1.
		// This way if the real first dimension is contiguous, we will extend the initial
		// value and not take the cold path in the loop.
		let mut merger = DimMerger {
			dims_increasing: smallvec![MergedDim { size: 1, strides: [1; N] }],
		};
		let mut prev_dim = merger.dims_increasing.last_mut().unwrap();

		for index_from_end in 1..=ndim {
			// Get input data. Some inputs may be shorter. We extend them with dummy dimensions.
			let next_dim = inputs.map(|inp| {
				if index_from_end <= inp.len() {
					// SAFETY: index_from_end >= 1 && index_from_end <= inp.len(),
					// Unfortunately, Rust generates bounds check with safe code
					unsafe { *inp.get_unchecked(inp.len() - index_from_end) }
				} else {
					SizeAndStride { size: 1, stride: 0 }
				}
			});

			// Find common size and reset stride to 0 for broadcasted inputs
			let size = next_dim.iter().fold(1, |size, inp| if size == 1 { inp.size } else { size });
			let strides = next_dim.map(|inp| {
				if inp.size == size {
					inp.stride
				} else {
					assert!(inp.size == 1, "dimensions don't match");
					0
				}
			});
			let next_dim = MergedDim { size, strides };

			// Do we have to add a new dimension?
			let are_strides_valid = next_dim.size > 1;
			if are_strides_valid
				&& (0..N).any(|i| next_dim.strides[i] != prev_dim.size * prev_dim.strides[i])
			{
				cold_path();
				if prev_dim.size != 1 {
					cold_path();
					prev_dim = merger.add_dim();
				}
				*prev_dim = next_dim;
				continue;
			}

			// Fast path: Extend the previous dimension
			prev_dim.size *= next_dim.size;
		}

		merger
	}

	#[inline(never)]
	fn add_dim(&mut self) -> &mut MergedDim<N> {
		// Why warn? I expect this to be really rare event. But if it happens more often,
		// I may remove the warning.
		warn!("DimMerger: adding a dimension");
		self.dims_increasing.push(MergedDim { size: 1, strides: [0; N] });
		self.dims_increasing.last_mut().unwrap()
	}

	pub fn dims_increasing(self) -> MergedDimList<N> {
		MergedDimList {
			dims_increasing: self.dims_increasing,
			start: 0,
		}
	}

	pub fn smallest_dim(&self) -> MergedDim<N> {
		// SAFETY: In `new()`, we create `dims_increasing` with at least one element
		// and we never remove elements from it.
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

	pub fn len(&self) -> usize {
		self.dims_increasing.len() - self.start
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
