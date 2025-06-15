//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use log::warn;
use smallvec::{SmallVec, smallvec};
use std::intrinsics::cold_path;

use crate::Result;
use crate::tensor::generic::map::SizeAndStride;

const MERGER_INLINE_DIMS: usize = if crate::tensor::generic::map::dd::INLINE_DIMS > 3 {
	crate::tensor::generic::map::dd::INLINE_DIMS
} else {
	3
};

#[derive(Clone, Copy)]
pub struct MergedDim<const N: usize> {
	pub size: usize,
	pub strides: [usize; N],
}

impl<const N: usize> MergedDim<N> {
	pub fn size_and_stride(&self, i: usize) -> SizeAndStride {
		SizeAndStride { size: self.size, stride: self.strides[i] }
	}
}

#[derive(Clone)]
pub struct DimMerger<const N: usize> {
	/// We order dimensions from smallest to largest stride.
	/// This is the reverse order of how they are stored in a Tensor.
	dims_increasing: SmallVec<[MergedDim<N>; MERGER_INLINE_DIMS]>,
}

impl<const N: usize> DimMerger<N> {
	#[inline(never)]
	pub fn new(inputs: [&[SizeAndStride]; N]) -> Result<DimMerger<N>> {
		// Get the max len of the input slices, or 0 if N == 0.
		let ndim = inputs.iter().map(|inp| inp.len()).max().unwrap_or(0);

		// Initialize the first dimension with strides == 1.
		// This way if the real first dimension is contiguous, we will extend the initial
		// value and not take the cold path in the loop.
		let mut merger = DimMerger {
			dims_increasing: smallvec![MergedDim { size: 1, strides: [1; N] }; MERGER_INLINE_DIMS],
		};
		merger.dims_increasing.truncate(1);
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
			let can_have_arbitrary_strides = next_dim.size <= 1;
			if !can_have_arbitrary_strides
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

		Ok(merger)
	}

	#[inline(never)]
	fn add_dim(&mut self) -> &mut MergedDim<N> {
		// Why warn? I expect this to be really rare event. But if it happens more often,
		// I may remove the warning.
		warn!("DimMerger: adding a dimension");
		self.dims_increasing.push(MergedDim { size: 1, strides: [0; N] });
		self.dims_increasing.last_mut().unwrap()
	}

	pub fn split<const K: usize>(&self) -> (&[MergedDim<N>; K], &[MergedDim<N>])
	where
		// In `::new()`, we initialize `dims_increasing` with MERGER_INLINE_DIMS elements,
		// so we know there is at least that many guaranteed.
		[(); K - MERGER_INLINE_DIMS]:,
	{
		unsafe {
			let result = self.dims_increasing.as_ptr();
			let result = result.cast::<[MergedDim<N>; K]>();
			let result = result.as_ref().unwrap_unchecked();
			let rest = self.dims_increasing.get_unchecked(K..);
			(result, rest)
		}
	}

	pub fn get<const K: usize>(&self) -> &[MergedDim<N>; K]
	where
		[(); K - MERGER_INLINE_DIMS]:,
	{
		let (result, rest) = self.split::<K>();
		assert!(rest.is_empty(), "rest is not empty");
		result
	}
}
