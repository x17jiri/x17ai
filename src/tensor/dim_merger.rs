//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::intrinsics::cold_path;

use crate::tensor::generic::map::SizeAndStride;

const MAX_SPLIT: usize = 3;

const MERGER_DIMS: usize = MAX_SPLIT + 2;

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

pub enum DimMergerError {
	DimsDontMatch,
	TooManyMergedDimensions,
}

pub struct DimMerger<const N: usize> {
	/// We order dimensions from smallest to largest stride.
	/// This is the reverse order of how they are stored in a Tensor.
	dims_increasing: [MergedDim<N>; MERGER_DIMS],
	ndim: usize,
}

impl<const N: usize> DimMerger<N> {
	#[inline(never)]
	pub fn new(
		inputs: [&[SizeAndStride]; N],
		max_dims: usize,
	) -> Result<DimMerger<N>, DimMergerError> {
		// Get the max len of the input slices, or 0 if N == 0.
		let ndim = inputs.iter().map(|inp| inp.len()).max().unwrap_or(0);

		// Initialize the first dimension with strides == 1.
		// This way if the real first dimension is contiguous, we will extend the initial
		// value and not take the cold path in the loop.
		let mut merger = DimMerger {
			dims_increasing: [MergedDim { size: 1, strides: [1; N] }; MERGER_DIMS],
			ndim: 1,
		};
		let mut prev_dim = &mut merger.dims_increasing[0];

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
			let strides = next_dim.try_map(|inp| {
				if inp.size == size {
					Ok(inp.stride)
				} else {
					if inp.size != 1 {
						cold_path();
						return Err(DimMergerError::DimsDontMatch);
					}
					Ok(0)
				}
			})?;
			let next_dim = MergedDim { size, strides };

			// Do we have to add a new dimension?
			let can_have_arbitrary_strides = next_dim.size <= 1;
			if !can_have_arbitrary_strides
				&& (0..N).any(|i| next_dim.strides[i] != prev_dim.size * prev_dim.strides[i])
			{
				cold_path();
				if prev_dim.size != 1 {
					let max_dims = max_dims.min(MERGER_DIMS);
					if merger.ndim >= max_dims {
						cold_path();
						return Err(DimMergerError::TooManyMergedDimensions);
					}
					prev_dim = &mut merger.dims_increasing[merger.ndim];
					merger.ndim += 1;
				}
				*prev_dim = next_dim;
				continue;
			}

			// Fast path: Extend the previous dimension
			prev_dim.size *= next_dim.size;
		}

		Ok(merger)
	}

	pub fn split<const K: usize>(&self) -> (&[MergedDim<N>; K], &[MergedDim<N>])
	where
		// In `::new()`, we initialize `dims_increasing` with MAX_SPLIT elements,
		// so we know there is at least that many guaranteed.
		[(); MAX_SPLIT - K]:,
	{
		let result = <&[MergedDim<N>; K]>::try_from(&self.dims_increasing[..K]).unwrap();
		let rest_len = self.dims_increasing.len().saturating_sub(K);
		let rest = &self.dims_increasing[K..K + rest_len];
		(result, rest)
	}

	pub fn merge<const K: usize>(
		inputs: [&[SizeAndStride]; N],
	) -> Result<[MergedDim<N>; K], DimMergerError>
	where
		// In `::new()`, we initialize `dims_increasing` with MAX_SPLIT elements,
		// so we know there is at least that many guaranteed.
		[(); MAX_SPLIT - K]:,
	{
		let merger = DimMerger::new(inputs, K)?;
		let (&result, _rest) = merger.split::<K>();
		Ok(result)
	}
}
