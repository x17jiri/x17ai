//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::tensor::generic::map::SizeAndStride;

const MAX_SPLIT: usize = 3;
const MERGER_DIMS: usize = MAX_SPLIT + 1;

#[derive(Clone, Copy)]
pub struct MergedDim<const N: usize> {
	pub size: usize,
	pub strides: [usize; N],
}

impl<const N: usize> MergedDim<N> {
	pub fn get(&self, i: usize) -> SizeAndStride {
		SizeAndStride { size: self.size, stride: self.strides[i] }
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DimMergerError {
	DimsDontMatch,
	TooManyMergedDimensions,
}

pub struct DimMerger<const N: usize> {
	/// We order dimensions from smallest to largest stride.
	/// This is the reverse order of how they are stored in a Tensor.
	dims: [MergedDim<N>; MERGER_DIMS],
	free_dims: usize,
}

impl<const N: usize> DimMerger<N> {
	pub fn new(inputs: [&[SizeAndStride]; N], first_dim: usize) -> Result<Self, DimMergerError> {
		// Get the max len of the input slices, or 0 if N == 0.
		let ndim = inputs.iter().map(|inp| inp.len()).max().unwrap_or(0);

		// Initialize the first dimension with strides == 1.
		// This way if the real first dimension is contiguous, we will extend the initial
		// value and not take the cold path in the loop.
		let mut dims = [MergedDim { size: 1, strides: [1; N] }; MERGER_DIMS];
		let mut prev_dim_pos = MERGER_DIMS - 1;
		debug_assert!(prev_dim_pos >= first_dim);
		let mut prev_dim = &mut dims[prev_dim_pos];

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

			if next_dim.size > 1 {
				// Can we extend previous dimension?
				if (0..N).all(|i| next_dim.strides[i] == prev_dim.size * prev_dim.strides[i]) {
					// Fast path: Extend the previous dimension
					prev_dim.size *= next_dim.size;
				} else {
					// Slow path: Add a new dimension
					cold_path();
					if prev_dim.size != 1 {
						if prev_dim_pos <= first_dim {
							cold_path();
							return Err(DimMergerError::TooManyMergedDimensions);
						}
						prev_dim_pos -= 1;
						prev_dim = &mut dims[prev_dim_pos];
					}
					*prev_dim = next_dim;
				}
			} else {
				cold_path();
				#[allow(clippy::redundant_else)]
				if next_dim.size < 1 {
					cold_path();
					prev_dim.size = 0;
					prev_dim.strides = [0; N];
					break;
				} else {
					// next_dim.size == 1, we can ignore it
				}
			}
		}

		Ok(Self { dims, free_dims: prev_dim_pos })
	}

	pub fn merge<const K: usize>(
		inputs: [&[SizeAndStride]; N],
	) -> Result<[MergedDim<N>; K], DimMergerError>
	where
		// In `::new()`, we initialize `dims` with MAX_SPLIT elements,
		// so we know there is at least that many guaranteed.
		[(); MAX_SPLIT - K]:,
	{
		let first_dim = MERGER_DIMS - K;
		let merger = Self::new(inputs, K)?;
		Ok(std::array::from_fn(|i| merger.dims[first_dim + i]))
	}
}
