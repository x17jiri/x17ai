//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::mem::MaybeUninit;

use super::map::SizeAndStride;

//--------------------------------------------------------------------------------------------------

// TODO - could this be more efficient if it worked similar to the
// `merge_dims(dims: &[SizeAndStride]) -> (SizeAndStride, &[SizeAndStride])` function?
// I.e., processing as much as possible in 1 dim and returning the rest?

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
pub struct DimsDontMatchError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TooManyMergedDimensionsError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DimMergerError {
	DimsDontMatch,
	TooManyMergedDimensions,
}

impl From<DimsDontMatchError> for DimMergerError {
	fn from(_: DimsDontMatchError) -> Self {
		Self::DimsDontMatch
	}
}

impl From<TooManyMergedDimensionsError> for DimMergerError {
	fn from(_: TooManyMergedDimensionsError) -> Self {
		Self::TooManyMergedDimensions
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReshapeError;

pub struct DimMerger<const N: usize>;

impl<const N: usize> DimMerger<N> {
	/// Finds common size and resets stride to 0 for broadcasted inputs
	///
	/// If there are no inputs (N == 0), this function always returns size = 1.
	pub fn merge_single_dim(dim: [SizeAndStride; N]) -> Result<MergedDim<N>, DimsDontMatchError> {
		let size = dim.iter().fold(1, |size, inp| if size == 1 { inp.size } else { size });
		let strides = dim.try_map(|inp| {
			if inp.size == size {
				Ok(inp.stride)
			} else {
				if inp.size != 1 {
					cold_path();
					return Err(DimsDontMatchError);
				}
				Ok(0)
			}
		})?;
		Ok(MergedDim { size, strides })
	}

	#[inline(never)]
	fn merge_impl(
		inputs: [&[SizeAndStride]; N],
		dims: &mut [MergedDim<N>],
	) -> Result<(), DimMergerError> {
		if dims.is_empty() {
			cold_path();
			return Err(DimMergerError::TooManyMergedDimensions);
		}

		// Get the max len of the input slices, or 0 if N == 0.
		let ndim = inputs.iter().map(|inp| inp.len()).max().unwrap_or(0);

		// We assume the caller initialized `dims.last()` with `size == 1` and
		// `strides = [1; N]`. This way if the real first dimension is contiguous,
		// we will extend the initial value and not take the cold path in the loop.
		let mut prev_dim_pos = dims.len() - 1;
		let mut prev_dim = unsafe { dims.get_unchecked_mut(prev_dim_pos) };

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
			let next_dim = Self::merge_single_dim(next_dim)?;

			if next_dim.size > 1 {
				// Can we extend previous dimension?
				if (0..N).all(|i| next_dim.strides[i] == prev_dim.size * prev_dim.strides[i]) {
					// Fast path: Extend the previous dimension
					prev_dim.size *= next_dim.size;
				} else {
					// Slow path: Add a new dimension
					cold_path();
					if prev_dim.size != 1 {
						if prev_dim_pos == 0 {
							cold_path();
							return Err(DimMergerError::TooManyMergedDimensions);
						}
						prev_dim_pos -= 1;
						prev_dim = unsafe { dims.get_unchecked_mut(prev_dim_pos) };
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

		Ok(())
	}

	pub fn merge<const K: usize>(
		inputs: [&[SizeAndStride]; N],
	) -> Result<[MergedDim<N>; K], DimMergerError> {
		let mut dims = [MergedDim { size: 1, strides: [1; N] }; K];
		Self::merge_impl(inputs, &mut dims)?;
		Ok(dims)
	}
}

//--------------------------------------------------------------------------------------------------

pub fn merge_dims(dims: &[SizeAndStride]) -> (SizeAndStride, &[SizeAndStride]) {
	let mut i = dims.len();
	let mut merged = SizeAndStride { size: 1, stride: 0 };
	if i > 0 {
		i -= 1;
		merged = *unsafe { dims.get_unchecked(i) };
		while i > 0 {
			i -= 1;
			let dim = *unsafe { dims.get_unchecked(i) };

			if dim.stride != merged.size * merged.stride && dim.size > 1 {
				if merged.size > 1 {
					return (merged, unsafe { dims.get_unchecked(0..=i) });
				}
				merged.stride = dim.stride;
			}

			merged.size *= dim.size;
		}
	}
	(merged, unsafe { dims.get_unchecked(0..0) })
}

pub fn reshape_dims(
	(mut inp, mut rest_inp): (SizeAndStride, &[SizeAndStride]),
	to: &[usize],
	dims: *mut MaybeUninit<SizeAndStride>,
) -> Result<(), ReshapeError> {
	let mut j = to.len();
	if j == 0 {
		if inp.size != 1 || !rest_inp.is_empty() {
			cold_path();
			return Err(ReshapeError);
		}
		return Ok(());
	}

	'next_inp: loop {
		let mut acc = SizeAndStride { size: 1, stride: inp.stride };
		'next_out: loop {
			j -= 1;
			let dim_size = *unsafe { to.get_unchecked(j) };
			let Some(mul) = acc.size.checked_mul(dim_size) else {
				cold_path();
				return Err(ReshapeError);
			};

			let w = unsafe { &mut *dims.add(j) };
			w.write(SizeAndStride { size: dim_size, stride: acc.stride });

			if mul == inp.size {
				break 'next_out;
			} else if mul > inp.size || j == 0 {
				cold_path();
				return Err(ReshapeError);
			}
			acc.stride *= dim_size;
			acc.size = mul;
		}
		if j == 0 {
			break 'next_inp;
		}
		(inp, rest_inp) = merge_dims(rest_inp);
	}

	if !rest_inp.is_empty() {
		cold_path();
		return Err(ReshapeError);
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------
