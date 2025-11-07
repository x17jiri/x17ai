//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::mem::MaybeUninit;

use super::DType;
use super::error::TensorOpError;
use super::map::{Map, SizeAndStride, TensorSizeOverflowError};

//--------------------------------------------------------------------------------------------------

pub trait Shape {
	/// Converts the shape to a `Map` with the specified `dtype`.
	///
	/// The new map is contiguous and has the same number of elements as specified by the shape.
	///
	/// Returns the new map and required buffer size in bytes.
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError>;
}

impl Shape for &Map {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		self.new_like(dtype)
	}
}

impl Shape for &[usize] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		Map::new(self, dtype)
	}
}

impl<const N: usize> Shape for &[usize; N] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		Map::new(self, dtype)
	}
}

impl Shape for &mut [usize] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		Map::new(self, dtype)
	}
}

impl<const N: usize> Shape for &mut [usize; N] {
	fn to_map(self, dtype: DType) -> Result<(Map, usize), TensorSizeOverflowError> {
		Map::new(self, dtype)
	}
}

//--------------------------------------------------------------------------------------------------

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

pub struct DimMerger<'a, const N: usize> {
	inputs: [&'a [SizeAndStride]; N],
	i: usize,
}

impl<'a, const N: usize> DimMerger<'a, N> {
	pub fn new(inputs: [&'a [SizeAndStride]; N]) -> Self {
		Self { inputs, i: 0 }
	}

	pub fn load_joint_dims(
		inputs: [&[SizeAndStride]; N],
		dist_from_end: usize,
	) -> [SizeAndStride; N] {
		debug_assert!(dist_from_end > 0);
		inputs.map(|inp| {
			if dist_from_end <= inp.len() {
				unsafe { *inp.get_unchecked(inp.len() - dist_from_end) }
			} else {
				SizeAndStride { size: 1, stride: 0 }
			}
		})
	}

	/// Finds common size and resets stride to 0 for broadcasted inputs
	///
	/// If there are no inputs (N == 0), this function always returns size = 1.
	pub fn broadcast_joint_dims(
		dims: [SizeAndStride; N],
	) -> Result<MergedDim<N>, DimsDontMatchError> {
		let size = dims.iter().fold(1, |size, inp| if size == 1 { inp.size } else { size });
		let strides = dims.try_map(|inp| {
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

	#[allow(clippy::redundant_else)]
	#[allow(clippy::should_implement_trait)]
	#[allow(clippy::indexing_slicing)]
	#[inline(never)]
	pub fn next<const K: usize>(
		&mut self,
	) -> Result<([MergedDim<N>; K], bool), DimsDontMatchError> {
		let n = self.inputs.iter().map(|inp| inp.len()).max().unwrap_or(0);
		let mut i = n - self.i;
		let mut result = [MergedDim { size: 1, strides: [0; N] }; K];
		if K > 0 && i > 0 {
			i -= 1;
			let mut k = K - 1;
			let mut merged = unsafe { result.get_unchecked_mut(k) };
			*merged = Self::broadcast_joint_dims(Self::load_joint_dims(self.inputs, n - i))?;
			while i > 0 {
				i -= 1;
				let joint_dim =
					Self::broadcast_joint_dims(Self::load_joint_dims(self.inputs, n - i))?;

				if (0..N).any(|i| joint_dim.strides[i] != merged.size * merged.strides[i])
					&& joint_dim.size > 1
				{
					if merged.size > 1 {
						if k > 0 {
							k -= 1;
							merged = unsafe { result.get_unchecked_mut(k) };
							*merged = joint_dim;
							continue;
						} else {
							self.i = n - (i + 1);
							return Ok((result, false));
						}
					}
					merged.strides = joint_dim.strides;
				}

				merged.size *= joint_dim.size;
			}
			self.i = n; // i == 0
		}
		Ok((result, true))
	}

	pub fn merge<const K: usize>(
		inputs: [&'a [SizeAndStride]; N],
	) -> Result<[MergedDim<N>; K], TensorOpError> {
		let mut merger = Self::new(inputs);
		let Ok((result, finished)) = merger.next::<K>() else {
			cold_path();
			return Err(DimsDontMatchError.into());
		};
		if !finished {
			cold_path();
			return Err(TooManyMergedDimensionsError.into());
		}
		Ok(result)
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

//--------------------------------------------------------------------------------------------------

pub unsafe fn reshape_dims(
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

pub fn is_overlapping<const N: usize>(dims: [SizeAndStride; N]) -> bool {
	let mut item_size = 1;
	for dim in dims.iter().rev() {
		if dim.size != 1 {
			if dim.size < 1 {
				return false;
			}
			if dim.stride < item_size {
				return true;
			}
			item_size = dim.size * dim.stride;
		}
	}
	false
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DimsDontMatchError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TooManyMergedDimensionsError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReshapeError;

//--------------------------------------------------------------------------------------------------
