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

pub struct DimMerger<const N: usize>;

impl<const N: usize> DimMerger<N> {
	pub fn load_joint_dims(inputs: &[&[SizeAndStride]; N], i: usize) -> [SizeAndStride; N] {
		let i = i + 1;
		inputs.map(|inp| {
			if i <= inp.len() {
				unsafe { *inp.get_unchecked(inp.len() - i) }
			} else {
				SizeAndStride { size: 1, stride: 0 }
			}
		})
	}

	/// Finds common size and resets stride to 0 for broadcasted inputs
	///
	/// If there are no inputs (N == 0), this function always returns size = 1.
	pub fn broadcast_joint_dims<const normalize_strides: bool>(
		dims: [SizeAndStride; N],
	) -> Result<MergedDim<N>, DimsDontMatchError> {
		let size = dims.iter().fold(1, |size, inp| if size == 1 { inp.size } else { size });
		let mut broadcast_check = 0;
		let strides = dims.try_map(|inp| {
			if inp.size == size {
				Ok(inp.stride)
			} else {
				broadcast_check |= inp.size - 1;
				Ok(0)
			}
		})?;
		if broadcast_check != 0 {
			cold_path();
			return Err(DimsDontMatchError);
		}
		if normalize_strides && size <= 1 {
			Ok(MergedDim { size, strides: [0; N] })
		} else {
			Ok(MergedDim { size, strides })
		}
	}

	pub fn load_and_broadcast<const normalize_strides: bool>(
		inputs: &[&[SizeAndStride]; N],
		i: usize,
	) -> Result<MergedDim<N>, DimsDontMatchError> {
		Self::broadcast_joint_dims::<normalize_strides>(Self::load_joint_dims(inputs, i))
	}

	#[allow(clippy::indexing_slicing)]
	#[inline(never)]
	pub fn next<const K: usize>(
		inputs: &[&[SizeAndStride]; N],
		mut i: usize,
	) -> Result<([MergedDim<N>; K], usize, bool), DimsDontMatchError> {
		let n = inputs.iter().map(|inp| inp.len()).max().unwrap_or(0);
		let mut result = [MergedDim { size: 1, strides: [0; N] }; K];
		if K > 0 && i < n {
			let mut k = K - 1;
			let mut merged = MergedDim { size: 1, strides: [0; N] };
			while i < n {
				let joint_dim = Self::load_and_broadcast::<false>(inputs, i)?;

				if merged.size > 1 {
					if joint_dim.size > 1
						&& (0..N).any(|i| joint_dim.strides[i] != merged.size * merged.strides[i])
					{
						// Strides do not match
						// Either start a new dimension, or return
						let slot = unsafe { result.get_unchecked_mut(k) };
						*slot = merged;
						if k == 0 {
							return Ok((result, i, false));
						}
						k -= 1;
						merged = joint_dim;
					} else {
						// Extend the merged dimension
						merged.size *= joint_dim.size;
					}
				} else {
					// If merged.size == 1, this replaces `merged` with `joint_dim`
					// If merged.size == 0, it will not change
					merged.size *= joint_dim.size;
					merged.strides = joint_dim.strides;
				}

				i += 1;
			}

			let slot = unsafe { result.get_unchecked_mut(k) };
			slot.size = merged.size;
			if merged.size > 1 {
				slot.strides = merged.strides;
			}
		}
		Ok((result, i, true))
	}

	pub fn merge<const K: usize>(
		inputs: &[&[SizeAndStride]; N],
		i: usize,
	) -> Result<[MergedDim<N>; K], TensorOpError> {
		cold_path();
		let Ok((result, _new_i, finished)) = Self::next::<K>(inputs, i) else {
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
