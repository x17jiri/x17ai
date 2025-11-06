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

pub struct DimMerger<'a, const N: usize> {
	inputs: [&'a [SizeAndStride]; N],
	i: usize,
}

impl<'a, const N: usize> DimMerger<'a, N> {
	pub fn new(inputs: [&'a [SizeAndStride]; N]) -> Self {
		let n = inputs.iter().map(|inp| inp.len()).max().unwrap_or(0);
		Self { inputs, i: n }
	}

	pub fn load_joint_dims(inputs: [&[SizeAndStride]; N], i: usize) -> [SizeAndStride; N] {
		inputs.map(|inp| {
			if i < inp.len() {
				unsafe { *inp.get_unchecked(inp.len() - 1 - i) }
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
	pub fn next<const K: usize>(&mut self) -> Result<[MergedDim<N>; K], DimsDontMatchError> {
		let mut result = [MergedDim { size: 1, strides: [0; N] }; K];
		let mut i = self.i;
		if i > 0 && K > 0 {
			i -= 1;
			let mut k = K - 1;
			let mut merged = unsafe { result.get_unchecked_mut(k) };
			*merged = Self::broadcast_joint_dims(Self::load_joint_dims(self.inputs, i))?;
			while i > 0 {
				i -= 1;
				let joint_dim = Self::broadcast_joint_dims(Self::load_joint_dims(self.inputs, i))?;

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
							self.i = i + 1;
							return Ok(result);
						}
					}
					merged.strides = joint_dim.strides;
				}

				merged.size *= joint_dim.size;
			}
			self.i = i;
		}
		Ok(result)
	}

	pub fn is_done(&self) -> bool {
		self.i == 0
	}

	pub fn finish(&self) -> Result<(), TooManyMergedDimensionsError> {
		if self.i == 0 {
			Ok(())
		} else {
			cold_path();
			Err(TooManyMergedDimensionsError)
		}
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
