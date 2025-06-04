//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use super::{IndexToOffset, Map, ND, Transpose};
use crate::Result;

#[derive(Clone, Copy)]
pub struct CompactND<const N: usize>
where
	[(); N - 2]:,
{
	pub shape: [usize; N],
	pub outer_stride: usize,
	pub offset: usize,
}

impl<const N: usize> Map for CompactND<N>
where
	[(); N - 2]:,
{
	fn ndim(&self) -> usize {
		N
	}

	fn elems(&self) -> usize {
		self.shape.iter().product()
	}
}

impl<const N: usize> IndexToOffset<N> for CompactND<N>
where
	[(); N - 2]:,
{
	fn index_to_offset(&self, index: [usize; N]) -> Result<usize> {
		let mut offset = self.offset;
		let mut stride = 1;
		for i in (1..N).rev() {
			let size = self.shape[i];
			if index[i] >= size {
				cold_path();
				return Err(format!(
					"Index {} out of range 0 ..< {} for dimension {}",
					index[i], size, i
				)
				.into());
			}
			offset += index[i] * stride;
			stride *= size;
		}
		offset += index[0] * self.outer_stride;
		Ok(offset)
	}
}

impl<const N: usize> Transpose for CompactND<N>
where
	[(); N - 2]:,
{
	type Output = ND<N>;

	fn transposed(self, d0: usize, d1: usize) -> Result<Self::Output> {
		let res: ND<N> = self.into();
		let res = res.transposed(d0, d1)?;
		Ok(res)
	}
}
