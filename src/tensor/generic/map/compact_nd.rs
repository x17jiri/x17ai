//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use super::{IndexToOffset, Map, ND, Transpose};
use crate::tensor::generic::Selection;
use crate::tensor::generic::map::Select;
use crate::{Error, Result};

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

	fn span(&self) -> std::ops::Range<usize> {
		// TODO - this will fail if self.outer_stride == 0
		let start = self.offset;
		let len =
			(self.shape[0] - 1) * self.outer_stride + self.shape[1..].iter().product::<usize>();
		let end = start + len;
		start..end
	}

	fn is_contiguous(&self) -> bool {
		self.shape[0] <= 1 || self.outer_stride == self.shape[1..].iter().product()
	}
}

impl<const N: usize> TryFrom<ND<N>> for CompactND<N>
where
	[(); N - 2]:,
{
	type Error = Error;

	fn try_from(nd: ND<N>) -> Result<Self> {
		let mut shape = [0; N];
		let mut stride = 1;
		for i in (1..N).rev() {
			if nd.dims[i].stride != stride {
				#[cold]
				fn err_cannot_convert_nd_to_compact_nd() -> Error {
					"Cannot convert ND with non-contiguous dimensions to CompactND.".into()
				}
				return Err(err_cannot_convert_nd_to_compact_nd());
			}
			shape[i] = nd.dims[i].size;
			stride *= nd.dims[i].size;
		}
		shape[0] = nd.dims[0].size;
		Ok(CompactND {
			shape,
			outer_stride: nd.dims[0].stride,
			offset: nd.offset,
		})
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

impl<const K: usize> Select<K> for CompactND<K>
where
	[(); K - 2]:,
{
	type Output = ND<K>;

	fn select(self, selections: [Selection; K]) -> Result<Self::Output> {
		let mut result = ND::<K>::from(self);
		for k in 0..K {
			let size = self.shape[k];
			let start = selections[k].start.unwrap_or(0);
			let end = selections[k].end.unwrap_or(size);
			if end < start {
				cold_path();
				return Err(format!(
					"Invalid range {start}..{end} for dimension {k} of size {size}",
				)
				.into());
			}
			if end > size {
				cold_path();
				return Err(
					format!("Range {start}..{end} exceeds size {size} for dimension {k}",).into()
				);
			}
			result.offset += start * self.outer_stride;
			result.dims[k].size = end - start;
		}
		Ok(result)
	}

	unsafe fn select_unchecked(self, selections: [Selection; K]) -> Self::Output {
		let mut result = ND::<K>::from(self);
		for k in 0..K {
			let size = self.shape[k];
			let start = selections[k].start.unwrap_or(0);
			let end = selections[k].end.unwrap_or(size);
			debug_assert!(end >= start);
			debug_assert!(end <= size);
			result.offset += start * self.outer_stride;
			result.dims[k].size = end - start;
		}
		result
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
