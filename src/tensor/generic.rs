//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

pub mod buffer;
pub mod dim_index;
pub mod map;

use buffer::Buffer;
use dim_index::DimIndex;
use map::{IndexToOffset, Map, MergeAllDims, MergeDims, ReshapeLastDim};

use crate::Result;
use crate::tensor::generic::map::Transpose;

//--------------------------------------------------------------------------------------------------
// Tensor

#[derive(Clone)]
pub struct Tensor<M: Map, B: Buffer> {
	pub map: M,
	pub buf: B,
}

impl<M: Map, B: Buffer> Tensor<M, B> {
	pub fn ndim(&self) -> usize {
		self.map.ndim()
	}

	pub fn elems(&self) -> usize {
		self.map.elems()
	}

	pub fn merge_dims<const K: usize>(self) -> Result<Tensor<M::Output, B>>
	where
		M: MergeDims<K>,
	{
		let new_map = self.map.merge_dims()?;
		Ok(Tensor { buf: self.buf, map: new_map })
	}

	pub fn merge_all_dims(self) -> Result<Tensor<M::Output, B>>
	where
		M: MergeAllDims,
	{
		let new_map = self.map.merge_all_dims()?;
		Ok(Tensor { buf: self.buf, map: new_map })
	}

	pub fn reshape_last_dim<const K: usize>(
		self, to_shape: [usize; K],
	) -> Result<Tensor<M::Output, B>>
	where
		M: ReshapeLastDim<K>,
	{
		let new_map = self.map.reshape_last_dim(to_shape)?;
		Ok(Tensor { buf: self.buf, map: new_map })
	}

	pub fn transposed<D0: DimIndex, D1: DimIndex>(
		self, d0: D0, d1: D1,
	) -> Result<Tensor<M::Output, B>>
	where
		M: Transpose,
	{
		let d0 = d0.resolve_index(self.ndim())?;
		let d1 = d1.resolve_index(self.ndim())?;
		let new_map = self.map.transposed(d0, d1)?;
		Ok(Tensor { buf: self.buf, map: new_map })
	}

	pub fn into_tensor<NewMap: Map, NewBuf: Buffer>(self) -> Tensor<NewMap, NewBuf>
	where
		M: Into<NewMap>,
		B: Into<NewBuf>,
	{
		Tensor {
			map: self.map.into(),
			buf: self.buf.into(),
		}
	}
}

//--------------------------------------------------------------------------------------------------

impl<const K: usize, M: Map + IndexToOffset<K>, T> std::ops::Index<[usize; K]> for Tensor<M, &[T]> {
	type Output = T;

	fn index(&self, index: [usize; K]) -> &Self::Output {
		let offset = self.map.index_to_offset(index).unwrap();
		&self.buf[offset]
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SliceRange {
	pub start: Option<usize>,
	pub end: Option<usize>,
}

impl From<std::ops::Range<usize>> for SliceRange {
	fn from(range: std::ops::Range<usize>) -> Self {
		SliceRange {
			start: Some(range.start),
			end: Some(range.end),
		}
	}
}

impl From<std::ops::RangeInclusive<usize>> for SliceRange {
	fn from(range: std::ops::RangeInclusive<usize>) -> Self {
		SliceRange {
			start: Some(*range.start()),
			end: Some(*range.end() + 1), // end is inclusive, so we add 1
		}
	}
}

impl From<std::ops::RangeFrom<usize>> for SliceRange {
	fn from(range: std::ops::RangeFrom<usize>) -> Self {
		SliceRange {
			start: Some(range.start),
			end: None, // No end means it goes to the end of the dimension
		}
	}
}

impl From<std::ops::RangeTo<usize>> for SliceRange {
	fn from(range: std::ops::RangeTo<usize>) -> Self {
		SliceRange {
			start: None, // No start means it starts from the beginning
			end: Some(range.end),
		}
	}
}

impl From<std::ops::RangeToInclusive<usize>> for SliceRange {
	fn from(range: std::ops::RangeToInclusive<usize>) -> Self {
		SliceRange {
			start: None,              // No start means it starts from the beginning
			end: Some(range.end + 1), // end is inclusive, so we add 1
		}
	}
}

impl From<std::ops::RangeFull> for SliceRange {
	fn from(_range: std::ops::RangeFull) -> Self {
		SliceRange { start: None, end: None } // Full range means no limits
	}
}

#[macro_export]
macro_rules! s {
	[$($range:expr),* $(,)?] => {
		[$($crate::tensor::generic::elem_index::SliceRange::from($range)),*]
	};
}

//--------------------------------------------------------------------------------------------------
