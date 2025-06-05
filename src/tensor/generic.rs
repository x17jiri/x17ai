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

use crate::tensor::generic::map::{Select, Transpose};
use crate::{Error, Result};

//--------------------------------------------------------------------------------------------------
// Tensor

#[derive(Clone)]
pub struct Tensor<M: Map, B: Buffer> {
	pub map: M,
	pub buf: B,
}

impl<M: Map + Copy, B: Buffer + Copy> Copy for Tensor<M, B> {}

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

	/// # Errors
	/// - If the tensor is not K-dimensional
	/// - If any range is invalid (end < start) or out of bounds
	pub fn select<const K: usize>(self, ranges: [Selection; K]) -> Result<Tensor<M::Output, B>>
	where
		M: Select<K>,
	{
		let new_map = self.map.select(ranges)?;
		Ok(Tensor { buf: self.buf, map: new_map })
	}

	/// # Safety
	/// - The tensor must be K-dimensional
	/// - Ranges must be valid (end >= start) and within bounds
	pub unsafe fn select_unchecked<const K: usize>(
		self, ranges: [Selection; K],
	) -> Tensor<M::Output, B>
	where
		M: Select<K>,
	{
		let new_map = self.map.select_unchecked(ranges);
		Tensor { buf: self.buf, map: new_map }
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

	pub fn try_into_tensor<NewMap: Map, NewBuf: Buffer>(self) -> Result<Tensor<NewMap, NewBuf>>
	where
		M: TryInto<NewMap, Error = Error>,
		B: TryInto<NewBuf, Error = Error>,
	{
		let map = self.map.try_into()?;
		let buf = self.buf.try_into()?;
		Ok(Tensor { map, buf })
	}
}

//--------------------------------------------------------------------------------------------------

impl<const K: usize, M: Map + IndexToOffset<K>, T> std::ops::Index<[usize; K]> for Tensor<M, &[T]> {
	type Output = T;

	fn index(&self, index: [usize; K]) -> &Self::Output {
		// Justification for allowing indexing and unwrap:
		// This is implementation of the index operator.
		// It is expected that it may panic if the index is out of bounds.
		#[allow(clippy::indexing_slicing)]
		#[allow(clippy::unwrap_used)]
		{
			let offset = self.map.index_to_offset(index).unwrap();
			&self.buf[offset]
		}
	}
}

impl<'a, M: Map, T> Tensor<M, &'a [T]> {
	/// Returns a slice with the elements of the tensor.
	///
	/// # Errors
	/// - If the tensor is not contiguous, because there would be gaps in the slice.
	pub fn as_slice(&self) -> Result<&'a [T]> {
		if !self.map.is_contiguous() {
			return Err(
				"Cannot view a tensor as a slice because the tensor is not contiguous.".into()
			);
		}
		// Justification for allowing indexing:
		// If tensors are constructed correctly, the tensor's span should always be inside
		// the buffer. So the indexing operation could only panic if there is a bug in the code.
		#[allow(clippy::indexing_slicing)]
		Ok(&self.buf[self.map.span()])
	}

	pub fn is_map_safe(&self) -> bool {
		let span = self.map.span();
		let buf_len = self.buf.len();
		span.start <= span.end && span.end <= buf_len
	}

	/// Returns a slice with the elements of the tensor.
	///
	/// # Safety
	/// - The tensor must be contiguous, otherwise the slice will contain gaps. This can be checked
	///   with `self.map.is_contiguous()`.
	/// - The map is safe, i.e., it never gives an out-of-bounds index. This can be checked with
	///   `self.is_map_safe()`.
	pub unsafe fn as_slice_unchecked(&self) -> &'a [T] {
		debug_assert!(self.map.is_contiguous());
		debug_assert!(self.is_map_safe());
		let span = self.map.span();
		self.buf.get_unchecked(span)
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Selection {
	pub start: Option<usize>,
	pub end: Option<usize>,
}

impl From<usize> for Selection {
	fn from(index: usize) -> Self {
		Self { start: Some(index), end: Some(index + 1) }
	}
}

impl From<std::ops::Range<usize>> for Selection {
	fn from(range: std::ops::Range<usize>) -> Self {
		Self {
			start: Some(range.start),
			end: Some(range.end),
		}
	}
}

impl From<std::ops::RangeInclusive<usize>> for Selection {
	fn from(range: std::ops::RangeInclusive<usize>) -> Self {
		Self {
			start: Some(*range.start()),
			end: Some(*range.end() + 1),
		}
	}
}

impl From<std::ops::RangeFrom<usize>> for Selection {
	fn from(range: std::ops::RangeFrom<usize>) -> Self {
		Self { start: Some(range.start), end: None }
	}
}

impl From<std::ops::RangeTo<usize>> for Selection {
	fn from(range: std::ops::RangeTo<usize>) -> Self {
		Self { start: None, end: Some(range.end) }
	}
}

impl From<std::ops::RangeToInclusive<usize>> for Selection {
	fn from(range: std::ops::RangeToInclusive<usize>) -> Self {
		Self { start: None, end: Some(range.end + 1) }
	}
}

impl From<std::ops::RangeFull> for Selection {
	fn from(_range: std::ops::RangeFull) -> Self {
		Self { start: None, end: None }
	}
}

#[macro_export]
macro_rules! s {
	[$($range:expr),* $(,)?] => {
		[$($crate::tensor::generic::Selection::from($range)),*]
	};
}

//--------------------------------------------------------------------------------------------------
