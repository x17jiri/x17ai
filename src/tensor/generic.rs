//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

pub mod buffer;
pub mod dim_index;
pub mod map;
pub mod universal_range;

use buffer::Buffer;
use dim_index::DimIndex;
use map::{IndexToOffset, Map, MergeAllDims, MergeDims, ReshapeLastDim};

use crate::tensor::generic::map::{NDShape, Narrow, Select, Transpose};
use crate::tensor::generic::universal_range::UniversalRange;
use crate::{Error, Result};

//--------------------------------------------------------------------------------------------------
// Tensor

#[derive(Clone, Debug)]
pub struct Tensor<M: Map, B: Buffer> {
	pub map: M,
	pub buf: B,
}

impl<M: Map + Copy, B: Buffer + Copy> Copy for Tensor<M, B> {}

impl<M: Map, B: Buffer> Tensor<M, B> {
	pub fn ndim(&self) -> usize {
		self.map.ndim()
	}

	pub fn size(&self, dim: usize) -> Result<usize> {
		if dim >= self.map.ndim() {
			#[cold]
			fn err_dim_out_of_bounds(dim: usize, ndim: usize) -> Error {
				format!("Dimension {dim} is out of bounds for tensor with {ndim} dimensions.")
					.into()
			}
			return Err(err_dim_out_of_bounds(dim, self.ndim()));
		}
		Ok(self.map.size(dim))
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

	pub fn select<D: DimIndex>(&self, dim: D, index: usize) -> Result<Tensor<M::Output, B>>
	where
		M: Select,
		B: Clone,
	{
		let dim = dim.resolve_index(self.ndim())?;
		let new_map = self.map.select(dim, index)?;
		Ok(Tensor { buf: self.buf.clone(), map: new_map })
	}

	pub fn iter_along_axis<D: DimIndex>(&self, dim: D) -> AxisIter<'_, M, B>
	where
		M: Select,
		B: Clone,
	{
		let dim = dim.resolve_index(self.ndim()).unwrap();
		let size = self.size(dim).unwrap();
		AxisIter { tensor: self, dim, current: 0, size }
	}

	pub fn narrow<D: DimIndex, R: Into<UniversalRange>>(
		self, dim: D, range: R,
	) -> Result<Tensor<M::Output, B>>
	where
		M: Narrow,
	{
		let dim = dim.resolve_index(self.ndim())?;
		let range = range.into();
		let new_map = self.map.narrow(dim, range)?;
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

	pub fn nd_shape<const K: usize>(&self) -> std::result::Result<[usize; K], M::Error>
	where
		M: NDShape<K>,
	{
		self.map.nd_shape()
	}

	pub fn conv_map<'a, NewMap>(&'a self) -> std::result::Result<Tensor<NewMap, B>, NewMap::Error>
	where
		NewMap: Map + TryFrom<&'a M>,
		NewMap::Error: Into<crate::Error>,
		B: Clone,
	{
		let map = NewMap::try_from(&self.map)?;
		Ok(Tensor { map, buf: self.buf.clone() })
	}

	pub fn is_contiguous(&self) -> bool {
		self.map.is_contiguous()
	}

	/// # Errors
	/// - If the map is not safe, i.e., if some index may be mapped to an out-of-bounds offset.
	pub fn ensure_safe(&self) -> Result<()> {
		let span = self.map.span();
		let buf_len = self.buf.len();
		let safe = span.start <= span.end && span.end <= buf_len;
		if !safe {
			#[cold]
			fn err_map_not_safe(span: std::ops::Range<usize>, buf_len: usize) -> Error {
				format!(
					"Tensor map is not safe: span {span:?} is out of bounds for buffer of length {buf_len}.",
				)
				.into()
			}
			return Err(err_map_not_safe(span, buf_len));
		}
		Ok(())
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
	/// - If the map is not safe, i.e., it gives an out-of-bounds offset.
	pub fn as_slice(&self) -> Result<&'a [T]> {
		if !self.map.is_contiguous() {
			#[cold]
			fn err_tensor_not_contiguous() -> Error {
				"Cannot view a tensor as a slice because the tensor is not contiguous.".into()
			}
			return Err(err_tensor_not_contiguous());
		}
		let span = self.map.span();
		let Some(slice) = self.buf.get(span.clone()) else {
			#[cold]
			fn err_slice_out_of_bounds(span: std::ops::Range<usize>, buf_len: usize) -> Error {
				format!("Slice {span:?} is out of bounds for buffer of length {buf_len}.",).into()
			}
			return Err(err_slice_out_of_bounds(span, self.buf.len()));
		};
		Ok(slice)
	}

	/// Returns a slice with the elements of the tensor.
	///
	/// # Safety
	/// - contiguous - The tensor must be contiguous, otherwise the slice will contain gaps. This
	///   can be checked with `self.map.is_contiguous()`.
	/// - safe-map - The map is safe, i.e., it never gives an out-of-bounds index. This can be
	///   checked with `self.ensure_safe()`.
	pub unsafe fn as_slice_unchecked(&self) -> &'a [T] {
		debug_assert!(self.is_contiguous());
		debug_assert!(self.clone().ensure_safe().is_ok());
		let span = self.map.span();
		self.buf.get_unchecked(span)
	}
}

//--------------------------------------------------------------------------------------------------

pub struct AxisIter<'a, M: Map + Select, B: Buffer + Clone> {
	tensor: &'a Tensor<M, B>,
	dim: usize,
	current: usize,
	size: usize,
}

impl<M: Map + Select, B: Buffer + Clone> Iterator for AxisIter<'_, M, B> {
	type Item = Tensor<M::Output, B>;

	fn next(&mut self) -> Option<Self::Item> {
		if self.current >= self.size {
			return None;
		}
		let index = self.current;
		self.current += 1;
		Some(Tensor {
			map: unsafe { self.tensor.map.clone().select_unchecked(self.dim, index) },
			buf: self.tensor.buf.clone(),
		})
	}
}

impl<M: Map + Select, B: Buffer + Clone> ExactSizeIterator for AxisIter<'_, M, B> {
	fn len(&self) -> usize {
		self.size - self.current
	}
}

//--------------------------------------------------------------------------------------------------
