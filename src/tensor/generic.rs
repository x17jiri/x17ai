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

use std::hint::cold_path;

use buffer::Buffer;
use dim_index::DimIndex;
use map::{Map, MergeAllDims, MergeDims, ReshapeLastDim};

use crate::tensor::generic::dim_index::DimIndexOutOfBoundsError;
use crate::tensor::generic::map::{NDShape, Narrow, Select, SizeAndStride, Transpose};
use crate::tensor::generic::universal_range::UniversalRange;
use crate::{ErrExtra, ErrPack};

//--------------------------------------------------------------------------------------------------
// Tensor

#[derive(Clone, Debug)]
pub struct GenericTensor<M: Map, B: Buffer> {
	map: M,
	buf: B,
}

impl<M: Map + Copy, B: Buffer + Copy> Copy for GenericTensor<M, B> {}

impl<M: Map, B: Buffer> GenericTensor<M, B> {
	pub fn new(map: M, buf: B) -> Option<Self> {
		let map_span = map.span();
		let buf_len = buf.len();
		let safe = map_span.start <= map_span.end && map_span.end <= buf_len;
		if !safe {
			cold_path();
			return None;
		}
		Some(Self { map, buf })
	}

	/// # Safety
	///
	/// The map must be safe, i.e., the span of the map must be within the bounds of the buffer.
	pub unsafe fn new_unchecked(map: M, buf: B) -> Self {
		let map_span = map.span();
		let buf_len = buf.len();
		let safe = map_span.start <= map_span.end && map_span.end <= buf_len;
		debug_assert!(safe);
		Self { map, buf }
	}

	pub fn into_parts(self) -> (M, B) {
		(self.map, self.buf)
	}

	pub fn map(&self) -> &M {
		&self.map
	}

	pub fn buf(&self) -> &B {
		&self.buf
	}

	/// # Safety
	///
	/// The map must not be modified in such a way that the tensor becomes unsafe.
	pub unsafe fn map_mut(&mut self) -> &mut M {
		&mut self.map
	}

	/// # Safety
	///
	/// The buffer must not be modified in such a way that the tensor becomes unsafe.
	pub unsafe fn buf_mut(&mut self) -> &mut B {
		&mut self.buf
	}

	pub fn ndim(&self) -> usize {
		self.map.ndim()
	}

	pub fn dim<D: DimIndex>(&self, dim: D) -> Result<SizeAndStride, DimIndexOutOfBoundsError> {
		let dim = dim.resolve_index(self.ndim())?;
		Ok(self.map.dim(dim))
	}

	pub fn size<D: DimIndex>(&self, dim: D) -> Result<usize, DimIndexOutOfBoundsError> {
		let dim = dim.resolve_index(self.ndim())?;
		Ok(self.map.dim(dim).size)
	}

	pub fn elems(&self) -> usize {
		self.map.elems()
	}

	pub fn merge_dims<const K: usize>(&self) -> Result<GenericTensor<M::Output, B>, M::Error>
	where
		M: MergeDims<K>,
		B: Clone,
	{
		let new_map = self.map.merge_dims()?;
		Ok(GenericTensor { buf: self.buf.clone(), map: new_map })
	}

	pub fn merge_all_dims(&self) -> Result<GenericTensor<M::Output, B>, M::Error>
	where
		M: MergeAllDims,
		B: Clone,
	{
		let new_map = self.map.merge_all_dims()?;
		Ok(GenericTensor { buf: self.buf.clone(), map: new_map })
	}

	pub fn reshape_last_dim<const K: usize>(
		self,
		to_shape: [usize; K],
	) -> Result<GenericTensor<M::Output, B>, M::Error>
	where
		M: ReshapeLastDim<K>,
	{
		let new_map = self.map.reshape_last_dim(to_shape)?;
		Ok(GenericTensor { buf: self.buf, map: new_map })
	}

	pub fn select<D: DimIndex>(
		&self,
		dim: D,
		index: usize,
	) -> Result<GenericTensor<M::Output, B>, M::Error>
	where
		M: Select,
		B: Clone,
	{
		let dim = dim.resolve_index(self.ndim())?;
		let new_map = self.map.select(dim, index)?;
		Ok(GenericTensor { buf: self.buf.clone(), map: new_map })
	}

	pub fn iter_along_axis<'a, D: DimIndex>(
		&'a self,
		dim: D,
	) -> Result<AxisIter<'a, M, B>, DimIndexOutOfBoundsError>
	where
		M: Select,
		B: Clone,
	{
		let dim = dim.resolve_index(self.ndim())?;
		let size = self.size(dim)?;
		Ok(AxisIter { tensor: self, dim, current: 0, size })
	}

	pub fn narrow<D: DimIndex, R: Into<UniversalRange>>(
		self,
		dim: D,
		range: R,
	) -> Result<GenericTensor<M::Output, B>, M::Error>
	where
		M: Narrow,
	{
		let dim = dim.resolve_index(self.ndim())?;
		let range = range.into();
		let new_map = self.map.narrow(dim, range)?;
		Ok(GenericTensor { buf: self.buf, map: new_map })
	}

	pub fn transposed<D0: DimIndex, D1: DimIndex>(
		self,
		d0: D0,
		d1: D1,
	) -> Result<GenericTensor<M::Output, B>, M::Error>
	where
		M: Transpose,
	{
		let d0 = d0.resolve_index(self.ndim())?;
		let d1 = d1.resolve_index(self.ndim())?;
		let new_map = self.map.transposed(d0, d1)?;
		Ok(GenericTensor { buf: self.buf, map: new_map })
	}

	pub fn nd_shape<const K: usize>(&self) -> std::result::Result<[usize; K], M::Error>
	where
		M: NDShape<K>,
	{
		self.map.nd_shape()
	}

	pub fn conv_map<'a, NewMap>(
		&'a self,
	) -> std::result::Result<GenericTensor<NewMap, B>, NewMap::Error>
	where
		NewMap: Map + TryFrom<&'a M>,
		B: Clone,
	{
		let map = NewMap::try_from(&self.map)?;
		Ok(GenericTensor { map, buf: self.buf.clone() })
	}

	pub fn conv_map_ref<NewMap>(
		&self,
	) -> std::result::Result<GenericTensor<NewMap, B>, NewMap::Error>
	where
		NewMap: Map + TryFrom<M>,
		B: Clone,
	{
		let map = NewMap::try_from(self.map.clone())?;
		Ok(GenericTensor { map, buf: self.buf.clone() })
	}

	pub fn ref_map(&self) -> GenericTensor<&M, B>
	where
		B: Clone,
	{
		GenericTensor { map: &self.map, buf: self.buf.clone() }
	}

	/// # Errors
	/// - If the map is not safe, i.e., if some index may be mapped to an out-of-bounds offset.
	pub fn ensure_safe(&self) -> Result<(), ErrPack<TensorUnsafeError>> {
		let span = self.map.span();
		let buf_len = self.buf.len();
		let safe = span.start <= span.end && span.end <= buf_len;
		if !safe {
			return Err(TensorUnsafeError::new(span, buf_len));
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct AxisIter<'a, M: Map + Select, B: Buffer + Clone> {
	tensor: &'a GenericTensor<M, B>,
	dim: usize,
	current: usize,
	size: usize,
}

impl<'a, M: Map + Select, B: Buffer + Clone> Iterator for AxisIter<'a, M, B> {
	type Item = GenericTensor<M::Output, B>;

	fn next(&mut self) -> Option<GenericTensor<M::Output, B>> {
		if self.current >= self.size {
			return None;
		}
		let index = self.current;
		self.current += 1;
		Some(GenericTensor {
			map: unsafe { self.tensor.map.clone().select_unchecked(self.dim, index) },
			buf: self.tensor.buf.clone(),
		})
	}
}

impl<'a, M: Map + Select, B: Buffer + Clone> ExactSizeIterator for AxisIter<'a, M, B> {
	fn len(&self) -> usize {
		self.size - self.current
	}
}

//--------------------------------------------------------------------------------------------------
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TensorUnsafeError;

impl TensorUnsafeError {
	#[cold]
	#[inline(never)]
	fn new(span: std::ops::Range<usize>, buf_len: usize) -> ErrPack<Self> {
		let message = format!(
			"Tensor map is not safe: span {span:?} is out of bounds for buffer of length {buf_len}."
		);
		ErrPack {
			code: Self,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}
}

//--------------------------------------------------------------------------------------------------
