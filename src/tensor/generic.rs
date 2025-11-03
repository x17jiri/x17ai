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

use std::borrow::Cow;
use std::hint::cold_path;

use buffer::Buffer;
use dim_index::DimIndex;

use crate::tensor::generic::dim_index::DimIndexOutOfBoundsError;
use crate::tensor::generic::map::SizeAndStride;
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

	pub fn nd_shape<const K: usize>(&self) -> std::result::Result<[usize; K], M::Error>
	where
		M: NDShape<K>,
	{
		self.map.nd_shape()
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
//--------------------------------------------------------------------------------------------------
