//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

pub mod compact_nd;
pub mod dyn_d;
pub mod nd;

pub use compact_nd::CompactND;
pub use dyn_d::DynD;
pub use nd::ND;

use crate::Result;

use super::SliceRange;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Default)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

impl SizeAndStride {
	pub fn is_contiguous(&self) -> bool {
		self.stride == 1 || self.size <= 1
	}

	pub fn is_broadcasted(&self) -> bool {
		self.stride < 1 && self.size > 1
	}
}

//--------------------------------------------------------------------------------------------------

pub trait Map: Clone {
	fn ndim(&self) -> usize;
	fn elems(&self) -> usize;
}

pub trait MergeDims<const M: usize> {
	type Output: Map;

	fn merge_dims(self) -> Result<Self::Output>;
}

pub trait MergeAllDims {
	type Output: Map;

	fn merge_all_dims(self) -> Result<Self::Output>;
}

pub trait ReshapeLastDim<const M: usize> {
	type Output: Map;

	fn reshape_last_dim(self, to_shape: [usize; M]) -> Result<Self::Output>;
}

pub trait IndexToOffset<const K: usize> {
	fn index_to_offset(&self, index: [usize; K]) -> Result<usize>;
}

pub trait Slice<const K: usize> {
	type Output: Map;

	fn slice(self, ranges: [SliceRange; K]) -> Self::Output;
}

pub trait Transpose {
	type Output: Map;

	fn transposed(self, d0: usize, d1: usize) -> Result<Self::Output>;
}

//--------------------------------------------------------------------------------------------------
