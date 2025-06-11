//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

pub mod dyn_d;
pub mod nd;

pub use dyn_d::DynD;
pub use nd::ND;

use crate::Result;
use crate::tensor::generic::universal_range::UniversalRange;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

impl SizeAndStride {
	pub fn is_contiguous(&self) -> bool {
		self.size <= 1 || self.stride == 1
	}

	pub fn is_broadcasted(&self) -> bool {
		self.size <= 1 || self.stride < 1
	}
}

//--------------------------------------------------------------------------------------------------

pub trait Map: Clone {
	fn ndim(&self) -> usize;
	fn elems(&self) -> usize;
	fn span(&self) -> std::ops::Range<usize>;
	fn is_contiguous(&self) -> bool;
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

pub trait Select {
	type Output: Map;

	fn select(self, dim: usize, index: usize) -> Result<Self::Output>;
}

pub trait Narrow {
	type Output: Map;

	fn narrow(self, dim: usize, range: UniversalRange) -> Result<Self::Output>;
}

pub trait Transpose {
	type Output: Map;

	fn transposed(self, d0: usize, d1: usize) -> Result<Self::Output>;
}

pub trait NDShape<const K: usize> {
	type Error: Into<crate::Error>;

	fn nd_shape(&self) -> std::result::Result<[usize; K], Self::Error>;
}

//--------------------------------------------------------------------------------------------------
