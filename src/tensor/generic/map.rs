// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

pub mod dyn_d;
pub mod nd;

pub use dyn_d::DynD;
pub use nd::ND;

use crate::Error;

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
	fn offset(&self) -> usize;
	fn dims(&self) -> &[SizeAndStride];
	fn dims_mut(&mut self) -> &mut [SizeAndStride];
}

pub trait MergeDims<const M: usize> {
	type Output: Map;

	fn merge_dims(self) -> Option<Self::Output>;
}

pub trait MergeAllDims {
	type Output: Map;

	fn merge_all_dims(self) -> Option<Self::Output>;
}

pub trait ReshapeLastDim<const M: usize> {
	type Output: Map;

	fn reshape_last_dim(self, to_shape: [usize; M]) -> Option<Self::Output>;
}

pub trait IndexToOffset<const K: usize> {
	fn index_to_offset(&self, index: [usize; K]) -> Result<usize, Error>;
}

//--------------------------------------------------------------------------------------------------
