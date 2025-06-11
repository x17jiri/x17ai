//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

pub mod dd;
pub mod nd;

use std::hint::likely;

pub use dd::DD;
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
	/// Returns true if the dimension can be treated as contiguous.
	///
	/// Note that dimension size 1 can be treated as both contiguous and broadcasted.
	pub fn is_contiguous(&self) -> bool {
		self.size <= 1 || self.stride == 1
	}

	/// Returns true if the dimension is broadcasted.
	///
	/// Note that dimension size 1 can be treated as both contiguous and broadcasted.
	pub fn is_broadcasted(&self) -> bool {
		self.size <= 1 || self.stride < 1
	}
}

//--------------------------------------------------------------------------------------------------

pub trait Map: Clone {
	fn ndim(&self) -> usize;
	fn size(&self, dim: usize) -> usize;
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
	unsafe fn select_unchecked(self, dim: usize, index: usize) -> Self::Output;
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

/// This function will initialize strides in a slice so a new tensor using them
/// is contiguous in memory.
///
/// It returns the total number of elements in the tensor.
pub fn init_strides(dims: &mut [SizeAndStride]) -> Result<usize> {
	let mut elems = 1;
	let mut nonzero_elems: usize = 1;
	for dim in dims.iter_mut().rev() {
		// Check that if we ignore zero length dimensions, the number of elements does not
		// overflow. This is done to make sure our calculations would not overflow even if we
		// had the same dimensions but in different order.
		if likely(dim.size != 0) {
			let Some(e) = nonzero_elems.checked_mul(dim.size) else {
				#[cold]
				fn err_init_strides_overflow() -> crate::Error {
					"Tensor dimensions overflowed while calculating strides.".into()
				}
				return Err(err_init_strides_overflow());
			};
			nonzero_elems = e;
		}

		dim.stride = elems;
		elems *= dim.size;
	}
	Ok(elems)
}

//--------------------------------------------------------------------------------------------------
