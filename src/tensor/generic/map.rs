//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

pub mod dd;
pub mod nd;

use std::hint::{cold_path, likely};

pub use dd::DD;
pub use nd::ND;

use crate::ErrPack;
use crate::tensor::generic::dim_index::DimIndexOutOfBoundsError;
use crate::tensor::generic::universal_range::UniversalRange;
use crate::tensor::math::TensorOpError;

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

#[derive(Debug, Copy, Clone)]
pub enum MergeDimsError {
	NotEnoughDimensions,
	IncompatibleStrides,
}

impl From<MergeAllDimsError> for MergeDimsError {
	fn from(err: MergeAllDimsError) -> Self {
		match err {
			MergeAllDimsError::IncompatibleStrides => MergeDimsError::IncompatibleStrides,
		}
	}
}

pub trait MergeDims<const M: usize> {
	type Output: Map;
	type Error;

	fn merge_dims(&self) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, Copy, Clone)]
pub enum MergeAllDimsError {
	IncompatibleStrides,
}

pub trait MergeAllDims {
	type Output: Map;
	type Error;

	fn merge_all_dims(&self) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, Copy, Clone)]
pub struct InvalidNumElementsError;

#[derive(Debug, Copy, Clone)]
pub enum ReshapeLastDimError {
	NotEnoughDimensions,
	InvalidNumElements,
}

pub trait ReshapeLastDim<const M: usize> {
	type Output: Map;
	type Error;

	fn reshape_last_dim(&self, to_shape: [usize; M]) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, Copy, Clone)]
pub struct IndexOutOfBoundsError;

pub trait IndexToOffset<const K: usize> {
	fn index_to_offset(&self, index: [usize; K]) -> Result<usize, IndexOutOfBoundsError>;
}

#[derive(Debug, Copy, Clone)]
enum SelectError {
	DimIndexOutOfBounds,
	IndexOutOfBounds,
}

impl From<SelectError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: SelectError) -> Self {
		match err {
			SelectError::DimIndexOutOfBounds => TensorOpError::DimIndexOutOfBounds,
			SelectError::IndexOutOfBounds => TensorOpError::IndexOutOfBounds,
		}
	}
}

impl From<SelectError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: SelectError) -> Self {
		ErrPack { code: err.into(), extra: None }
	}
}

impl From<DimIndexOutOfBoundsError> for SelectError {
	fn from(_: DimIndexOutOfBoundsError) -> Self {
		SelectError::DimIndexOutOfBounds
	}
}

pub trait Select {
	type Output: Map;
	type Error: From<DimIndexOutOfBoundsError>;

	fn select(&self, dim: usize, index: usize) -> Result<Self::Output, Self::Error>;
	unsafe fn select_unchecked(&self, dim: usize, index: usize) -> Self::Output;
}

type NarrowError = SelectError;

pub trait Narrow {
	type Output: Map;
	type Error: From<DimIndexOutOfBoundsError>;

	fn narrow(&self, dim: usize, range: UniversalRange) -> Result<Self::Output, Self::Error>;
}

pub trait Transpose {
	type Output: Map;
	type Error: From<DimIndexOutOfBoundsError>;

	fn transposed(self, d0: usize, d1: usize) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, Copy, Clone)]
pub struct InvalidNDimError;

pub trait NDShape<const K: usize> {
	type Error;

	fn nd_shape(&self) -> std::result::Result<[usize; K], Self::Error>;
}

//--------------------------------------------------------------------------------------------------

/// The total nubmer of elements in a tensor is larger than the maximum allowed.
pub struct ElementsOverflowError;

pub struct StrideCounter {
	pub elems: usize,
	pub nonzero_elems: usize,
}

impl StrideCounter {
	pub fn new() -> Self {
		Self { elems: 1, nonzero_elems: 1 }
	}

	pub fn prepend_dim(&mut self, size: usize) -> Result<SizeAndStride, ElementsOverflowError> {
		// Check that if we ignore zero length dimensions, the number of elements does not
		// overflow. This is done to make sure our calculations would not overflow even if we
		// had the same dimensions but in different order.
		if likely(size != 0) {
			let Some(e) = self.nonzero_elems.checked_mul(size) else {
				cold_path();
				return Err(ElementsOverflowError);
			};
			self.nonzero_elems = e;
		}

		let stride = self.elems;
		self.elems *= size;

		Ok(SizeAndStride { size, stride })
	}

	pub fn elems(&self) -> usize {
		self.elems
	}
}

//--------------------------------------------------------------------------------------------------

/// This is like `StrideCounter`, but it does not check for overflow.
///
/// The typical use case is creating a new map from an existing map. Since we have an existing map,
/// we know the number of elements will not overflow.
pub struct StrideCounterUnchecked {
	pub elems: usize,
}

impl StrideCounterUnchecked {
	pub fn new() -> Self {
		Self { elems: 1 }
	}

	pub fn with_stride(stride: usize) -> Self {
		Self { elems: stride }
	}

	pub fn prepend_dim(&mut self, size: usize) -> SizeAndStride {
		let stride = self.elems;
		self.elems *= size;
		SizeAndStride { size, stride }
	}

	pub fn elems(&self) -> usize {
		self.elems
	}
}

//--------------------------------------------------------------------------------------------------

pub fn merge_dims(dims: &[SizeAndStride]) -> Result<SizeAndStride, MergeAllDimsError> {
	let mut merged = SizeAndStride { size: 1, stride: 1 };
	for dim in dims.iter().rev() {
		if dim.stride == merged.size * merged.stride || dim.size <= 1 {
			merged.size *= dim.size;
		} else {
			cold_path();
			if merged.size == 1 {
				merged = *dim;
			} else if merged.size > 1 {
				cold_path();
				return Err(MergeAllDimsError::IncompatibleStrides);
			}
		}
	}
	Ok(merged)
}

//--------------------------------------------------------------------------------------------------
