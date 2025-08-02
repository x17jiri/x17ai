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

use crate::tensor::dim_merger::DimMerger;
use crate::tensor::generic::dim_index::DimIndexOutOfBoundsError;
use crate::tensor::generic::universal_range::UniversalRange;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
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
	type Deref: Map;
	fn as_ref(&self) -> &Self::Deref;

	fn ndim(&self) -> usize;
	fn size(&self, dim: usize) -> usize;
	fn elems(&self) -> usize;
	fn span(&self) -> std::ops::Range<usize>;
	fn is_contiguous(&self) -> bool;
}

pub trait MergeDims<const M: usize> {
	type Output: Map;
	type Error;

	fn merge_dims(&self) -> Result<Self::Output, Self::Error>;
}

pub trait MergeAllDims {
	type Output: Map;
	type Error;

	fn merge_all_dims(&self) -> Result<Self::Output, Self::Error>;
}

pub trait SpanDims<const M: usize> {
	type Output: Map;
	type Error;

	fn span_dims(&self) -> Result<Self::Output, Self::Error>;
}

pub trait SpanAllDims {
	type Output: Map;
	type Error;

	fn span_all_dims(&self) -> Result<Self::Output, Self::Error>;
}

pub trait ReshapeLastDim<const M: usize> {
	type Output: Map;
	type Error;

	fn reshape_last_dim(&self, to_shape: [usize; M]) -> Result<Self::Output, Self::Error>;
}

pub trait IndexToOffset<const K: usize> {
	fn index_to_offset(&self, index: [usize; K]) -> Result<usize, IndexOutOfBoundsError>;
}

impl From<DimIndexOutOfBoundsError> for SelectError {
	fn from(_: DimIndexOutOfBoundsError) -> Self {
		Self::DimIndexOutOfBounds
	}
}

pub trait Select {
	type Output: Map;
	type Error: From<DimIndexOutOfBoundsError>;

	fn select(&self, dim: usize, index: usize) -> Result<Self::Output, Self::Error>;

	/// # Safety
	/// Both `dim` and `index` must be within bounds of the map.
	unsafe fn select_unchecked(&self, dim: usize, index: usize) -> Self::Output;
}

pub type NarrowError = SelectError;

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

pub trait NDShape<const K: usize> {
	type Error;

	fn nd_shape(&self) -> std::result::Result<[usize; K], Self::Error>;
}

//--------------------------------------------------------------------------------------------------

pub struct StrideCounter {
	pub elems: usize,
	pub nonzero_elems: usize,
}

impl Default for StrideCounter {
	fn default() -> Self {
		Self::new()
	}
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

impl Default for StrideCounterUnchecked {
	fn default() -> Self {
		Self::new()
	}
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

pub fn merge_dims(dims: &[SizeAndStride]) -> Result<SizeAndStride, IncompatibleStridesError> {
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
				return Err(IncompatibleStridesError);
			}
		}
	}
	Ok(merged)
}

pub fn reshape_dims(
	dims: &[SizeAndStride],
	into: &mut [SizeAndStride],
) -> Result<(), ReshapeError> {
	let Ok(dims) = DimMerger::merge::<5>([dims]) else {
		cold_path();
		return Err(ReshapeError);
	};
	let mut inp_iter =
		dims.iter().rev().map(|i| SizeAndStride { size: i.size, stride: i.strides[0] });
	let mut inp = inp_iter.next().unwrap_or(SizeAndStride { size: 1, stride: 0 });
	let mut out_acc = SizeAndStride { size: 1, stride: inp.stride };
	for out in into.iter_mut().rev() {
		let Some(mul) = out_acc.size.checked_mul(out.size) else {
			cold_path();
			return Err(ReshapeError);
		};
		out_acc.size = mul;
		if out_acc.size < inp.size {
			out.stride = out_acc.stride;
			out_acc.stride *= out.size;
		} else if out_acc.size == inp.size {
			out.stride = out_acc.stride;
			inp = inp_iter.next().unwrap_or(SizeAndStride { size: 1, stride: 0 });
			out_acc = SizeAndStride { size: 1, stride: inp.stride };
		} else {
			cold_path();
			return Err(ReshapeError);
		}
	}
	if inp.size == 1 {
		Ok(())
	} else {
		cold_path();
		Err(ReshapeError)
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NotEnoughDimensionsError;

impl From<NotEnoughDimensionsError> for MergeDimsError {
	fn from(_: NotEnoughDimensionsError) -> Self {
		Self::NotEnoughDimensions
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MergeDimsError {
	NotEnoughDimensions,
	IncompatibleStrides,
}

impl From<IncompatibleStridesError> for MergeDimsError {
	fn from(_: IncompatibleStridesError) -> Self {
		Self::IncompatibleStrides
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct IncompatibleStridesError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvalidNumElementsError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReshapeError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ReshapeLastDimError {
	NotEnoughDimensions,
	InvalidNumElements,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct IndexOutOfBoundsError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SelectError {
	DimIndexOutOfBounds,
	IndexOutOfBounds,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvalidNDimError;

/// The total nubmer of elements in a tensor is larger than the maximum allowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ElementsOverflowError;

//--------------------------------------------------------------------------------------------------

impl<'a, T: Map> Map for &'a T {
	type Deref = T::Deref;

	fn as_ref(&self) -> &Self::Deref {
		(*self).as_ref()
	}

	fn ndim(&self) -> usize {
		(*self).ndim()
	}

	fn size(&self, dim: usize) -> usize {
		(*self).size(dim)
	}

	fn elems(&self) -> usize {
		(*self).elems()
	}

	fn span(&self) -> std::ops::Range<usize> {
		(*self).span()
	}

	fn is_contiguous(&self) -> bool {
		(*self).is_contiguous()
	}
}

impl<T, const M: usize> MergeDims<M> for &T
where
	T: MergeDims<M>,
{
	type Output = T::Output;
	type Error = T::Error;

	fn merge_dims(&self) -> Result<Self::Output, Self::Error> {
		(*self).merge_dims()
	}
}

impl<T> MergeAllDims for &T
where
	T: MergeAllDims,
{
	type Output = T::Output;
	type Error = T::Error;

	fn merge_all_dims(&self) -> Result<Self::Output, Self::Error> {
		(*self).merge_all_dims()
	}
}

impl<T, const M: usize> SpanDims<M> for &T
where
	T: SpanDims<M>,
{
	type Output = T::Output;
	type Error = T::Error;

	fn span_dims(&self) -> Result<Self::Output, Self::Error> {
		(*self).span_dims()
	}
}

impl<T> SpanAllDims for &T
where
	T: SpanAllDims,
{
	type Output = T::Output;
	type Error = T::Error;

	fn span_all_dims(&self) -> Result<Self::Output, Self::Error> {
		(*self).span_all_dims()
	}
}

impl<T, const M: usize> ReshapeLastDim<M> for &T
where
	T: ReshapeLastDim<M>,
{
	type Output = T::Output;
	type Error = T::Error;

	fn reshape_last_dim(&self, to_shape: [usize; M]) -> Result<Self::Output, Self::Error> {
		(*self).reshape_last_dim(to_shape)
	}
}

impl<T> IndexToOffset<1> for &T
where
	T: IndexToOffset<1>,
{
	fn index_to_offset(&self, index: [usize; 1]) -> Result<usize, IndexOutOfBoundsError> {
		(*self).index_to_offset(index)
	}
}

impl<T> Select for &T
where
	T: Select,
{
	type Output = T::Output;
	type Error = T::Error;

	fn select(&self, dim: usize, index: usize) -> Result<Self::Output, Self::Error> {
		(*self).select(dim, index)
	}

	unsafe fn select_unchecked(&self, dim: usize, index: usize) -> Self::Output {
		unsafe { (*self).select_unchecked(dim, index) }
	}
}

impl<T> Narrow for &T
where
	T: Narrow,
{
	type Output = T::Output;
	type Error = T::Error;

	fn narrow(&self, dim: usize, range: UniversalRange) -> Result<Self::Output, Self::Error> {
		(*self).narrow(dim, range)
	}
}

/*impl<T> Transpose for &T
where
	T: Transpose,
{
	type Output = T::Output;
	type Error = T::Error;

	fn transposed(self, d0: usize, d1: usize) -> Result<Self::Output, Self::Error> {
		self.transposed(d0, d1)
	}
}*/

impl<T, const K: usize> NDShape<K> for &T
where
	T: NDShape<K>,
{
	type Error = T::Error;

	fn nd_shape(&self) -> std::result::Result<[usize; K], Self::Error> {
		(*self).nd_shape()
	}
}

//--------------------------------------------------------------------------------------------------
