//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::{cold_path, likely};
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use super::DType;
use super::dim_index::DimIndexOutOfBoundsError;
use super::dim_merger::DimMerger;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

impl SizeAndStride {
	/// Returns true if the dimension can be treated as contiguous.
	pub fn is_contiguous(&self) -> bool {
		self.size <= 1 || self.stride == 1
	}

	/// Returns true if the dimension is broadcasted.
	pub fn is_broadcasted(&self) -> bool {
		self.size > 1 && self.stride == 0
	}
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

#[derive(Clone, Debug)]
pub struct Map {
	dims: DimVec,
	offset: usize,
	dtype: DType, // TODO - `dtype` should be stored inside `dims` to save space
}

impl Map {
	pub fn new(shape: &[usize], dtype: DType) -> Result<(Self, usize), ElementsOverflowError> {
		let mut dims = DimVecBuilder::new(shape.len());
		let slice = dims.as_slice_mut();

		let mut stride_counter = StrideCounter::new();
		for (dim, &size) in slice.iter_mut().zip(shape.iter()).rev() {
			dim.write(stride_counter.prepend_dim(size)?);
		}
		let elems = stride_counter.elems();

		let dims = unsafe { dims.assume_init() };
		Ok((Self { dims, offset: 0, dtype }, elems))
	}

	// NOTE: This is not a clone. We keep dimension sizes, but
	// initialize strides so the new map is contiguous.
	pub fn new_like(&self, dtype: DType) -> (Self, usize) {
		let src_slice = self.dims.as_slice();

		let mut dims = DimVecBuilder::new(src_slice.len());
		let slice = dims.as_slice_mut();

		let mut stride_counter = StrideCounterUnchecked::new();
		for (dim, src_dim) in slice.iter_mut().zip(src_slice.iter()).rev() {
			dim.write(stride_counter.prepend_dim(src_dim.size));
		}
		let elems = stride_counter.elems();

		let dims = unsafe { dims.assume_init() };
		(Self { dims, offset: 0, dtype }, elems)
	}

	pub fn new_replace_tail(
		&self,
		tail_len: usize,
		replace_with: &[usize],
		dtype: DType,
	) -> Result<(Self, usize), ReplaceTailError> {
		let src_slice = self.dims.as_slice();
		if tail_len > src_slice.len() {
			cold_path();
			return Err(ReplaceTailError::NotEnoughDimensions);
		}
		let n_keep = src_slice.len() - tail_len;
		let src_slice = unsafe { &src_slice.get_unchecked(..n_keep) };
		let ndim = n_keep + replace_with.len();

		let mut dims = DimVecBuilder::new(ndim);
		let slice = dims.as_slice_mut();

		let mut stride_counter = StrideCounter::new();
		let second_part = unsafe { slice.get_unchecked_mut(n_keep..) };
		for (dim, &size) in second_part.iter_mut().zip(replace_with.iter()).rev() {
			dim.write(stride_counter.prepend_dim(size)?);
		}
		let first_part = unsafe { slice.get_unchecked_mut(..n_keep) };
		for (dim, src_dim) in first_part.iter_mut().zip(src_slice.iter()).rev() {
			dim.write(stride_counter.prepend_dim(src_dim.size)?);
		}
		let elems = stride_counter.elems();

		let dims = unsafe { dims.assume_init() };
		Ok((Self { dims, offset: 0, dtype }, elems))
	}

	pub fn split_last_n<const N: usize>(
		&self,
	) -> Result<(&[SizeAndStride], [SizeAndStride; N], usize), NotEnoughDimensionsError> {
		let slice = self.dims.as_slice();
		if slice.len() < N {
			cold_path();
			return Err(NotEnoughDimensionsError);
		}
		let split_at = slice.len() - N;
		let (prefix, nd) = slice.split_at(split_at);
		let mut dims = [SizeAndStride::default(); N];
		for i in 0..N {
			dims[i] = nd[i];
		}
		Ok((prefix, dims, self.offset))
	}

	pub fn offset(&self) -> usize {
		self.offset
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn ndim(&self) -> usize {
		self.dims.len()
	}

	pub fn dims(&self) -> &[SizeAndStride] {
		self.dims.as_slice()
	}

	pub fn dim(&self, dim: usize) -> SizeAndStride {
		let dims = self.dims.as_slice();
		dims[dim]
	}

	pub fn elems(&self) -> usize {
		let dims = self.dims.as_slice();
		dims.iter().map(|dim| dim.size).product()
	}

	pub fn span(&self) -> std::ops::Range<usize> {
		let start = self.offset;
		let mut elems = 1;
		let mut len = 1;
		for dim in self.dims.as_slice() {
			elems *= dim.size;
			len += dim.size.wrapping_sub(1).wrapping_mul(dim.stride);
		}
		if elems == 0 {
			len = 0;
		}
		start..start + len
	}

	/// Merges the last `n` dimensions into a single dimension.
	///
	/// If ndim < n, the missing dimensions are treated as size 1 dimensions.
	pub fn merge_dims(&self, n: usize) -> Result<Self, IncompatibleStridesError> {
		let old_slice = self.dims.as_slice();
		let n_keep = old_slice.len().saturating_sub(n);
		let ndim = n_keep + 1;

		let merged = merge_dims(&old_slice[n_keep..])?;

		let mut dims = DimVecBuilder::new(ndim);
		let slice = dims.as_slice_mut();
		for i in 0..n_keep {
			slice[i].write(old_slice[i]);
		}
		slice[n_keep].write(merged);
		let dims = unsafe { dims.assume_init() };

		Ok(Self { dims, offset: 0, dtype: self.dtype })
	}

	/// Merges all dimensions into a single dimension.
	pub fn merge_all_dims(&self) -> Result<Self, IncompatibleStridesError> {
		let merged = merge_dims(self.dims.as_slice())?;

		let mut dims = DimVecBuilder::new(1);
		let slice = dims.as_slice_mut();
		slice[0].write(merged);

		let dims = unsafe { dims.assume_init() };
		Ok(Self { dims, offset: 0, dtype: self.dtype })
	}

	/// Reshapes single last dimension into `to_shape.len()` dimensions.
	///
	/// If there are no dimensions, it works as if there was a single dimension of size 1.
	pub fn reshape_last_dim(&self, to_shape: &[usize]) -> Result<Self, ReshapeError> {
		let old_slice = self.dims.as_slice();
		let (n_keep, last_dim) = if let Some(&last) = old_slice.last() {
			(old_slice.len() - 1, last)
		} else {
			(0, SizeAndStride { size: 1, stride: 0 })
		};

		let elems = to_shape.iter().copied().product::<usize>();
		if elems != last_dim.size {
			cold_path();
			return Err(ReshapeError);
		}

		let ndim = n_keep + to_shape.len();

		let mut dims = DimVecBuilder::new(ndim);
		let slice = dims.as_slice_mut();

		for i in 0..n_keep {
			slice[i].write(old_slice[i]);
		}
		let mut stride_counter = StrideCounterUnchecked::with_stride(last_dim.stride);
		for i in (n_keep..ndim).rev() {
			slice[i].write(stride_counter.prepend_dim(to_shape[i - n_keep]));
		}

		let dims = unsafe { dims.assume_init() };
		Ok(Self { dims, offset: 0, dtype: self.dtype })
	}

	pub fn select(&self, dim: usize, index: usize) -> Result<Self, SelectError> {
		let old_slice = self.dims.as_slice();

		let Some(removed_dim) = old_slice.get(dim) else {
			cold_path();
			return Err(SelectError::DimIndexOutOfBounds);
		};

		if index >= removed_dim.size {
			cold_path();
			return Err(SelectError::IndexOutOfBounds);
		}

		let ndim = old_slice.len() - 1;
		let mut dims = DimVecBuilder::new(ndim);
		let slice = dims.as_slice_mut();

		for i in 0..dim {
			slice[i].write(old_slice[i]);
		}
		for i in dim..ndim {
			slice[i].write(old_slice[i + 1]);
		}

		let dims = unsafe { dims.assume_init() };
		Ok(Self {
			dims,
			offset: self.offset + index * removed_dim.stride,
			dtype: self.dtype,
		})
	}
}

//--------------------------------------------------------------------------------------------------

pub const INLINE_DIMS: usize = 4;

#[derive(Copy, Clone)]
union DimVecItems {
	inline: [MaybeUninit<SizeAndStride>; INLINE_DIMS],
	heap: NonNull<MaybeUninit<SizeAndStride>>,
}

pub struct DimVecBuilder {
	len: usize,
	items: DimVecItems,
}

impl DimVecBuilder {
	#[inline(never)]
	fn new_heap_items(len: usize) -> NonNull<MaybeUninit<SizeAndStride>> {
		let layout = std::alloc::Layout::array::<MaybeUninit<SizeAndStride>>(len)
			.expect("Failed to create layout for DimVecBuilderItems");
		let mem = NonNull::new(unsafe { std::alloc::alloc(layout) })
			.expect("Failed to allocate memory for DimVecBuilderItems");
		mem.cast()
	}

	pub fn new(len: usize) -> Self {
		if len <= INLINE_DIMS {
			Self {
				len,
				items: DimVecItems {
					inline: [MaybeUninit::uninit(); INLINE_DIMS],
				},
			}
		} else {
			cold_path();
			Self {
				len,
				items: DimVecItems { heap: Self::new_heap_items(len) },
			}
		}
	}

	pub fn as_slice_mut(&mut self) -> &mut [MaybeUninit<SizeAndStride>] {
		unsafe {
			if self.len <= INLINE_DIMS {
				self.items.inline.get_unchecked_mut(..self.len)
			} else {
				cold_path();
				std::slice::from_raw_parts_mut(self.items.heap.as_ptr(), self.len)
			}
		}
	}

	pub unsafe fn assume_init(self) -> DimVec {
		let Self { len, items } = self;
		DimVec { len, items }
	}
}

pub struct DimVec {
	len: usize,
	items: DimVecItems,
}

impl DimVec {
	pub fn len(&self) -> usize {
		self.len
	}

	pub fn is_empty(&self) -> bool {
		self.len == 0
	}

	pub fn as_slice(&self) -> &[SizeAndStride] {
		unsafe {
			if self.len <= INLINE_DIMS {
				let slice = self.items.inline.get_unchecked(..self.len);
				slice.assume_init_ref()
			} else {
				cold_path();
				let slice = std::slice::from_raw_parts(self.items.heap.as_ptr(), self.len);
				slice.assume_init_ref()
			}
		}
	}

	pub fn as_slice_mut(&mut self) -> &mut [SizeAndStride] {
		unsafe {
			if self.len <= INLINE_DIMS {
				let slice = self.items.inline.get_unchecked_mut(..self.len);
				slice.assume_init_mut()
			} else {
				cold_path();
				let slice = std::slice::from_raw_parts_mut(self.items.heap.as_ptr(), self.len);
				slice.assume_init_mut()
			}
		}
	}

	#[inline(never)]
	fn finish_clone_large(&mut self) {
		unsafe {
			let new_items = DimVecBuilder::new_heap_items(self.len);
			let src = std::slice::from_raw_parts(self.items.heap.as_ptr(), self.len);
			let dst = std::slice::from_raw_parts_mut(new_items.as_ptr(), self.len);
			dst.copy_from_slice(src);
			self.items.heap = new_items;
		}
	}
}

impl Clone for DimVec {
	fn clone(&self) -> Self {
		let mut result = Self { len: self.len, items: self.items };
		if self.len > INLINE_DIMS {
			cold_path();
			result.finish_clone_large();
		}
		result
	}
}

impl std::fmt::Debug for DimVec {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let slice = self.as_slice();
		f.debug_list().entries(slice.iter()).finish()
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NotEnoughDimensionsError;

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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ReplaceTailError {
	ElementsOverflow,
	NotEnoughDimensions,
}

impl From<ElementsOverflowError> for ReplaceTailError {
	fn from(_: ElementsOverflowError) -> Self {
		Self::ElementsOverflow
	}
}

//--------------------------------------------------------------------------------------------------
