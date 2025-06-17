//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::intrinsics::cold_path;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::tensor::generic::dim_index::DimIndexOutOfBoundsError;
use crate::tensor::generic::map::{
	ElementsOverflowError, IncompatibleStridesError, MergeAllDims, MergeDims, MergeDimsError,
	ReshapeLastDim, ReshapeLastDimError, Select, SelectError, StrideCounter,
	StrideCounterUnchecked, merge_dims,
};

use super::{Map, SizeAndStride, Transpose};

//--------------------------------------------------------------------------------------------------

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

#[derive(Clone, Eq, PartialEq)]
pub struct DD {
	pub dims: DimVec,
	pub offset: usize,
}

impl DD {
	#[inline(never)] // TODO
	pub fn new(shape: &[usize]) -> Result<(Self, usize), ElementsOverflowError> {
		let mut dims = DimVecBuilder::new(shape.len());
		let slice = dims.as_slice_mut();

		let mut stride_counter = StrideCounter::new();
		for (dim, &size) in slice.iter_mut().zip(shape.iter()).rev() {
			dim.write(stride_counter.prepend_dim(size)?);
		}
		let elems = stride_counter.elems();

		let dims = unsafe { dims.assume_init() };
		Ok((Self { dims, offset: 0 }, elems))
	}

	#[inline(never)] // TODO
	pub fn new_like(&self) -> (Self, usize) {
		let src_slice = self.dims.as_slice();

		let mut dims = DimVecBuilder::new(src_slice.len());
		let slice = dims.as_slice_mut();

		let mut stride_counter = StrideCounterUnchecked::new();
		for (dim, src_dim) in slice.iter_mut().zip(src_slice.iter()).rev() {
			dim.write(stride_counter.prepend_dim(src_dim.size));
		}
		let elems = stride_counter.elems();

		let dims = unsafe { dims.assume_init() };
		(Self { dims, offset: 0 }, elems)
	}

	#[inline(never)] // TODO
	pub fn new_replace_tail(
		&self,
		tail_len: usize,
		replace_with: &[usize],
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
		Ok((Self { dims, offset: 0 }, elems))
	}
}

impl Map for DD {
	fn ndim(&self) -> usize {
		self.dims.len()
	}

	fn size(&self, dim: usize) -> usize {
		let dims = self.dims.as_slice();
		dims[dim].size
	}

	fn elems(&self) -> usize {
		let dims = self.dims.as_slice();
		dims.iter().map(|dim| dim.size).product()
	}

	fn span(&self) -> std::ops::Range<usize> {
		todo!("<DD as Map>::span is not implemented yet");
	}

	fn is_contiguous(&self) -> bool {
		todo!("<DD as Map>::is_contiguous is not implemented yet");
	}
}

impl<const M: usize> MergeDims<M> for DD {
	type Output = Self;
	type Error = MergeDimsError;

	#[inline(never)] // TODO
	fn merge_dims(&self) -> Result<Self::Output, MergeDimsError> {
		let old_slice = self.dims.as_slice();
		let old_ndim = old_slice.len();
		if old_ndim < M {
			cold_path();
			return Err(MergeDimsError::NotEnoughDimensions);
		}
		let n_keep = old_ndim - M;
		let ndim = n_keep + 1;

		let merged = merge_dims(&old_slice[n_keep..old_ndim])?;

		let mut dims = DimVecBuilder::new(ndim);
		let slice = dims.as_slice_mut();
		for i in 0..n_keep {
			slice[i].write(old_slice[i]);
		}
		slice[n_keep].write(merged);

		let dims = unsafe { dims.assume_init() };
		Ok(Self { dims, offset: 0 })
	}
}

impl MergeAllDims for DD {
	type Output = Self;
	type Error = IncompatibleStridesError;

	#[inline(never)] // TODO
	fn merge_all_dims(&self) -> Result<Self::Output, IncompatibleStridesError> {
		let merged = merge_dims(self.dims.as_slice())?;

		let mut dims = DimVecBuilder::new(1);
		let slice = dims.as_slice_mut();
		slice[0].write(merged);

		let dims = unsafe { dims.assume_init() };
		Ok(Self { dims, offset: 0 })
	}
}

impl<const M: usize> ReshapeLastDim<M> for DD {
	type Output = Self;
	type Error = ReshapeLastDimError;

	#[inline(never)] // TODO
	fn reshape_last_dim(&self, to_shape: [usize; M]) -> Result<Self::Output, ReshapeLastDimError> {
		let old_slice = self.dims.as_slice();
		let old_ndim = old_slice.len();

		let Some(last_dim) = old_slice.last() else {
			cold_path();
			return Err(ReshapeLastDimError::NotEnoughDimensions);
		};

		let elems = to_shape.iter().copied().product::<usize>();
		if elems != last_dim.size {
			cold_path();
			return Err(ReshapeLastDimError::InvalidNumElements);
		}

		let n_keep = old_ndim - 1;
		let ndim = n_keep + M;

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
		Ok(Self { dims, offset: 0 })
	}
}

impl Select for DD {
	type Output = Self;
	type Error = SelectError;

	#[inline(never)] // TODO
	fn select(&self, dim: usize, index: usize) -> Result<Self, SelectError> {
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
		})
	}

	#[inline(never)] // TODO
	unsafe fn select_unchecked(&self, dim: usize, index: usize) -> Self::Output {
		let old_slice = self.dims.as_slice();
		debug_assert!(dim < old_slice.len());

		let removed_dim = old_slice.get_unchecked(dim);
		debug_assert!(index < removed_dim.size);

		let ndim = old_slice.len() - 1;
		let mut dims = DimVecBuilder::new(ndim);
		let slice = dims.as_slice_mut();

		for i in 0..dim {
			slice.get_unchecked_mut(i).write(*old_slice.get_unchecked(i));
		}
		for i in dim..ndim {
			slice.get_unchecked_mut(i).write(*old_slice.get_unchecked(i + 1));
		}

		let dims = unsafe { dims.assume_init() };
		Self {
			dims,
			offset: self.offset + index * removed_dim.stride,
		}
	}
}

impl Transpose for DD {
	type Output = Self;
	type Error = DimIndexOutOfBoundsError;

	#[inline(never)] // TODO
	fn transposed(mut self, d0: usize, d1: usize) -> Result<Self, DimIndexOutOfBoundsError> {
		let slice = self.dims.as_slice_mut();
		if d0 >= slice.len() || d1 >= slice.len() {
			cold_path();
			return Err(DimIndexOutOfBoundsError);
		}
		slice.swap(d0, d1);
		Ok(self)
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

impl PartialEq for DimVec {
	fn eq(&self, other: &Self) -> bool {
		self.as_slice() == other.as_slice()
	}
}

impl Eq for DimVec {}

//--------------------------------------------------------------------------------------------------
