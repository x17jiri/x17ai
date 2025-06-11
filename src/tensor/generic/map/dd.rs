//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use smallvec::SmallVec;
use std::hint::likely;
use std::intrinsics::cold_path;
use std::ops::{Deref, DerefMut, Index, IndexMut};

use crate::Result;
use crate::tensor::generic::map::{MergeAllDims, MergeDims, ReshapeLastDim, init_strides};

use super::{Map, SizeAndStride, Transpose};

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct DD {
	pub dims: DimVec,
	pub offset: usize,
}

impl DD {
	pub fn new(shape: &[usize]) -> Result<(Self, usize)> {
		let mut dims =
			DimVec::new_from_iter(shape.iter().map(|&size| SizeAndStride { size, stride: 0 }));
		let elems = init_strides(&mut dims)?;
		let map = Self { dims, offset: 0 };
		Ok((map, elems))
	}

	pub fn new_like(&self) -> (Self, usize) {
		let mut dims = self.dims.clone();
		let elems = init_strides(&mut dims).unwrap();
		let map = Self { dims, offset: 0 };
		(map, elems)
	}

	pub fn new_replace_tail(
		&self, tail_len: usize, replace_with: &[usize],
	) -> Result<(Self, usize)> {
		let n_keep = self.dims.len().checked_sub(tail_len).expect("not enough dimensions");
		let ndim = n_keep + replace_with.len();
		let mut dims = DimVec::with_capacity(ndim);
		unsafe {
			dims.extend_unchecked(self.dims.get_unchecked(..n_keep).iter().copied());
			dims.extend_unchecked(
				replace_with.iter().map(|&size| SizeAndStride { size, stride: 0 }),
			);
		}
		let elems = init_strides(&mut dims)?;
		let map = Self { dims, offset: 0 };
		Ok((map, elems))
	}
}

impl Map for DD {
	fn ndim(&self) -> usize {
		self.dims.len()
	}

	fn size(&self, dim: usize) -> usize {
		if dim >= self.dims.len() {
			cold_path();
			return 1;
		}
		self.dims[dim].size
	}

	fn elems(&self) -> usize {
		self.dims.iter().map(|dim| dim.size).product()
	}

	fn span(&self) -> std::ops::Range<usize> {
		todo!();
	}

	fn is_contiguous(&self) -> bool {
		todo!();
	}
}

impl<const M: usize> MergeDims<M> for DD {
	type Output = Self;

	fn merge_dims(mut self) -> Result<Self::Output> {
		let dims = &mut self.dims;
		let ndim = dims.len();
		if M > ndim {
			cold_path();
			return Err(
				format!("Cannot merge {M} dimensions in a tensor with {ndim} dimensions.",).into(),
			);
		}

		if M == 0 {
			dims.push(SizeAndStride { size: 1, stride: 0 });
		} else {
			let mut merged = *unsafe { dims.get_unchecked(ndim - 1) };
			for i in (ndim - M..ndim - 1).rev() {
				let dim = *unsafe { dims.get_unchecked(i) };
				if dim.size > 1 && dim.stride != merged.size * merged.stride {
					cold_path();
					return Err("Cannot merge because of discontinuity".into());
				}
				merged.size *= dim.size;
			}
			dims.pop_n(M - 1);
			*unsafe { dims.get_unchecked_mut(ndim - M) } = merged;
		}

		Ok(self)
	}
}

impl MergeAllDims for DD {
	type Output = Self;

	fn merge_all_dims(mut self) -> Result<Self::Output> {
		let dims = &mut self.dims;
		let ndim = dims.len();

		match ndim {
			0 => {
				dims.push(SizeAndStride { size: 1, stride: 0 });
			},
			1 => {},
			_ => {
				let mut merged = *unsafe { dims.get_unchecked(ndim - 1) };
				for i in (0..ndim - 1).rev() {
					let dim = *unsafe { dims.get_unchecked(i) };
					if dim.size > 1 && dim.stride != merged.size * merged.stride {
						cold_path();
						return Err("cannot merge because of discontinuity".into());
					}
					merged.size *= dim.size;
				}
				dims.pop_n(ndim - 1);
				*unsafe { dims.get_unchecked_mut(0) } = merged;
			},
		}

		Ok(self)
	}
}

impl<const M: usize> ReshapeLastDim<M> for DD {
	type Output = Self;

	fn reshape_last_dim(mut self, to_shape: [usize; M]) -> Result<Self::Output> {
		let dims = &mut self.dims;

		let Some(removed_dim) = dims.pop() else {
			cold_path();
			return Err("Not enough dimensions".into());
		};

		let elems = to_shape.iter().copied().product::<usize>();
		if elems != removed_dim.size {
			cold_path();
			return Err("incompatible reshape".into());
		}

		let mut stride = removed_dim.stride;
		dims.extend_rev(to_shape.iter().rev().map(|&size| {
			let dim = SizeAndStride { size, stride };
			stride *= size;
			dim
		}));

		Ok(self)
	}
}

impl Transpose for DD {
	type Output = Self;

	fn transposed(mut self, d0: usize, d1: usize) -> Result<Self> {
		if d0 >= self.dims.len() || d1 >= self.dims.len() {
			return Err(format!(
				"Cannot transpose dimension {} with {}. Tensor has {} dimensions.",
				d0,
				d1,
				self.dims.len()
			)
			.into());
		}
		self.dims.swap(d0, d1);
		Ok(self)
	}
}

//--------------------------------------------------------------------------------------------------
// I expect that 99.99% of the time, the DimVec will use inline storage.
// This implementaton is optimized so that we inline functions as long as they use the inline
// storage, but functions that would extend the storage are never inlined.
// This is to avoid code bloat.

pub const INLINE_DIMS: usize = 4;

pub struct DimVec {
	vec: SmallVec<[SizeAndStride; INLINE_DIMS]>,
}

impl DimVec {
	#[inline(never)]
	fn with_large_capacity(capacity: usize) -> Self {
		Self { vec: SmallVec::with_capacity(capacity) }
	}

	pub fn with_capacity(capacity: usize) -> Self {
		if capacity <= INLINE_DIMS {
			Self { vec: SmallVec::with_capacity(capacity) }
		} else {
			cold_path();
			Self::with_large_capacity(capacity)
		}
	}

	pub fn new() -> Self {
		Self { vec: SmallVec::new() }
	}

	pub fn new_from_iter<I: IntoIterator<Item = SizeAndStride> + ExactSizeIterator>(
		iter: I,
	) -> Self {
		let len = iter.len();
		let mut t = Self::with_capacity(len);
		unsafe {
			let mut ptr = t.vec.as_mut_ptr();
			for i in iter {
				*ptr = i;
				ptr = ptr.add(1);
			}
			t.vec.set_len(len);
		}
		t
	}

	pub unsafe fn extend_unchecked<I: IntoIterator<Item = SizeAndStride> + ExactSizeIterator>(
		&mut self, iter: I,
	) {
		let len = self.vec.len();
		let add = iter.len();
		debug_assert!(len + add <= self.vec.capacity());
		unsafe {
			let mut ptr = self.vec.as_mut_ptr().add(len);
			for i in iter {
				*ptr = i;
				ptr = ptr.add(1);
			}
			self.vec.set_len(len + add);
		}
	}

	pub fn extend_rev<I: IntoIterator<Item = SizeAndStride> + ExactSizeIterator>(
		&mut self, iter: I,
	) {
		let len = self.vec.len();
		let add = iter.len();
		let cap = self.vec.capacity();

		let ptr = if len + add <= cap {
			self.vec.as_mut_ptr()
		} else {
			cold_path();
			self.reserve_large(add)
		};

		unsafe {
			let mut ptr = ptr.add(len + add);
			for i in iter {
				ptr = ptr.sub(1);
				*ptr = i;
			}
			self.vec.set_len(len + add);
		}
	}

	#[inline(never)]
	fn clone_large(&self) -> Self {
		Self { vec: self.vec.clone() }
	}

	#[inline]
	pub fn len(&self) -> usize {
		self.vec.len()
	}

	#[inline(never)]
	fn reserve_large(&mut self, additional: usize) -> *mut SizeAndStride {
		self.vec.reserve(additional);
		self.vec.as_mut_ptr()
	}

	pub fn swap(&mut self, a: usize, b: usize) {
		self.vec.swap(a, b);
	}

	#[inline(never)]
	fn push_large(&mut self, dim: SizeAndStride) {
		self.vec.push(dim);
	}

	pub fn push(&mut self, dim: SizeAndStride) {
		if self.vec.len() < self.vec.capacity() {
			self.vec.push(dim);
		} else {
			cold_path();
			self.push_large(dim);
		}
	}

	pub fn pop(&mut self) -> Option<SizeAndStride> {
		self.vec.pop()
	}

	/// Removes `n` elements from the end of the vector.
	///
	/// # Panics
	///
	/// Panics if `n` is greater than the current length of the vector.
	pub fn pop_n(&mut self, n: usize) {
		let len = self.vec.len();
		assert!(n <= len);
		unsafe { self.vec.set_len(len - n) };
	}

	pub fn as_slice(&self) -> &[SizeAndStride] {
		self.vec.as_slice()
	}

	pub fn as_mut_slice(&mut self) -> &mut [SizeAndStride] {
		self.vec.as_mut_slice()
	}
}

impl Clone for DimVec {
	fn clone(&self) -> Self {
		if self.vec.capacity() <= INLINE_DIMS {
			Self { vec: self.vec.clone() }
		} else {
			cold_path();
			self.clone_large()
		}
	}
}

impl Index<usize> for DimVec {
	type Output = SizeAndStride;

	fn index(&self, index: usize) -> &Self::Output {
		&self.vec[index]
	}
}

impl IndexMut<usize> for DimVec {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.vec[index]
	}
}

impl Index<std::ops::Range<usize>> for DimVec {
	type Output = [SizeAndStride];

	fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
		&self.vec[index]
	}
}

impl Index<std::ops::RangeTo<usize>> for DimVec {
	type Output = [SizeAndStride];

	fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
		&self.vec[index]
	}
}

impl Index<std::ops::RangeFrom<usize>> for DimVec {
	type Output = [SizeAndStride];

	fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
		&self.vec[index]
	}
}

impl Deref for DimVec {
	type Target = [SizeAndStride];

	fn deref(&self) -> &Self::Target {
		&self.vec
	}
}

impl DerefMut for DimVec {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.vec
	}
}

//--------------------------------------------------------------------------------------------------
