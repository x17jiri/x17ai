// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

//--------------------------------------------------------------------------------------------------

use smallvec::SmallVec;
use std::intrinsics::cold_path;
use std::ops::{Deref, DerefMut, Index, IndexMut};

use super::TensorSize;

#[derive(Clone, Copy, PartialEq, Default)]
pub struct SizeAndStride {
	pub size: TensorSize,
	pub stride: TensorSize,
}

pub enum DimType {
	Contiguous,
	Broadcasted,
	Strided,
}

impl SizeAndStride {
	pub fn is_contiguous(&self) -> bool {
		self.stride == 1 || self.size <= 1
	}

	pub fn is_broadcasted(&self) -> bool {
		self.stride < 1 && self.size > 1
	}

	pub fn is_strided(&self) -> bool {
		self.stride > 1 && self.size > 1
	}

	pub fn dim_type(&self) -> DimType {
		if self.stride == 1 || self.size <= 1 {
			DimType::Contiguous
		} else if self.stride < 1 {
			DimType::Broadcasted
		} else {
			DimType::Strided
		}
	}
}

//--------------------------------------------------------------------------------------------------
// I expect that 99.99% of the time, the DimVec will use inline storage.
// This implementaton is optimized so that we inline functions as long as they use the inline
// storage, but functions that would extend the storage are never inlined.
// This is to avoid code bloat.

pub const INLINE_DIMS: usize = 5;

pub struct DimVec {
	pub(crate) vec: SmallVec<[SizeAndStride; INLINE_DIMS]>,
}

impl DimVec {
	#[inline(never)]
	fn with_large_capacity(capacity: usize) -> DimVec {
		DimVec { vec: SmallVec::with_capacity(capacity) }
	}

	pub fn with_capacity(capacity: usize) -> DimVec {
		if capacity <= INLINE_DIMS {
			DimVec { vec: SmallVec::with_capacity(capacity) }
		} else {
			cold_path();
			DimVec::with_large_capacity(capacity)
		}
	}

	#[inline(never)]
	fn clone_large(&self) -> DimVec {
		DimVec { vec: self.vec.clone() }
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

	pub fn extend<'a, I: IntoIterator<Item = &'a SizeAndStride> + ExactSizeIterator>(
		&mut self, iterable: I,
	) {
		let mut ptr = self.vec.as_mut_ptr();
		let len = self.vec.len();
		let cap = self.vec.capacity();
		let add = iterable.len();
		if len + add > cap {
			ptr = self.reserve_large(add);
		}
		unsafe {
			for i in iterable {
				*ptr = *i;
				ptr = ptr.add(1);
			}
			self.vec.set_len(len + add);
		}
	}

	pub fn extend_with_array<const N: usize>(&mut self, array: &[SizeAndStride; N]) {
		let mut ptr = self.vec.as_mut_ptr();
		let len = self.vec.len();
		let cap = self.vec.capacity();
		let add = N;
		if len + add > cap {
			ptr = self.reserve_large(add);
		}
		unsafe {
			ptr = ptr.add(len);
			for i in array {
				*ptr = *i;
				ptr = ptr.add(1);
			}
			self.vec.set_len(len + add);
		}
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

	pub fn pop_n(&mut self, n: usize) {
		let len = self.vec.len();
		assert!(n <= len);
		unsafe { self.vec.set_len(len - n) };
	}

	pub fn as_slice(&self) -> &[SizeAndStride] {
		self.vec.as_slice()
	}
}

impl Clone for DimVec {
	fn clone(&self) -> DimVec {
		if self.vec.capacity() <= INLINE_DIMS {
			DimVec { vec: self.vec.clone() }
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
