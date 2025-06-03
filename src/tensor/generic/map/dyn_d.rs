// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use smallvec::SmallVec;
use std::intrinsics::cold_path;
use std::ops::{Deref, DerefMut, Index, IndexMut};

use super::{IMap, SizeAndStride};

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct DynD {
	dims: DimVec,
	offset: usize,
}

impl IMap for DynD {
	fn offset(&self) -> usize {
		self.offset
	}

	fn dims(&self) -> &[SizeAndStride] {
		self.dims.as_slice()
	}

	fn dims_mut(&mut self) -> &mut [SizeAndStride] {
		self.dims.as_mut_slice()
	}
}

//--------------------------------------------------------------------------------------------------
// I expect that 99.99% of the time, the DimVec will use inline storage.
// This implementaton is optimized so that we inline functions as long as they use the inline
// storage, but functions that would extend the storage are never inlined.
// This is to avoid code bloat.

pub const INLINE_DIMS: usize = 5;

pub struct DimVec {
	vec: SmallVec<[SizeAndStride; INLINE_DIMS]>,
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

	pub fn new() -> DimVec {
		DimVec { vec: SmallVec::new() }
	}

	pub fn new_from_iter<I: IntoIterator<Item = SizeAndStride> + ExactSizeIterator>(
		iter: I,
	) -> DimVec {
		let len = iter.len();
		let mut t = DimVec::with_capacity(len);
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

	pub fn as_mut_slice(&mut self) -> &mut [SizeAndStride] {
		self.vec.as_mut_slice()
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
