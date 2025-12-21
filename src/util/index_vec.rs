//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hash::Hash;
use std::hint::cold_path;

//--------------------------------------------------------------------------------------------------

pub trait IndexTrait: Copy + Clone + PartialEq + Eq + Hash + Ord {
	const MAX: usize;
	fn to_raw(self) -> usize;
	fn from_raw(raw: usize) -> Self;
}

//--------------------------------------------------------------------------------------------------

#[macro_export]
macro_rules! define_index_type {
	($name:ident) => {
		#[repr(transparent)]
		#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
		pub struct $name {
			pub raw: usize,
		}

		impl $name {
			pub fn new(raw: usize) -> Self {
				debug_assert!(isize::try_from(raw).is_ok());
				$name { raw }
			}

			pub fn new_sentinel() -> Self {
				$name { raw: usize::MAX }
			}

			#[allow(clippy::cast_possible_wrap)]
			pub fn is_sentinel(&self) -> bool {
				self.raw >= usize::MAX
			}

			pub fn to_untyped(&self) -> $crate::util::index_vec::UntypedIndex {
				$crate::util::index_vec::UntypedIndex { raw: self.raw }
			}
		}

		impl $crate::util::index_vec::IndexTrait for $name {
			const MAX: usize = isize::MAX as usize;

			fn to_raw(self) -> usize {
				self.raw
			}

			fn from_raw(raw: usize) -> Self {
				debug_assert!(raw <= Self::MAX);
				$name { raw }
			}
		}

		impl From<$crate::util::index_vec::UntypedIndex> for $name {
			fn from(value: $crate::util::index_vec::UntypedIndex) -> Self {
				$name { raw: value.raw }
			}
		}
	};
}

//--------------------------------------------------------------------------------------------------

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UntypedIndex {
	pub raw: usize,
}

impl UntypedIndex {
	const MAX: usize = isize::MAX as usize;

	pub fn new(raw: usize) -> Self {
		debug_assert!(raw <= Self::MAX);
		Self { raw }
	}

	pub fn new_sentinel() -> Self {
		Self { raw: usize::MAX }
	}

	#[allow(clippy::cast_possible_wrap)]
	pub fn is_sentinel(&self) -> bool {
		self.raw >= Self::MAX
	}
}

//--------------------------------------------------------------------------------------------------

#[macro_export]
macro_rules! define_index_type32 {
	($name:ident) => {
		#[repr(transparent)]
		#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
		pub struct $name {
			pub raw: u32,
		}

		impl $name {
			pub fn new(raw: u32) -> Self {
				debug_assert!(i32::try_from(raw).is_ok());
				$name { raw }
			}

			pub fn new_sentinel() -> Self {
				$name { raw: u32::MAX }
			}

			pub fn is_sentinel(&self) -> bool {
				self.raw >= u32::MAX
			}

			pub fn to_untyped(&self) -> $crate::util::index_vec::UntypedIndex32 {
				$crate::util::index_vec::UntypedIndex32 { raw: self.raw }
			}
		}

		impl $crate::util::index_vec::IndexTrait for $name {
			const MAX: usize = i32::MAX as usize;

			fn to_raw(self) -> usize {
				self.raw as usize
			}

			fn from_raw(raw: usize) -> Self {
				debug_assert!(raw <= Self::MAX);
				#[allow(clippy::cast_possible_truncation)]
				$name { raw: raw as u32 }
			}
		}

		impl From<$crate::util::index_vec::UntypedIndex32> for $name {
			fn from(value: $crate::util::index_vec::UntypedIndex32) -> Self {
				$name { raw: value.raw }
			}
		}
	};
}

//--------------------------------------------------------------------------------------------------

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UntypedIndex32 {
	pub raw: u32,
}

impl UntypedIndex32 {
	pub const MAX: u32 = i32::MAX as u32;

	pub fn new_sentinel() -> Self {
		Self { raw: u32::MAX }
	}

	#[allow(clippy::cast_possible_wrap)]
	pub fn is_sentinel(&self) -> bool {
		self.raw >= Self::MAX
	}
}

//--------------------------------------------------------------------------------------------------

pub struct IndexVec<Index: IndexTrait, T> {
	pub raw: Vec<T>,
	_marker: std::marker::PhantomData<Index>,
}

impl<Index: IndexTrait, T> Default for IndexVec<Index, T> {
	fn default() -> Self {
		Self::new()
	}
}

impl<Index: IndexTrait, T> IndexVec<Index, T> {
	pub fn new() -> Self {
		Self {
			raw: Vec::new(),
			_marker: std::marker::PhantomData,
		}
	}

	pub fn with_capacity(capacity: usize) -> Self {
		Self {
			raw: Vec::with_capacity(capacity),
			_marker: std::marker::PhantomData,
		}
	}

	pub fn from_vec(raw: Vec<T>) -> Self {
		Self { raw, _marker: std::marker::PhantomData }
	}

	pub fn push(&mut self, item: T) -> Index {
		let index = Index::from_raw(self.raw.len());
		self.raw.push(item);
		index
	}

	pub fn push_within_capacity(&mut self, item: T) -> Result<Index, T> {
		let index = Index::from_raw(self.raw.len());
		match self.raw.push_within_capacity(item) {
			Ok(_) => Ok(index),
			Err(item) => {
				cold_path();
				Err(item)
			},
		}
	}

	pub fn next_index(&self) -> Index {
		Index::from_raw(self.raw.len())
	}

	pub fn indexes(&self) -> impl DoubleEndedIterator<Item = Index> + use<Index, T> {
		let len = self.raw.len();
		(0..len).map(Index::from_raw)
	}

	pub fn len(&self) -> usize {
		self.raw.len()
	}

	pub fn is_empty(&self) -> bool {
		self.raw.is_empty()
	}

	pub fn iter(&self) -> std::slice::Iter<'_, T> {
		self.raw.iter()
	}

	pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
		self.raw.iter_mut()
	}

	pub fn borrow_multiple<'a>(
		&'a mut self,
		index: Index,
	) -> (IndexSliceMut<'a, Index, T>, &'a mut T, IndexSliceMut<'a, Index, T>) {
		let (prefix, t) = self.raw.split_at_mut(index.to_raw());
		let (t, suffix) = t.split_at_mut(1);
		(
			IndexSliceMut {
				raw: prefix,
				offset: 0,
				_marker: std::marker::PhantomData,
			},
			&mut t[0],
			IndexSliceMut {
				raw: suffix,
				offset: index.to_raw() + 1,
				_marker: std::marker::PhantomData,
			},
		)
	}

	pub fn is_valid(&self, index: Index) -> bool {
		index.to_raw() < self.raw.len()
	}
}

impl<Index: IndexTrait, T> From<Vec<T>> for IndexVec<Index, T> {
	fn from(raw: Vec<T>) -> Self {
		Self { raw, _marker: std::marker::PhantomData }
	}
}

#[allow(clippy::indexing_slicing)]
impl<Index: IndexTrait, T> std::ops::Index<Index> for IndexVec<Index, T> {
	type Output = T;

	fn index(&self, index: Index) -> &T {
		&self.raw[index.to_raw()]
	}
}

#[allow(clippy::indexing_slicing)]
impl<Index: IndexTrait, T> std::ops::IndexMut<Index> for IndexVec<Index, T> {
	fn index_mut(&mut self, index: Index) -> &mut T {
		&mut self.raw[index.to_raw()]
	}
}

impl<'a, Index: IndexTrait, T> IntoIterator for &'a IndexVec<Index, T> {
	type Item = &'a T;
	type IntoIter = std::slice::Iter<'a, T>;
	fn into_iter(self) -> Self::IntoIter {
		self.raw.iter()
	}
}

impl<'a, Index: IndexTrait, T> IntoIterator for &'a mut IndexVec<Index, T> {
	type Item = &'a mut T;
	type IntoIter = std::slice::IterMut<'a, T>;
	fn into_iter(self) -> Self::IntoIter {
		self.raw.iter_mut()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct IndexSliceMut<'a, Index: IndexTrait, T> {
	pub raw: &'a mut [T],
	offset: usize,
	_marker: std::marker::PhantomData<Index>,
}

impl<'a, Index: IndexTrait, T> std::ops::Index<Index> for IndexSliceMut<'a, Index, T> {
	type Output = T;

	fn index(&self, index: Index) -> &T {
		&self.raw[index.to_raw() - self.offset]
	}
}

impl<'a, Index: IndexTrait, T> std::ops::IndexMut<Index> for IndexSliceMut<'a, Index, T> {
	fn index_mut(&mut self, index: Index) -> &mut T {
		&mut self.raw[index.to_raw() - self.offset]
	}
}

impl<'a, Index: IndexTrait, T> std::ops::Deref for IndexSliceMut<'a, Index, T> {
	type Target = [T];

	fn deref(&self) -> &[T] {
		self.raw
	}
}

impl<'a, Index: IndexTrait, T> std::ops::DerefMut for IndexSliceMut<'a, Index, T> {
	fn deref_mut(&mut self) -> &mut [T] {
		self.raw
	}
}

//--------------------------------------------------------------------------------------------------
