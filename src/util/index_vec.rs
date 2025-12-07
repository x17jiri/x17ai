//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------

pub trait IndexTrait {
	fn to_raw(self) -> usize;
	fn from_raw(raw: usize) -> Self;
}

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
				$name { raw }
			}

			pub fn new_invalid() -> Self {
				$name { raw: usize::MAX }
			}

			#[allow(clippy::cast_possible_wrap)]
			pub fn is_valid(&self) -> bool {
				(self.raw as isize) >= 0
			}
		}

		impl $crate::util::index_vec::IndexTrait for $name {
			fn to_raw(self) -> usize {
				self.raw
			}

			fn from_raw(raw: usize) -> Self {
				$name { raw }
			}
		}
	};
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
