//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#[allow(clippy::len_without_is_empty)]
pub trait Buffer {
	fn len(&self) -> usize;
}

impl<T> Buffer for &[T] {
	fn len(&self) -> usize {
		<[T]>::len(self)
	}
}

impl<T> Buffer for &mut [T] {
	fn len(&self) -> usize {
		<[T]>::len(self)
	}
}

pub trait BufferSlice {
	type SliceType<'a>: Buffer + 'a
	where
		Self: 'a;

	fn buffer_slice<'a>(&'a self) -> Self::SliceType<'a>;
}

impl<T> BufferSlice for &[T] {
	type SliceType<'a>
		= &'a [T]
	where
		T: 'a,
		Self: 'a;

	fn buffer_slice<'a>(&'a self) -> Self::SliceType<'a> {
		self
	}
}

impl<T> BufferSlice for &mut [T] {
	type SliceType<'a>
		= &'a [T]
	where
		T: 'a,
		Self: 'a;

	fn buffer_slice<'a>(&'a self) -> Self::SliceType<'a> {
		&self[..]
	}
}

pub trait BufferSliceMut {
	type SliceMutType<'a>: Buffer + 'a
	where
		Self: 'a;

	fn buffer_slice_mut<'a>(&'a mut self) -> Self::SliceMutType<'a>;
}

impl<T> BufferSliceMut for &mut [T] {
	type SliceMutType<'a>
		= &'a mut [T]
	where
		T: 'a,
		Self: 'a;

	fn buffer_slice_mut<'a>(&'a mut self) -> Self::SliceMutType<'a> {
		&mut self[..]
	}
}
