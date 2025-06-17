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
