// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::fmt;
use std::intrinsics::{likely, unlikely};
use std::rc::Rc;

pub const MAX_DIMS: usize = 5;

#[derive(Debug, PartialEq, Clone)]
pub struct Shape {
	pub __ndim: usize,
	pub __dims: [usize; MAX_DIMS],
	pub __elems: usize,
}

impl Shape {
	pub fn new_scalar() -> Self {
		Self {
			__ndim: 0,
			__dims: [1; MAX_DIMS],
			__elems: 1,
		}
	}

	pub fn new(dims: &[usize]) -> Rc<Self> {
		assert!(dims.len() <= MAX_DIMS);
		let mut __dims = [1; MAX_DIMS];
		for (o, i) in __dims.iter_mut().zip(dims.iter()) {
			*o = *i;
		}
		Rc::new(Self {
			__ndim: dims.len(),
			__dims,
			__elems: dims.iter().product(),
		})
	}

	// Converts a negative dimension number to a positive one
	pub fn dim_to_usize(&self, dim: isize) -> Option<usize> {
		let dim = if dim >= 0 { dim as usize } else { self.__ndim - ((-dim) as usize) };
		if likely(dim < self.__ndim) { Some(dim) } else { None }
	}

	pub fn replace_last_n(&mut self, n: usize, replacement: &[usize]) {
		let start = self.__ndim.saturating_sub(n);
		self.__ndim = start + replacement.len();
		assert!(self.__ndim <= MAX_DIMS);

		// extend replacement with 1s
		let t = replacement.iter().chain(std::iter::repeat(&1));

		for (o, i) in self.__dims[start..].iter_mut().zip(t) {
			*o = *i;
		}

		self.__elems = self.__dims.iter().product();
	}

	// Prepends n dimensions of size 1 to the shape
	// example:
	//     shape = [2, 3]
	//     shape.prepend_dims(2)
	//     shape = [1, 1, 2, 3]
	pub fn prepend_dims(&mut self, n: usize) {
		assert!(self.__ndim + n <= MAX_DIMS);
		self.__ndim += n;
		for i in (n..self.__ndim).rev() {
			self.__dims[i] = self.__dims[i - n];
		}
		for i in 0..n {
			self.__dims[i] = 1;
		}
	}

	// Merges the first n dimensions into a single dimension
	pub fn merge_dims(&mut self, n: usize) {
		assert!(n > 0);
		assert!(n <= self.__ndim);
		assert!(self.__ndim > 0);

		self.__dims[0] = self.__dims[..n].iter().product();
		self.__ndim = self.__ndim - n + 1;
		for i in 1..self.__ndim {
			self.__dims[i] = self.__dims[i + n];
		}
	}

	pub fn swap(&mut self, dim1: usize, dim2: usize) {
		self.__dims.swap(dim1, dim2);
	}

	pub fn ndim(&self) -> usize {
		self.__ndim
	}

	pub fn elems(&self) -> usize {
		self.__elems
	}

	pub fn iter(&self) -> std::slice::Iter<usize> {
		self.__dims.iter()
	}
}

impl std::ops::Index<isize> for Shape {
	type Output = usize;

	fn index(&self, i: isize) -> &usize {
		let ndim = self.ndim();
		if let i = self.dim_to_usize(i) {
			&self.__dims[i]
		} else {
			cold_path();
			&1
		}
	}
}

impl std::ops::IndexMut<isize> for Shape {
	fn index_mut(&mut self, i: isize) -> &mut usize {
		let ndim = self.ndim();
		let i = self.dim_to_usize(i).unwrap();
		&mut self.__dims[i]
	}
}

impl std::ops::Index<std::ops::Range<isize>> for Shape {
	type Output = [usize];

	fn index(&self, r: std::ops::Range<isize>) -> &[usize] {
		let start = self.dim_to_usize(r.start).unwrap();
		let end = self.dim_to_usize(r.end).unwrap();
		&self.__dims[start..end]
	}
}

impl std::ops::Index<std::ops::RangeTo<isize>> for Shape {
	type Output = [usize];

	fn index(&self, r: std::ops::RangeTo<isize>) -> &[usize] {
		let end = self.dim_to_usize(r.end).unwrap();
		&self.__dims[..end]
	}
}

impl std::ops::Index<std::ops::RangeFrom<isize>> for Shape {
	type Output = [usize];

	fn index(&self, r: std::ops::RangeFrom<isize>) -> &[usize] {
		let start = self.dim_to_usize(r.start).unwrap();
		&self.__dims[start..]
	}
}

impl std::ops::Index<std::ops::RangeFull> for Shape {
	type Output = [usize];

	fn index(&self, _: std::ops::RangeFull) -> &[usize] {
		&self.__dims[..self.ndim()]
	}
}

impl fmt::Display for Shape {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "[")?;
		for i in 0..self.__dims.len() {
			if i != 0 {
				write!(f, ", ")?;
			}
			write!(f, "{}", self.__dims[i])?;
		}
		write!(f, "]")
	}
}
