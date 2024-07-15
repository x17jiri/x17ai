// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use core::fmt;
use std::intrinsics::{likely, unlikely};
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub struct Shape {
	// TODO - could use some sort of small vec optimization
	__dims: Box<[usize]>,
	__elems: usize,
}

pub enum BroadcastType {
	Error,
	NoBroadcast,
	Broadcast(bool, bool, Rc<Shape>),
}

impl Shape {
	pub fn new_scalar() -> Rc<Self> {
		Rc::new(Self {
			__dims: Vec::new().into_boxed_slice(),
			__elems: 1,
		})
	}

	pub fn new(dims: &[usize]) -> Rc<Self> {
		Self::from_iter(dims.iter().copied())
	}

	pub fn from_iter<'a, D: Iterator<Item = usize> + ExactSizeIterator<Item = usize>>(
		dims: D,
	) -> Rc<Self> {
		let ndim = dims.len();
		let mut elems: usize = 1;
		let mut vec = Vec::<usize>::with_capacity(ndim);
		for (i, dim) in dims.enumerate() {
			unsafe {
				let p = vec.as_mut_ptr();
				let p = p.add(i);
				std::ptr::write(p, dim);
			}

			if let Some(e) = elems.checked_mul(dim) {
				elems = e;
			} else {
				panic!("Too many elements");
			}
		}
		unsafe {
			vec.set_len(ndim);
		}
		Rc::new(Self {
			__dims: vec.into_boxed_slice(),
			__elems: elems,
		})
	}

	pub fn new_transposed(&self, x1: usize, x2: usize) -> Rc<Self> {
		let mut new_dims = self.__dims.clone();
		new_dims.swap(x1, x2);
		Rc::new(Self { __dims: new_dims, __elems: self.__elems })
	}

	pub fn new_reduced(&self, dims_to_reduce: &[isize]) -> Rc<Self> {
		let mut new_dims = self.__dims.clone();
		let ndim = new_dims.len();
		for dim in dims_to_reduce {
			let dim = if *dim >= 0 {
				*dim as usize
			} else {
				ndim.wrapping_sub(dim.wrapping_neg() as usize)
			};
			if unlikely(dim >= ndim) {
				panic!("Invalid dimension");
			}
			new_dims[dim] = 1;
		}
		let elems = new_dims.iter().product();
		Rc::new(Self { __dims: new_dims, __elems: elems })
	}

	pub fn ndim(&self) -> usize {
		self.__dims.len()
	}

	pub fn dims(&self) -> &[usize] {
		&self.__dims
	}

	fn __fix_dim_index(&self, i: isize) -> Option<usize> {
		let ndim = self.ndim();
		let i = if i >= 0 {
			i as usize
		} else {
			ndim - ((-i) as usize)
		};
		if unlikely(i >= ndim) { None } else { Some(i) }
	}

	pub fn dim(&self, i: isize) -> usize {
		if let Some(i) = self.__fix_dim_index(i) {
			self.__dims[i]
		} else {
			1
		}
	}

	pub fn elems(&self) -> usize {
		self.__elems
	}

	pub fn broadcast_type(&self, other: &Self) -> BroadcastType {
		if self.__dims == other.__dims {
			return BroadcastType::NoBroadcast;
		}
		let a_ndim = self.ndim();
		let b_ndim = other.ndim();
		let a_dims = &self.__dims;
		let b_dims = &other.__dims;

		let ndim = a_ndim.max(b_ndim);
		let a_prefix = ndim - a_ndim;
		let b_prefix = ndim - b_ndim;

		let mut dims = vec![0; ndim];
		let mut elems: usize = 1;
		let mut a_broadcast = false;
		let mut b_broadcast = false;

		for i in 0..ndim {
			let a_dim = if i < a_prefix {
				1
			} else {
				a_dims[i - a_prefix]
			};

			let b_dim = if i < b_prefix {
				1
			} else {
				b_dims[i - b_prefix]
			};

			if a_dim != b_dim {
				if a_dim == 1 {
					a_broadcast = true;
					dims[i] = b_dim;
				} else if b_dim == 1 {
					b_broadcast = true;
					dims[i] = a_dim;
				} else {
					return BroadcastType::Error;
				}
			} else {
				dims[i] = a_dim;
			}

			if let Some(e) = elems.checked_mul(dims[i]) {
				elems = e;
			} else {
				return BroadcastType::Error;
			}
		}

		BroadcastType::Broadcast(
			a_broadcast,
			b_broadcast,
			Rc::new(Self {
				__dims: dims.into_boxed_slice(),
				__elems: elems,
			}),
		)
	}
}

impl std::ops::Index<isize> for Shape {
	type Output = usize;

	fn index(&self, i: isize) -> &usize {
		if let Some(i) = self.__fix_dim_index(i) {
			&self.__dims[i]
		} else {
			&1
		}
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
