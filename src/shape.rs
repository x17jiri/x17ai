// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use core::fmt;
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
		let elems = dims.iter().product();
		Rc::new(Self { __dims: dims.into(), __elems: elems })
	}

	pub fn new_transposed(&self, x1: usize, x2: usize) -> Rc<Self> {
		let mut new_dims = self.__dims.clone();
		new_dims.swap(x1, x2);
		Rc::new(Self { __dims: new_dims, __elems: self.__elems })
	}

	pub fn new_reduced(&self, dims_to_reduce: &[usize]) -> Rc<Self> {
		let mut new_dims = self.__dims.clone();
		for dim in dims_to_reduce {
			if *dim >= new_dims.len() {
				panic!("Invalid dimension");
			}
			new_dims[*dim] = 1;
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

	pub fn elems(&self) -> usize {
		self.__elems
	}

	pub fn broadcast_type(&self, other: &Self) -> BroadcastType {
		let ndim = std::cmp::min(self.ndim(), other.ndim());
		let a = &self.__dims[self.ndim() - ndim..];
		let b = &other.__dims[other.ndim() - ndim..];
		let mut a_broadcast = false;
		let mut b_broadcast = false;
		for i in 0..ndim {
			if a[i] != b[i] {
				if a[i] == 1 {
					a_broadcast = true;
				} else if b[i] == 1 {
					b_broadcast = true;
				} else {
					return BroadcastType::Error;
				}
			}
		}
		if a_broadcast || b_broadcast {
			let ndim = self.ndim().max(other.ndim());

			// Number of dimensions of len 1 to add to each shape
			let a1 = ndim - self.ndim();
			let b1 = ndim - other.ndim();

			let mut new_shape = vec![0; ndim];
			let mut new_elems = 1;
			for i in 0..ndim {
				let a_dim = if i < a1 { 1 } else { self.__dims[i - a1] };
				let b_dim = if i < b1 { 1 } else { other.__dims[i - b1] };
				new_shape[i] = std::cmp::max(a_dim, b_dim);
				new_elems *= new_shape[i]; // TODO - check for overflow
			}

			BroadcastType::Broadcast(
				a_broadcast,
				b_broadcast,
				Rc::new(Self {
					__dims: new_shape.into_boxed_slice(),
					__elems: new_elems,
				}),
			)
		} else {
			BroadcastType::NoBroadcast
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
