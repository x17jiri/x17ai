// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::rand::Rng;
use crate::shape::Shape;

use std::fmt;
use std::ops::Index;
use std::rc::Rc;

//--------------------------------------------------------------------------------------------------
// DType

#[derive(Debug, Clone, Copy)]
pub enum DType {
	Float(u8),
	Int(u8),
	Uint(u8),
}

//--------------------------------------------------------------------------------------------------
// Device

pub trait Device {
	fn name(&self) -> &str;

	fn new_uninit(self: Rc<Self>, dtype: DType, elems: usize) -> Option<Rc<dyn Buffer>>;
}

//--------------------------------------------------------------------------------------------------
// Buffer

pub trait Buffer {
	fn device(&self) -> Rc<dyn Device>;

	fn dtype(&self) -> DType;

	fn zero_all_(&self);
	fn randn_all_(&self, rng: &mut Rng);

	fn zero_(&self, shape: &Shape);
	fn randn_(&self, shape: &Shape, rng: &mut Rng);

	fn exp(&self, shape: &Shape) -> Tensor;

	fn sum(&self, shape: &Shape) -> Tensor;
	fn max(&self, shape: &Shape) -> Tensor;

	// requirement:
	//     index.len() == shape.ndim() - 1
	// This takes indexes for all except the last dimension and
	// prints values in the last dimension.
	fn __format(
		&self,
		f: &mut fmt::Formatter,
		off: isize,
		len: usize,
		stride: isize,
	) -> fmt::Result;
}

//--------------------------------------------------------------------------------------------------
// Tensor

#[derive(Clone)]
pub struct Tensor {
	pub buf: Rc<dyn Buffer>,
	pub shape: Shape,
}

impl Tensor {
	pub fn new_zeros(dev: Rc<dyn Device>, dtype: DType, dims: &[usize]) -> Tensor {
		let (shape, elems) = Shape::new(dims).ok().unwrap();
		let buf = dev.new_uninit(dtype, elems).unwrap();
		buf.zero_(&shape);
		Tensor { buf, shape }
	}

	pub fn new_randn(dev: Rc<dyn Device>, dtype: DType, dims: &[usize], rng: &mut Rng) -> Tensor {
		let (shape, elems) = Shape::new(dims).ok().unwrap();
		let buf = dev.new_uninit(dtype, elems).unwrap();
		buf.randn_all_(rng);
		Tensor { buf, shape }
	}

	//-- nullary operations

	pub fn zero_(&self) {
		self.buf.zero_(&self.shape);
	}

	pub fn randn_(&self, rng: &mut Rng) {
		self.buf.randn_(&self.shape, rng);
	}

	//-- unary operations

	pub fn exp(&self) -> Tensor {
		self.buf.exp(&self.shape)
	}

	//-- reduction operations

	pub fn sum(&self) -> Tensor {
		self.buf.sum(&self.shape)
	}

	pub fn max(&self) -> Tensor {
		self.buf.max(&self.shape)
	}

	//--

	pub fn slice<R: std::ops::RangeBounds<usize>>(&self, dim: usize, range: R) -> Tensor {
		Tensor {
			buf: self.buf.clone(),
			shape: self.shape.slice(dim, range),
		}
	}
}

//--------------------------------------------------------------------------------------------------
