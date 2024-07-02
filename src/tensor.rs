// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::rand::Rng;
use crate::shape::{prep_op, prep_op_1, MatMul, Shape, Traversal};

use std::any::*;
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

	unsafe fn new_uninit(self: Rc<Self>, dtype: DType, elems: usize) -> Option<Rc<dyn Buffer>>;
}

//--------------------------------------------------------------------------------------------------
// Buffer

pub trait Buffer: Any {
	fn device(&self) -> Rc<dyn Device>;

	fn dtype(&self) -> DType;

	//-- nullary operations working on the entire buffer

	fn zero_all_(&self);
	fn randn_all_(&self, rng: &mut Rng);

	//-- nullary operations

	fn zero_(&self, traversal: Traversal<1>);
	fn randn_(&self, traversal: Traversal<1>, rng: &mut Rng);

	//-- unary operations

	fn exp(&self, traversal: Traversal<1>) -> Rc<dyn Buffer>;

	//-- reduction operations

	fn sum(&self, traversal: Traversal<1>) -> Rc<dyn Buffer>;
	fn max(&self, traversal: Traversal<1>) -> Rc<dyn Buffer>;

	//-- binary operations

	fn add(&self, other: &dyn Buffer, traversal: Traversal<2>) -> Rc<dyn Buffer>;

	//-- special

	fn matmul(&self, other: &dyn Buffer, data: MatMul) -> Rc<dyn Buffer>;

	//-- formatting

	fn format(&self, f: &mut fmt::Formatter, off: isize, len: usize, stride: isize) -> fmt::Result;
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
		let buf;
		unsafe {
			// SAFETY: zero_all_() is called immediately after this
			buf = dev.new_uninit(dtype, elems).unwrap();
		}
		buf.zero_all_();
		Tensor { buf, shape }
	}

	pub fn new_randn(dev: Rc<dyn Device>, dtype: DType, dims: &[usize], rng: &mut Rng) -> Tensor {
		let (shape, elems) = Shape::new(dims).ok().unwrap();
		let buf;
		unsafe {
			// SAFETY: randn_all_() is called immediately after this
			buf = dev.new_uninit(dtype, elems).unwrap();
		}
		buf.randn_all_(rng);
		Tensor { buf, shape }
	}

	//-- nullary operations

	pub fn zero_(&self) {
		let (traversal, _) = prep_op_1(&self.shape);
		self.buf.zero_(traversal);
	}

	pub fn randn_(&self, rng: &mut Rng) {
		let (traversal, _) = prep_op_1(&self.shape);
		self.buf.randn_(traversal, rng);
	}

	//-- unary operations

	pub fn exp(&self) -> Tensor {
		let (traversal, out_shape) = prep_op_1(&self.shape);
		let out_buf = self.buf.exp(traversal);
		Tensor { buf: out_buf, shape: out_shape }
	}

	//-- reduction operations

	pub fn sum(&self) -> Tensor {
		let (traversal, _) = prep_op_1(&self.shape);
		let out_buf = self.buf.sum(traversal);
		Tensor { buf: out_buf, shape: Shape::new_scalar() }
	}

	pub fn max(&self) -> Tensor {
		let (traversal, _) = prep_op_1(&self.shape);
		let out_buf = self.buf.max(traversal);
		Tensor { buf: out_buf, shape: Shape::new_scalar() }
	}

	//-- binary operations

	fn __prep_binary(&self, other: &Tensor) -> (Traversal<2>, Shape) {
		if self.buf.type_id() != other.buf.type_id() {
			panic!("Tensor types do not match");
		}
		let prep = prep_op([&self.shape, &other.shape]);
		if prep.is_none() {
			panic!("Tensor shapes do not match");
		}
		prep.unwrap()
	}

	pub fn add(&self, other: &Tensor) -> Tensor {
		let (traversal, out_shape) = self.__prep_binary(other);
		let out_buf = self.buf.add(other.buf.as_ref(), traversal);
		Tensor { buf: out_buf, shape: out_shape }
	}

	//-- special

	pub fn matmul(&self, other: &Tensor) -> Tensor {
		let mm = MatMul::new(&self.shape, &other.shape);
		let out_buf = self.buf.matmul(other.buf.as_ref(), mm);
		let (out_shape, _) = Shape::new(&[mm.a_rows, mm.b_cols]).ok().unwrap();
		Tensor { buf: out_buf, shape: out_shape }
	}

	//-- slicing

	pub fn slice<R: std::ops::RangeBounds<usize>>(&self, dim: usize, range: R) -> Tensor {
		Tensor {
			buf: self.buf.clone(),
			shape: self.shape.slice(dim, range),
		}
	}
}

//--------------------------------------------------------------------------------------------------
