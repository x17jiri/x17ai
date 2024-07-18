// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#![allow(incomplete_features)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(stmt_expr_attributes)]
#![warn(clippy::cast_lossless)]
#![feature(let_chains)]
#![allow(unused_imports)] // TODO - remove when project stabilizes

#[cold]
fn cold_path() {}

#[derive(Debug)]
pub enum Error {
	TooManyDims,
	TooManyElems,
}

mod alloc;
mod buffer;
mod cpu;
mod device;
mod dtype;
mod expr;
mod format;
mod rand;
mod shape;
mod tensor;

use crate::alloc::*;
use crate::buffer::*;
use crate::cpu::*;
use crate::device::*;
use crate::dtype::*;
use crate::expr::*;
use crate::format::*;
use crate::rand::*;
use crate::shape::*;
use crate::tensor::*;
use std::rc::Rc;

struct Linear {
	pub inputs: usize,
	pub outputs: usize,
	pub weights: Tensor,
}

impl Linear {
	pub fn new(inputs: usize, outputs: usize, buffer: Rc<dyn Buffer>) -> Linear {
		let weights = buffer.new_tensor(Shape::new(&[outputs, inputs]), DType::f32());
		Linear { inputs, outputs, weights }
	}

	pub fn forward(&self, input: &Tensor, output: &Tensor) {
		m_dot_v(&self.weights, input, output);
	}
}

struct Attention {
	pub inputs: usize,
	pub internal: usize,
	pub k: Linear,
	pub v: Linear,
	pub q: Linear,
}

impl Attention {
	pub fn new(inputs: usize, internal: usize, buffer: Rc<dyn Buffer>) -> Attention {
		let k = Linear::new(inputs, internal, buffer.clone());
		let v = Linear::new(inputs, internal, buffer.clone());
		let q = Linear::new(inputs, internal, buffer.clone());
		Attention { inputs, internal, k, v, q }
	}

	pub fn forward(
		&self,
		input: &Tensor, // [batch, input, embeding]
		output: &Tensor,
	) {
		let buf = input.buffer.clone();
		let k = buf.new_tensor(Shape::new(&[self.internal]), DType::f32());
		let v = buf.new_tensor(Shape::new(&[self.internal]), DType::f32());
		let q = buf.new_tensor(Shape::new(&[self.internal]), DType::f32());

		self.k.forward(input, &k);
		self.v.forward(input, &v);
		self.q.forward(input, &q);

		// TODO - implement the rest
	}
}

fn main() {
	let dev = CPUDevice::new("CPU".to_string());
	let buf = dev.new_buffer(1024 * 1024, "buf".to_string());

	let x = Tensor {
		shape: Shape::new(&[2, 3]),
		dtype: DType { kind: DTypeKind::Float, bits: 32 },
		buffer: buf.clone(),
		byte_offset: 0,
	};

	let y = Tensor {
		shape: Shape::new(&[3, 2]),
		dtype: DType { kind: DTypeKind::Float, bits: 32 },
		buffer: buf.clone(),
		byte_offset: 6 * 4,
	};

	randn_(&x);
	randn_(&y);

	println!("x = {}", x);
	println!("y = {}", y);

	let z = Tensor {
		shape: Shape::new(&[2, 2]),
		dtype: DType { kind: DTypeKind::Float, bits: 32 },
		buffer: buf.clone(),
		byte_offset: 12 * 4,
	};

	mm(&x, &y, &z);

	println!("z = {}", z);
}
