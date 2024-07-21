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
//use crate::expr::*;
use crate::format::*;
use crate::rand::*;
use crate::shape::*;
use crate::tensor::*;
use std::rc::Rc;

pub struct Context {
	pub alloc: ScopedAllocator,
}

impl Context {
	pub fn new(alloc: ScopedAllocator) -> Context {
		Context { alloc }
	}

	pub fn scoped_tensor(&self, shape: Rc<Shape>, dtype: DType) -> ScopedTensor {
		self.alloc.new_tensor(shape, dtype)
	}
}

pub trait Module {
	fn output_info(&self, input: &Tensor) -> (Rc<Shape>, DType);

	fn forward_(&self, input: &Tensor, output: &Tensor, _ctx: &Context);

	fn forward(&self, input: &Tensor, ctx: &Context) -> ScopedTensor {
		let (shape, dtype) = self.output_info(input);
		let output = ctx.scoped_tensor(shape, dtype);
		self.forward_(input, output.get(), ctx);
		output
	}
}

// Linear layer transforming inputs to outputs
//
// input: [..., inputs]
// output: [..., outputs]
struct Linear {
	pub inputs: usize,
	pub outputs: usize,
	pub weights: Tensor,
	pub scale: f64,
	pub dtype: DType,
}

impl Linear {
	pub fn new(inputs: usize, outputs: usize, dtype: DType, alloc: &mut dyn Allocator) -> Linear {
		let shape = Shape::new(&[outputs, inputs]);
		let weights = alloc.new_tensor(shape, dtype);
		let scale = 1.0 / (inputs as f64).sqrt();
		Linear { inputs, outputs, weights, scale, dtype }
	}
}

impl Module for Linear {
	fn output_info(&self, input: &Tensor) -> (Rc<Shape>, DType) {
		let shape = input.shape.reshape(-1, &[self.outputs]);
		(shape, self.dtype)
	}

	fn forward_(&self, input: &Tensor, output: &Tensor, _ctx: &Context) {
		gemm(self.scale, &self.weights, input, 0.0, &output);
	}
}

struct RMSNorm;

impl RMSNorm {
	pub fn new() -> RMSNorm {
		RMSNorm
	}
}

impl Module for RMSNorm {
	fn output_info(&self, input: &Tensor) -> (Rc<Shape>, DType) {
		(input.shape.clone(), input.dtype)
	}

	fn forward_(&self, input: &Tensor, output: &Tensor, _ctx: &Context) {
		rms_norm(input, output);
	}
}

struct Attention {
	pub input_features: usize,
	pub heads: usize,
	pub qk_size: usize,
	pub v_size: usize,
	pub dtype: DType,

	pub k: Linear,
	pub q: Linear,
	pub v: Linear,
}

impl Attention {
	pub fn new(
		input_features: usize,
		heads: usize,
		qk_size: usize,
		v_size: usize,
		dtype: DType,
		alloc: &mut dyn Allocator,
	) -> Attention {
		let k = Linear::new(input_features, qk_size, dtype, alloc);
		let q = Linear::new(input_features, heads * qk_size, dtype, alloc);
		let v = Linear::new(input_features, heads * v_size, dtype, alloc);

		Attention {
			input_features,
			heads,
			qk_size,
			v_size,
			dtype,
			k,
			q,
			v,
		}
	}

	// input is of the form: [..., inputs, embeding]
	pub fn forward(&self, input: &Tensor, ctx: &Context) -> Tensor {
		// explanation of dimension names:
		// *: batch (can be any number of dimensions >= 0)
		// i: input sequence
		// h: head
		// q, k, v: key, query, value

		// input: [*, i, input_features]
		let seq_len = input.shape[-2];

		// k: [*, i, k]
		// -> [*, 1, i, k]
		// -> [*, h, i, k]
		let k = self.k.forward(input, ctx);
		let k = k.get();
		let k = k.reshape_last_n(2, &[1, seq_len, self.qk_size]);
		let k = k.broadcast(-3, self.heads);

		// q: [*, i, h * q]
		// -> [*, i, h, q]
		// -> [*, h, i, q]
		let q = self.q.forward(input, ctx);
		let q = q.get();
		let q = q.reshape_last_n(1, &[self.heads, self.qk_size]);
		let q = q.transpose(-3, -2);

		// v: [*, i, h * v]
		// -> [*, i, h, v]
		// -> [*, h, i, v]
		let v = self.v.forward(input, ctx);
		let v = v.get();
		let v = v.reshape_last_n(1, &[self.heads, self.v_size]);
		let w_shape = v.shape.clone(); // [*, i, h, v]
		let v = v.transpose(-3, -2);

		// scores: [*, h, i, i]
		let scores = m_dot_m(k, q);

		// w = reweighted v
		// w: [*, h, i, v]
		// -> [*, i, h, v]
		// -> [*, i, w = h * v]

		let w = ctx.scoped_tensor(w_shape, v.dtype); // [*, i, h, v]
		let w = w.get();
		let w = w.transpose(-3, -2); // [*, h, i, v]

		m_dot_m_(scores, v, w);
		let w = w.transpose(-3, -2); // [*, i, h, v]
		let w = w.reshape_last_n(2, &[self.heads * self.v_size]);

		w
	}
}

struct Transformer {
	pub attention: Attention,
	pub feed_forward: Linear,
}

impl Transformer {
	pub fn new(
		input_features: usize,
		heads: usize,
		qk_size: usize,
		v_size: usize,
		dtype: DType,
		alloc: &mut dyn Allocator,
	) -> Transformer {
		let attention = Attention::new(input_features, heads, qk_size, v_size, dtype, alloc);
		let feed_forward = Linear::new(heads * v_size, 2 * input_features, dtype, alloc);
		Transformer { attention, feed_forward }
	}
}

impl Module for Transformer {
	fn output_info(&self, input: &Tensor) -> (Rc<Shape>, DType) {
		(input.shape.clone(), input.dtype)
	}

	fn forward_(&self, input: &Tensor, output: &Tensor, _ctx: &Context) {
		let a = self.rms_norm.forward(input);
		let b = self.attention.forward(a);
		let c = self.feed_forward.forward(b);
		swiglu_(c, output);
	}
}

fn main() {
	let dev = CPUDevice::new("CPU".to_string());
	let buf = dev.new_buffer(1024 * 1024, "my_buf".to_string());
	let mut alloc = BumpAllocator::new(buf);

	let x = alloc.new_tensor(DType::f32(), Shape::new(&[2, 3]));
	let y = alloc.new_tensor(DType::f32(), Shape::new(&[3, 2]));

	x.randn_();
	y.randn_();

	println!("x = {}", x);
	println!("y = {}", y);

	let z = alloc.new_tensor(DType::f32(), Shape::new(&[2, 2]));

	//	mm(&x, &y, &z);

	println!("z = {}", z);

	println!("capacity: {}", alloc.capacity);
	println!("allocated bytes: {}", alloc.offset);
}
