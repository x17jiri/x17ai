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
	// number of heads
	pub heads: usize,

	// length of input sequence
	pub inputs: usize,

	// size of embeding
	pub embeding: usize,

	// size of q and k
	pub qk_size: usize,

	// size of v
	pub v_size: usize,

	pub rms_norm: RMSNorm,
	pub q: Linear,
	pub k: Linear,
	pub v: Linear,
	pub mix: Linear,
}

impl Attention {
	pub fn new(
		heads: usize,
		inputs: usize,
		embeding: usize,
		qk_size: usize,
		v_size: usize,
		dtype: DType,
		alloc: &mut dyn Allocator,
	) -> Attention {
		let rms_norm = RMSNorm::new();
		let q = Linear::new(embeding, heads * qk_size, dtype, alloc);
		let k = Linear::new(embeding, heads * qk_size, dtype, alloc);
		let v = Linear::new(embeding, heads * v_size, dtype, alloc);
		let mix = Linear::new(heads * v_size, 2 * embeding, dtype, alloc);

		Attention {
			heads,
			inputs,
			embeding,
			qk_size,
			v_size,
			rms_norm,
			q,
			k,
			v,
			mix,
		}
	}

	//
	// q: [batch, inputs, heads, qk_size]
	// q: [batch, inputs * heads, qk_size]
	// k: [batch, inputs, qk_size]
	//
	// scores = q * k.T
	// scores: [batch, inputs, heads, inputs]
	// scores: [batch, inputs * heads, inputs]
	//
	//
	//
	//
	//
	//
	//
	//

	// input is of the form: [..., inputs, embeding]
	pub fn forward(&self, input: Rc<Expr>) -> Rc<Expr> {
		// input: [batch, inputs, embedings]
		let input = self.rms_norm.forward(input);
		let seq_len = input.shape[-2];

		// k: [batch, seq, qk_size]
		let k = self.k.forward(input);

		// q: [batch, seq, heads * qk_size]
		let q = self.q.forward(input);
		// q: [batch, seq * heads, qk_size]
		let q = q.reshape(-2, &[seq_len * self.heads, self.qk_size]);
		// q: [batch, qk_size, seq * heads]
		let q = q.T;

		// scores: [batch, seq, heads * seq]
		let scores = m_dot_m(k, q);

		// v: [batch, seq, heads * v_size]
		let v = self.v.forward(input);
		// v: [batch, seq * heads, v_size]
		let v = v.reshape(-2, &[seq_len * self.heads, self.v_size]);
		// v: [batch, v_size, seq * heads]
		let v = v.T;

		let w = m_dot_v(v, scores);

		//
		//
		//
		//
		//
		//
		//
		//

		// q: [batch, inputs, heads, qk_size]
		let q = self.q.forward(input);
		// q: [batch, heads, inputs, qk_size]
		let q = q.transpose(1, 2);

		// k: [batch, inputs, qk_size]
		let k = self.k.forward(input);
		// k: [batch, heads, inputs, qk_size]
		let k = k.transpose(1, 2);

		// scores: [batch, inputs, inputs, heads]
		let scores = m_dot_mt(q, k);
		let scores = attention_mask(scores);
		let scores = softmax(scores);

		// v: [batch, inputs, heads,  v_size]
		let v = self.v.forward(input);
		let v = v.reshape(-1, &[self.heads, self.v_size]);

		// w: [batch, inputs, v_size * heads]
		let w = m_dot_m(scores, v);
		let w = w.reshape(-2, &[self.heads * self.v_size]);

		// mix: [batch, inputs, 2 * embeding]
		let mix = self.mix.forward(w);

		// result: [batch, inputs, embeding]
		swiglu(mix)
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
