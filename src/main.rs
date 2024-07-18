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

// Linear layer transforming inputs to outputs
//
// input: [..., inputs]
// output: [..., outputs]
struct Linear {
	pub inputs: usize,
	pub outputs: usize,
	pub weights: Tensor,
}

impl Linear {
	pub fn new(inputs: usize, outputs: usize, alloc: &mut dyn Allocator) -> Linear {
		let weights = alloc.new_tensor(DType::f32(), Shape::new(&[outputs, inputs]));
		Linear { inputs, outputs, weights }
	}

	pub fn forward(&self, input: Rc<Expr>) -> Rc<Expr> {
		let weights = self.weights.as_expr();
		m_dot_v(weights, input)
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
		alloc: &mut dyn Allocator,
	) -> Attention {
		let q = Linear::new(embeding, heads * qk_size, alloc);
		let k = Linear::new(embeding, heads * qk_size, alloc);
		let v = Linear::new(embeding, heads * v_size, alloc);
		let mix = Linear::new(heads * v_size, embeding, alloc);

		Attention {
			heads,
			inputs,
			embeding,
			qk_size,
			v_size,
			q,
			k,
			v,
			mix,
		}
	}

	// input is of the form: [..., inputs, embeding]
	pub fn forward(&self, input: Rc<Expr>) -> Rc<Expr> {
		// input: [..., inputs, embeding]
		let input = rms_norm(input);

		// q: [..., inputs, heads * qk_size]
		let q = self.q.forward(input);
		// q: [..., inputs, heads, qk_size]
		let q = q.reshape(-1, &[self.heads, self.qk_size]);

		// k: [..., inputs, heads * qk_size]
		let k = self.k.forward(input);
		// k: [..., inputs, heads, qk_size]
		let k = k.reshape(-1, &[self.heads, self.qk_size]);

		// v: [..., inputs, heads * v_size]
		let v = self.v.forward(input);
		// v: [..., inputs, heads, v_size]
		let v = v.reshape(-1, &[self.heads, self.v_size]);

		// scores: [..., inputs, heads, heads]
		let scores = m_dot_mt(q, k);
		let scores = attention_mask(scores);
		let scores = softmax(scores);

		// w: [..., inputs, heads, v_size]
		let w = m_dot_m(scores, v);
		// w: [..., inputs, heads * v_size]
		let w = w.reshape(-2, &[self.heads * self.v_size]);

		// mix: [..., inputs, 2, embeding]
		let mix = self.mix.forward(w);

		// result: [..., inputs, embeding]
		silu_glu(mix)
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
