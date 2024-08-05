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
#![feature(arbitrary_self_types)]
#![feature(dispatch_from_dyn)]

#[cold]
fn cold_path() {}

#[derive(Debug)]
pub enum Error {
	TooManyDims,
	TooManyElems,
}

mod buffer;
mod cpu;
mod device;
mod dtype;
mod format;
mod rand;
mod tensor;

use crate::buffer::*;
use crate::cpu::*;
use crate::device::*;
use crate::dtype::*;
use crate::format::*;
use crate::rand::*;
use crate::tensor::*;
use smallvec::{smallvec, SmallVec};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

// Inspired by Adam-mini: https://arxiv.org/abs/2406.16793
pub struct AdamParam {
	pub parts: usize,
	pub part_elems: usize,
	pub value: Tensor,   // shape: [parts, part_elems]
	pub grad: Tensor,    // shape: [parts, part_elems]
	pub m: Tensor,       // shape: [parts, part_elems]
	pub v: Tensor,       // shape: [parts, 1]
	pub v_recip: Tensor, // shape: [parts, 1]
	pub stored_tensors: Vec<Tensor>,
	pub beta1: f64,
	pub beta2: f64,
	pub eps: f64,
}

impl OptParam for AdamParam {
	// grad needs to have shape: [*, parts, part_elems]
	// where * is any number of batch dimensions
	fn acc(&mut self, grads: Tensor) {
		assert!(grads.ndim() >= self.grad.ndim());
		assert!(grads.size(-2) == self.grad.size(-2));
		assert!(grads.size(-1) == self.grad.size(-1));

		// merge all the batch dimensions into one
		let batch_size = grads.elems() / self.grad.elems();
		let grads = grads.reshape(&[batch_size, self.parts, self.part_elems]);

		// permute dimensions: [batch, ...] -> [..., batch]
		let grads = grads.permuted(&[1, 2, 0]);

		self.grad.acc_sum_(1.0, 1.0, &grads, false);
	}

	fn zero_grad(&mut self) {
		self.grad.zeros_();
	}

	fn step(&mut self, learning_rate: f64) {
		self.m.acc_(self.beta1, 1.0 - self.beta1, &self.grad);
		self.v.acc_mean_(self.beta2, 1.0 - self.beta2, &self.grad.square(), true);
		self.v.rsqrt_into(self.eps, &self.v_recip);
		self.value.acc_mul_(1.0, -learning_rate, &self.m, &self.v_recip);
	}

	fn push_tensor(&mut self, tensor: Tensor) {
		self.stored_tensors.push(tensor);
	}

	fn pop_tensor(&mut self) -> Tensor {
		self.stored_tensors.pop().unwrap()
	}

	fn value(&self) -> &Tensor {
		&self.value
	}
}

pub struct Context {
	pub learning_rate: f64,
	pub beta1: f64,
	pub beta2: f64,
	pub eps: f64,
	pub params: Vec<Rc<RefCell<AdamParam>>>,
	pub device: Rc<dyn Device>,
}

impl Context {
	pub fn new(device: Rc<dyn Device>) -> Context {
		Context {
			learning_rate: 0.001,
			beta1: 0.9,
			beta2: 0.999,
			eps: 1e-8,
			params: Vec::new(),
			device,
		}
	}

	pub fn add_param(
		&mut self,
		dtype: DType,
		parts: usize,
		part_elems: usize,
	) -> Rc<RefCell<dyn OptParam>> {
		let value = Tensor::new_empty_on(&[parts, part_elems], dtype, self.device.clone());
		let grad = value.new_empty_like();
		let m = value.new_empty_like();

		let v = Tensor::new_empty_on(&[parts, 1], dtype, self.device.clone());
		let v_recip = v.new_empty_like();

		let param = Rc::new(RefCell::new(AdamParam {
			parts,
			part_elems,
			value,
			grad,
			m,
			v,
			v_recip,
			stored_tensors: Vec::new(),
			beta1: self.beta1,
			beta2: self.beta2,
			eps: self.eps,
		}));
		self.params.push(param.clone());
		param
	}

	pub fn zero_grad(&mut self) {
		for param in &self.params {
			param.borrow_mut().zero_grad();
		}
	}

	pub fn step(&mut self, learning_rate: f64) {
		for param in &self.params {
			param.borrow_mut().step(learning_rate);
		}
	}
}

pub trait OptParam {
	fn acc(&mut self, value: Tensor);
	fn zero_grad(&mut self);
	fn step(&mut self, learning_rate: f64);
	fn push_tensor(&mut self, tensor: Tensor);
	fn pop_tensor(&mut self) -> Tensor;
	fn value(&self) -> &Tensor;
}

/// Linear layer transforming inputs to outputs
///
/// This is basically a thin wrapper around a matrix multiplication.
/// It does not include a bias term.
///
/// If nhead != 0, this works like n parallel linear transformations of the same input:
///
///     input: [*, inputs]
///     output: [*, head, outputs]
///
/// Otherwise:
///
///     input: [*, inputs]
///     output: [*, outputs]
///
/// Even with multiple heads, there is only one matrix multiplication. The output is then reshaped
/// to include the head dimension.
struct Linear {
	pub inputs: usize,
	pub outputs: usize,
	pub nhead: usize,
	pub parts: usize,
	pub w: Tensor,
	pub w_opt: Rc<RefCell<dyn OptParam>>,
	pub scale: f64,
	pub backward_scale: f64,
	pub dtype: DType,
}

impl Linear {
	pub fn new(
		inputs: usize,
		outputs: usize,
		nhead: usize,
		dtype: DType,
		ctx: &mut Context,
	) -> Linear {
		let parts = nhead.max(1);
		let w_opt = ctx.add_param(dtype, parts, outputs * inputs);
		let w = w_opt.borrow().value().clone();
		let w = w.reshape(&[parts * outputs, inputs]);
		let scale = MatMul::normalizing_scale(inputs);
		let backward_scale = MatMul::normalizing_scale(outputs);

		#[rustfmt::skip]
		Linear {
			inputs, outputs, nhead: parts, parts,
			w, w_opt,
			scale, backward_scale, dtype,
		}
	}

	fn forward(&self, x: Tensor) -> Tensor {
		let mut w_opt = self.w_opt.borrow_mut();

		let y = mm(&self.w, x.as_col());
		let y = y.scale(self.scale).assert_normalizing_scale();
		let y = y.eval();

		w_opt.push_tensor(x);

		if self.nhead != 0 {
			// [..., head * outputs] -> [..., head, outputs]
			y.reshape_last_n(1, &[self.nhead, self.outputs])
		} else {
			y
		}
	}

	fn backward(&self, mut dy: Tensor) -> Tensor {
		let mut w_opt = self.w_opt.borrow_mut();
		let x = w_opt.pop_tensor();

		if self.nhead != 0 {
			// [..., head, outputs] -> [..., head * outputs]
			dy = dy.reshape_last_n(2, &[self.nhead * self.outputs])
		};
		let grad = mm(&self.w, x.as_col()).backward(&dy);

		// dw
		let dw = grad.da();
		let dw = dw.scale(1.0).assert_normalizing_scale();
		let dw = dw.eval();

		// Reshape the gradient to the shape expected by the optimizer
		// [..., parts * outputs, inputs] -> [..., parts, outputs * inputs]
		let dw = dw.reshape_last_n(2, &[self.parts, self.outputs * self.inputs]);

		w_opt.acc(dw);

		// dx
		let dx = grad.db();
		let dx = dx.scale(self.backward_scale).assert_normalizing_scale();
		let dx = dx.eval();

		dx
	}
}

/*
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
}

impl Module for Attention {
	// input is of the form: [..., inputs, embeding]
	fn forward(&self, input: &Tensor) -> Tensor {
		// explanation of dimension names:
		// *: batch (can be any number of dimensions >= 0)
		// i: input sequence
		// h: head
		// q, k, v: key, query, value

		// TODO - use scopes so tensors are freed when not needed

		// input: [*, i, input_features]
		let seq_len = input.shape()[-2];

		// k: [*, i, k]
		// -> [*, 1, i, k]
		// -> [*, h, i, k]
		let k = self.k.forward(input);
		let k = k.reshape_last_n(2, &[1, seq_len, self.qk_size]);

		// q: [*, i, h * q]
		// -> [*, i, h, q]
		// -> [*, h, i, q]
		// -> [*, h, q, i]
		let q = self.q.forward(input);
		let q = q.reshape_last_n(1, &[self.heads, self.qk_size]);
		let q = q.transposed(-3, -2);
		let q = q.transposed(-2, -1);

		// v: [*, i, h * v]
		// -> [*, i, h, v]
		// -> [*, h, i, v]
		let v = self.v.forward(input);
		let v = v.reshape_last_n(1, &[self.heads, self.v_size]);
		let w_shape = v.shape().to_vec(); // [*, i, h, v]
		let v = v.transposed(-3, -2);

		// scores: [*, h, i, i]
		let scores = matmul(&k, &q);

		// w = reweighted v
		// w: [*, h, i, v]
		// -> [*, i, h, v]
		// -> [*, i, w = h * v]

		let w = v.new_tensor(&w_shape, v.dtype()); // [*, i, h, v]
		let w = w.transposed(-3, -2); // [*, h, i, v]

		matmul_(scores, v, w);
		let w = w.transposed(-3, -2); // [*, i, h, v]
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
*/
fn main() {
	stderrlog::new().module(module_path!()).init().unwrap();

	let dev = CPUDevice::new("CPU".to_string());

	let x = Tensor::new_empty_on(&[2, 3], DType::f32(), dev.clone());
	let y = Tensor::new_empty_on(&[3, 2], DType::f32(), dev.clone());

	x.randn_();
	y.randn_();

	println!("x = {}", x);
	println!("y = {}", y);

	let nx = rms_norm(&x);
	let ny = rms_norm(&y);

	println!("rms_norm(x) = {}", nx);
	println!("rms_norm(y) = {}", ny);

	let xs = softmax(&x);
	let ys = softmax(&y);

	println!("softmax(x) = {}", xs);
	println!("softmax(y) = {}", ys);

	//	let z = Tensor::new(&[2, 2], DType::f32(), dev.clone());
	let z = mm(y.T(), x.T()).eval();

	println!("z = {}", z);
}
