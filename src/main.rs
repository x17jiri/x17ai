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
#![allow(dead_code)] // TODO - remove when project stabilizes
#![feature(arbitrary_self_types)]
#![feature(dispatch_from_dyn)]
#![feature(generic_const_exprs)]

#[cold]
fn cold_path() {}

mod batch;
mod buffer;
mod cpu;
mod device;
mod dim_merger;
mod dtype;
mod expr;
mod format;
mod matrix;
mod rand;
mod tensor;

use crate::batch::*;
use crate::buffer::*;
use crate::cpu::*;
use crate::device::*;
use crate::dim_merger::*;
use crate::dtype::*;
use crate::expr::*;
use crate::format::*;
use crate::matrix::*;
use crate::rand::*;
use crate::tensor::*;
use smallvec::{SmallVec, smallvec};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/*

	b.acc_to(a, alpha, beta)                       # acc_(a, alpha, b, beta)

	b.vec_mul(c).save_to(a)                        # acc_mul_(a, 0.0, b, c, 1.0)
	b.vec_mul(c).acc_to(a, alpha, beta)            # acc_mul_(a, alpha, b, c, beta)

	b.sum(keepdim).save_to(a)                      # acc_sum_(a, 0.0, b, 1.0)
	b.sum(keepdim).acc_to(a, alpha, beta)          # acc_sum_(a, alpha, b, beta)

	b.sum_square(keepdim).save_to(a)               # acc_sum_square_(a, 0.0, b, 1.0)
	b.sum_square(keepdim).acc_to(a, alpha, beta)   # acc_sum_square_(a, alpha, b, beta)


	# adam mini
	self.grad.acc_to(&self.m, self.beta1, 1.0 - self.beta1);
	self.grad.mean_square(keepdim: true).acc_to(&self.v, self.beta2, 1.0 - self.beta2);
	self.v.one_over_sqrt().save_to(&self.v_recip);
	self.m.vec_mul(&self.v_recip).acc_to(&self.value, 1.0, -self.learning_rate);

	# rms norm
	val.mean_square(keepdim: true).save_to(&scale);
	scale.one_over_sqrt().save_to(&scale_recip);

*/

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
	fn grad(&self) -> &Tensor {
		&self.grad
	}

	fn zero_grad(&mut self) {
		zeros_(&self.grad);
	}

	fn step(&mut self, learning_rate: f64) {
		// update momentum
		acc_(&self.m, self.beta1, &self.grad, 1.0 - self.beta1);

		// update velocity
		acc_mean_(&self.v, self.beta2, &square(&self.grad), true, 1.0 - self.beta2);
		rsqrt_into(&self.v, self.eps, &self.v_recip);

		acc_mul_(&self.value, 1.0, &self.m, &self.v_recip, -learning_rate);
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
		&mut self, dtype: DType, parts: usize, part_elems: usize,
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
	fn grad(&self) -> &Tensor;
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
	pub nhead: Option<usize>,
	pub parts: usize,
	pub w: Tensor,
	pub w_opt: Rc<RefCell<dyn OptParam>>,
	pub scale: f64,
	pub backward_scale: f64,
	pub dtype: DType,
}

impl Linear {
	pub fn new(
		inputs: usize, outputs: usize, nhead: Option<usize>, dtype: DType, ctx: &mut Context,
	) -> Linear {
		let parts = nhead.unwrap_or(1);
		let w_opt = ctx.add_param(dtype, parts, outputs * inputs);
		let w = w_opt.borrow().value().clone();
		let w = w.reshape(&[parts * outputs, inputs]);
		let scale = MatMul::normalizing_scale(inputs);
		let backward_scale = MatMul::normalizing_scale(outputs);

		#[rustfmt::skip]
		Linear {
			inputs, outputs, nhead, parts,
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

		if let Some(nhead) = self.nhead {
			// [..., head * outputs] -> [..., head, outputs]
			y.reshape_last_n(1, &[nhead, self.outputs])
		} else {
			y
		}
	}

	fn backward(&self, mut dy: Tensor) -> Tensor {
		let mut w_opt = self.w_opt.borrow_mut();
		let x = w_opt.pop_tensor();

		if let Some(nhead) = self.nhead {
			// [..., head, outputs] -> [..., head * outputs]
			dy = dy.reshape_last_n(2, &[nhead * self.outputs])
		};
		let grad = mm(&self.w, x.as_col()).backward(&dy);

		// dw
		let dw = grad.da();
		let dw = dw.scale(1.0).assert_normalizing_scale();
		dw.acc_into(&w_opt.grad().clone().reshape(&[self.parts * self.outputs, self.inputs]));

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
	let eps = 1e-8;
	stderrlog::new().module(module_path!()).init().unwrap();

	let dev = CPUDevice::new("CPU".to_string());

	let x = Tensor::new_empty_on(&[2, 3], DType::f32(), dev.clone());
	let y = Tensor::new_empty_on(&[3, 2], DType::f32(), dev.clone());

	randn_(&x);
	randn_(&y);

	println!("x = {}", x);
	println!("y = {}", y);

	let nx = rms_norm(&x, eps);
	let ny = rms_norm(&y, eps);

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
