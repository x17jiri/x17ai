// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#![allow(non_snake_case)]
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

mod batch;
mod buffer;
mod context;
mod cpu;
mod device;
mod dim_merger;
mod dtype;
mod expr;
mod format;
mod matrix;
mod optimizer;
mod rand;
mod tensor;

use crate::batch::*;
use crate::buffer::*;
use crate::context::*;
use crate::cpu::*;
use crate::device::*;
use crate::dim_merger::*;
use crate::dtype::*;
use crate::expr::*;
use crate::format::*;
use crate::matrix::*;
use crate::optimizer::*;
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

/// Multihead Linear Layer transforming inputs to outputs
///
/// This is basically a thin wrapper around a matrix multiplication.
/// It does not include a bias term.
///
/// This works like n parallel linear transformations of the same input:
///
///     input: [*, inputs]
///     output: [*, head, outputs]
struct Linear {
	pub inputs: usize,
	pub outputs: usize,
	pub heads: usize,

	pub w: Tensor,
	pub w_opt: Rc<RefCell<OptParam>>,

	pub forward_scale: f64,
	pub backward_scale: f64,
	pub dtype: DType,
}

impl Linear {
	pub fn new(
		inputs: usize, outputs: usize, heads: usize, dtype: DType, ctx: &mut Context,
	) -> Linear {
		let w_opt = ctx.add_param(dtype, heads, outputs * inputs);
		let w = w_opt.borrow().value().clone();
		let w = w.reshape(&[heads * outputs, inputs]);

		let forward_scale = 1.0 / (inputs as f64).sqrt();
		let backward_scale = 1.0 / (outputs as f64).sqrt();

		#[rustfmt::skip]
		Linear {
			inputs, outputs, heads,
			w, w_opt,
			forward_scale, backward_scale, dtype,
		}
	}

	fn forward(&self, inp: Tensor, out: Tensor) {
		// [..., heads, outputs] -> [..., heads * outputs]
		let out = out.reshape(2, &[self.heads * self.outputs]);

		let w = matrix(&self.w);
		let i = col_matrix(&inp);
		let o = col_matrix(&out);
		mm(w, i).scale(self.forward_scale).save_to(o);

		self.w_opt.borrow_mut().save_tensors([inp]);
	}

	fn backward(&self, d_out: Tensor, d_inp: Tensor) {
		let [inp] = self.w_opt.borrow_mut().load_tensors();

		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.reshape(2, &[self.heads * self.outputs]);

		// [heads, outputs * inputs] -> [heads * outputs, inputs]
		let d_weights = self.w_opt.borrow().grad().clone();
		let d_weights = d_weights.reshape_all(&[self.heads * self.outputs, self.inputs]);

		let d_w = matrix(&d_weights);
		let i = col_matrix(&inp);
		let d_o = col_matrix(&d_out);
		mm(d_o, i.T()).scale(1.0).acc_to(d_w, 1.0, 1.0);

		let w = matrix(&self.w);
		let d_i = col_matrix(&d_inp);
		mm(w.T(), d_o).scale(self.backward_scale).save_to(d_i);
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
