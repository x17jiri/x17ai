//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::math::dot;
use crate::tensor::{self, Tensor};

use super::Layer;

pub enum SoftmaxGradientMode {
	Precise,
	StraightThrough,
}

pub struct Softmax {
	shape: [usize; 1],
	gradient_mode: SoftmaxGradientMode,
}

impl Softmax {
	pub fn new(n_inputs: usize) -> Softmax {
		Softmax {
			shape: [n_inputs],
			gradient_mode: SoftmaxGradientMode::Precise,
		}
	}

	pub fn set_gradient_mode(&mut self, mode: SoftmaxGradientMode) {
		self.gradient_mode = mode;
	}
}

impl Layer for Softmax {
	fn input_shape(&self) -> &[usize] {
		&self.shape
	}

	fn output_shape(&self) -> &[usize] {
		&self.shape
	}

	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		let out = inp.reuse_or_new_like();

		tensor::math::softmax(&inp).save_to(&out);

		if ctx.is_training() {
			match self.gradient_mode {
				SoftmaxGradientMode::Precise => ctx.tensors.set([out.clone()]),
				SoftmaxGradientMode::StraightThrough => {},
			}
		}

		out
	}

	fn randomize(&mut self) {
		// no parameters to randomize
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		match self.gradient_mode {
			SoftmaxGradientMode::Precise => {
				let [out] = ctx.tensors.get();

				let g = out.new_replace_tail(1, &[1]); // [..., 1]
				g.assign(dot(&out, &d_out));

				let d_inp = d_out.reuse_or_new_like();

				// TODO - we could merge `-` and `*` into a single kernel
				d_inp.assign(&d_out - &g);
				d_inp.assign(&d_inp * &out);

				d_inp
			},
			SoftmaxGradientMode::StraightThrough => d_out,
		}
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}
