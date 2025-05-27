// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::cell::RefCell;
use std::rc::Rc;

use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::math::Savable;
use crate::tensor::{self, Tensor};

use super::Layer;

pub struct SwiGLU {
	input_shape: [usize; 2],
	output_shape: [usize; 1],
}

impl SwiGLU {
	pub fn new(n_outputs: usize) -> SwiGLU {
		SwiGLU {
			input_shape: [2, n_outputs],
			output_shape: [n_outputs],
		}
	}
}

impl Layer for SwiGLU {
	fn input_shape(&self) -> &[usize] {
		&self.input_shape
	}

	fn output_shape(&self) -> &[usize] {
		&self.output_shape
	}

	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		let out = inp.new_replace_tail(2, &self.output_shape);
		let lin = inp.clone().slice(-2, 0..1).merge_dims::<2>();
		let gate = inp.slice(-2, 1..2).merge_dims::<2>();

		tensor::math::swiglu(&lin, &gate).save_to(&out);

		if ctx.is_training() {
			ctx.tensors.set([lin, gate]);
		}

		out
	}

	fn randomize(&mut self) {
		// no parameters to randomize
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		let [lin, gate] = ctx.tensors.get();

		let d_inp = d_out.new_replace_tail(1, &self.input_shape);
		let d_lin = d_inp.clone().slice(-2, 0..1).merge_dims::<2>();
		let d_gate = d_inp.clone().slice(-2, 1..2).merge_dims::<2>();

		tensor::math::swiglu_backward(&d_out, &lin, &gate).save_to(&d_lin, &d_gate);

		d_inp
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}
