// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::cell::RefCell;
use std::rc::Rc;

use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::math::Savable;
use crate::tensor::{self, Tensor, TensorSize};

use super::Layer;

pub enum SoftmaxGradientMode {
	Precise,
	StraightThrough,
}

pub struct Softmax {
	shape: [TensorSize; 1],
	gradient_mode: SoftmaxGradientMode,
}

impl Softmax {
	pub fn new(n_inputs: TensorSize) -> Softmax {
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
	fn input_shape(&self) -> &[TensorSize] {
		&self.shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.shape
	}

	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		// try to reuse `inp` for `out` if possible
		let (out, out_ref);
		if inp.owns_buffer() {
			out = None;
			out_ref = &inp;
		} else {
			out = Some(inp.new_empty_like());
			out_ref = out.as_ref().unwrap();
		}

		tensor::math::softmax(&inp).save_to(out_ref);

		if ctx.is_training() {
			match self.gradient_mode {
				SoftmaxGradientMode::Precise => ctx.tensors.set([out_ref.clone()]),
				SoftmaxGradientMode::StraightThrough => {},
			}
		}

		out.unwrap_or(inp)
	}

	fn randomize(&mut self) {
		// no parameters to randomize
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		match self.gradient_mode {
			SoftmaxGradientMode::Precise => {
				let [out] = ctx.tensors.get();

				let g = out.new_replace_tail(1, &[1]); // [..., 1]
				tensor::math::dot(&out, &d_out).save_to(&g);

				// try to reuse the `d_out` for `d_inp` if possible
				let (d_inp, d_inp_ref);
				if d_out.owns_buffer() {
					d_inp = None;
					d_inp_ref = &d_out;
				} else {
					d_inp = Some(d_out.new_empty_like());
					d_inp_ref = d_inp.as_ref().unwrap();
				}

				// TODO - we could merge `sub` and `mul` into a single kernel
				tensor::math::sub(&d_out, &g).save_to(d_inp_ref);
				tensor::math::mul(d_inp_ref, &out).save_to(d_inp_ref);

				d_inp.unwrap_or(d_out)
			},
			SoftmaxGradientMode::StraightThrough => d_out,
		}
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}
