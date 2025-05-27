// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::cell::RefCell;
use std::rc::Rc;

use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::math::Savable;
use crate::tensor::{self, Tensor};

use super::{Layer, LossFunction, Softmax, SoftmaxGradientMode};

pub struct SoftmaxCrossEntropy {
	softmax: Softmax,
}

impl SoftmaxCrossEntropy {
	pub fn new(n_inputs: usize) -> SoftmaxCrossEntropy {
		let mut softmax = Softmax::new(n_inputs);
		softmax.set_gradient_mode(SoftmaxGradientMode::StraightThrough);
		SoftmaxCrossEntropy { softmax }
	}
}

impl Layer for SoftmaxCrossEntropy {
	fn input_shape(&self) -> &[usize] {
		self.softmax.input_shape()
	}

	fn output_shape(&self) -> &[usize] {
		self.softmax.output_shape()
	}

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.softmax.collect_params(f);
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.softmax.collect_named_params(prefix, f);
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		self.softmax.forward(inp, ctx)
	}

	fn randomize(&mut self) {
		self.softmax.randomize();
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		self.softmax.backward(d_out, ctx)
	}

	fn backward_finish(&self, d_out: Tensor, ctx: &mut EvalContext) {
		self.softmax.backward_finish(d_out, ctx)
	}

	fn as_loss_function(&self) -> Option<&dyn LossFunction> {
		Some(self)
	}
}

impl LossFunction for SoftmaxCrossEntropy {
	fn backward_start(&self, out: Tensor, expected_out: Tensor, _ctx: &mut EvalContext) -> Tensor {
		let d_inp = out.new_empty_like();
		tensor::math::sub(&out, &expected_out).save_to(&d_inp);
		d_inp
	}

	fn loss(&self, out: Tensor, expected_out: Tensor) -> f64 {
		let tmp = out.new_empty_like();
		tensor::math::log_clamped(&out).save_to(&tmp);
		tensor::math::mul(&tmp, &expected_out).save_to(&tmp);

		tensor::math::sum_all(&tmp) / -(tmp.batch_size(1) as f64)
	}
}
