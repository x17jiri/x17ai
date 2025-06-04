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
use crate::tensor::math::Savable;
use crate::tensor::{self, Tensor};

use super::Layer;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RMSNormGradientMode {
	Precise,
	NormGradients,
	StraightThrough,
}

pub struct RMSNorm {
	shape: [usize; 1],
	eps: f64,
	gradient_mode: RMSNormGradientMode,
}

impl RMSNorm {
	pub fn new(n_inputs: usize, eps: f64) -> RMSNorm {
		RMSNorm {
			shape: [n_inputs],
			eps,
			gradient_mode: RMSNormGradientMode::Precise,
		}
	}
}

impl Layer for RMSNorm {
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

		if ctx.is_training() && self.gradient_mode == RMSNormGradientMode::Precise {
			let scale = out.new_replace_tail(1, &[1]);

			tensor::math::rms_norm(&inp, self.eps).scale_storage(&scale).save_to(&out);

			ctx.tensors.set([out.clone(), scale]);
		} else {
			tensor::math::rms_norm(&inp, self.eps).save_to(&out);
		}

		out
	}

	fn randomize(&mut self) {
		// no parameters to randomize
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		match self.gradient_mode {
			RMSNormGradientMode::Precise => {
				let [out, scale] = ctx.tensors.get();

				let g = scale.new_empty_like(); // [..., 1]
				tensor::math::dot(&out, &d_out).scale(1.0 / self.shape[0] as f64).save_to(&g);

				let d_inp = out.reuse_or_new_like();

				// TODO - could we merge `mul, sub, mul` into a single kernel?
				tensor::math::mul(&out, &g).save_to(&d_inp);
				tensor::math::sub(&d_out, &d_inp).save_to(&d_inp);
				tensor::math::mul(&d_inp, &scale).save_to(&d_inp);

				d_inp
			},
			RMSNormGradientMode::NormGradients => {
				let d_inp = d_out.reuse_or_new_like();
				tensor::math::rms_norm(&d_out, self.eps).save_to(&d_inp);
				d_inp
			},
			RMSNormGradientMode::StraightThrough => d_out,
		}
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}
