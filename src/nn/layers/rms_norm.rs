// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::cell::RefCell;
use std::rc::Rc;

use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::math::Savable;
use crate::tensor::{self, Tensor, TensorSize};

use super::Layer;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RMSNormGradientMode {
	Precise,
	NormGradients,
	StraightThrough,
}

pub struct RMSNorm {
	shape: [TensorSize; 1],
	eps: f64,
	gradient_mode: RMSNormGradientMode,
}

impl RMSNorm {
	pub fn new(n_inputs: TensorSize, eps: f64) -> RMSNorm {
		RMSNorm {
			shape: [n_inputs],
			eps,
			gradient_mode: RMSNormGradientMode::Precise,
		}
	}
}

impl Layer for RMSNorm {
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

		if ctx.is_training() && self.gradient_mode == RMSNormGradientMode::Precise {
			let scale = out_ref.new_replace_tail(1, &[1]);

			tensor::math::rms_norm(&inp, self.eps).scale_storage(&scale).save_to(out_ref);

			ctx.tensors.set([out_ref.clone(), scale]);
		} else {
			tensor::math::rms_norm(&inp, self.eps).save_to(out_ref);
		}

		out.unwrap_or(inp)
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

				// try to reuse `out` for `d_inp` if possible
				let (d_inp, d_inp_ref);
				if out.owns_buffer() {
					d_inp = None;
					d_inp_ref = &out;
				} else {
					d_inp = Some(out.new_empty_like());
					d_inp_ref = d_inp.as_ref().unwrap();
				}

				// TODO - could we merge `mul, sub, mul` into a single kernel?
				tensor::math::mul(&out, &g).save_to(d_inp_ref);
				tensor::math::sub(&d_out, d_inp_ref).save_to(d_inp_ref);
				tensor::math::mul(d_inp_ref, &scale).save_to(d_inp_ref);

				d_inp.unwrap_or(out)
			},
			RMSNormGradientMode::NormGradients => {
				// try to reuse `d_out` for `d_inp` if possible
				let (d_inp, d_inp_ref);
				if d_out.owns_buffer() {
					d_inp = None;
					d_inp_ref = &d_out;
				} else {
					d_inp = Some(d_out.new_empty_like());
					d_inp_ref = d_inp.as_ref().unwrap();
				}

				tensor::math::rms_norm(&d_out, self.eps).save_to(d_inp_ref);

				d_inp.unwrap_or(d_out)
			},
			RMSNormGradientMode::StraightThrough => d_out,
		}
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}
