//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::Result;
use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::math::{RSqrt, Sum};
use crate::util::LossyInto;

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
	pub fn new(n_inputs: usize, eps: f64) -> Self {
		Self {
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

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Result<Tensor> {
		let scale = inp.new_replace_tail(1, &[1]);
		scale.assign((&inp * &inp).sum().rsqrt(self.eps))?;

		let out = inp.reuse_or_new_like();
		out.assign(&inp * &scale)?;

		if ctx.is_training() && self.gradient_mode == RMSNormGradientMode::Precise {
			ctx.tensors.set([out.clone(), scale]);
		}

		Ok(out)
	}

	fn randomize(&mut self) -> Result<()> {
		// no parameters to randomize
		Ok(())
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Result<Tensor> {
		match self.gradient_mode {
			RMSNormGradientMode::Precise => {
				let [out, scale] = ctx.tensors.get();

				let g = scale.new_empty_like(); // [..., 1]
				g.assign((&out * &d_out).sum() * (1.0 / self.shape[0].lossy_into()))?;

				let d_inp = d_out.reuse_or_new_like();

				// TODO - could we merge `mul, sub, mul` into a single kernel?
				d_inp.assign(&d_out - (&out * &g))?;
				d_inp.assign(&d_inp * &scale)?;

				Ok(d_inp)
			},
			RMSNormGradientMode::NormGradients => {
				let scale = d_out.new_replace_tail(1, &[1]);
				scale.assign((&d_out * &d_out).sum().rsqrt(self.eps))?;

				let d_inp = d_out.reuse_or_new_like();
				d_inp.assign(&d_out * &scale)?;
				Ok(d_inp)
			},
			RMSNormGradientMode::StraightThrough => Ok(d_out),
		}
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) -> Result<()> {
		// no parameters to update
		Ok(())
	}
}
