//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::{self, AutogradTensor, BackwardFn, StraightThroughBackwardFn};
use crate::nn::fragments::UnaryFragment;
use crate::nn::param::Param;
use crate::rng::Rng;
use crate::tensor::{Tensor, TensorOpError};
use crate::{ErrPack, custom_kernel};

use super::Fragment;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RMSNormGradMode {
	Precise,
	StraightThrough,
}

pub struct RMSNorm {
	grad_mode: RMSNormGradMode,
	eps: f64,
}

impl RMSNorm {
	pub fn new(eps: f64, grad_mode: RMSNormGradMode) -> Self {
		Self { grad_mode, eps }
	}
}

impl UnaryFragment for RMSNorm {
	fn forward(&self, inp: AutogradTensor) -> Result<AutogradTensor, ErrPack<TensorOpError>> {
		let (inp, inp_backward) = inp.into_parts();
		let sum_to_mean = inp.sum_to_mean();

		let magn_recip = inp.new_replace_tail(1, &[1])?;
		magn_recip.assign(custom_kernel!(
			[inp: &inp], (sum_to_mean: sum_to_mean, eps: self.eps), {
				(((inp * inp).sum() * sum_to_mean).sqrt() + eps).recip()
			}
		))?;

		let out = inp.reuse_or_new_like()?;
		out.assign(custom_kernel!(
			[inp: &inp, magn_recip: &magn_recip], (), {
				inp * magn_recip
			}
		))?;

		let backward_fn = inp_backward.map(|inp_backward| match self.grad_mode {
			RMSNormGradMode::Precise => {
				//
				Box::new(RMSNormBackwardFn_Precise {
					out: out.clone(),
					magn_recip,
					inp_backward,
				}) as Box<dyn BackwardFn>
			},
			RMSNormGradMode::StraightThrough => {
				Box::new(StraightThroughBackwardFn::new(inp_backward)) as Box<dyn BackwardFn>
			},
		});

		Ok(AutogradTensor::new(out, backward_fn))
	}
}

impl Fragment for RMSNorm {
	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn randomize(&mut self, _rng: &mut Rng) -> Result<(), ErrPack<TensorOpError>> {
		// no parameters to randomize
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RMSNormBackwardFn_Precise {
	out: Tensor,
	magn_recip: Tensor,
	inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for RMSNormBackwardFn_Precise {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { out, magn_recip, inp_backward } = Box::into_inner(self);
		let sum_to_mean = out.sum_to_mean();

		let g = magn_recip.new_empty_like()?; // [..., 1]
		g.assign(custom_kernel!(
			[out: &out, d_out: &d_out], (sum_to_mean: sum_to_mean), {
				(out * d_out).sum() * sum_to_mean
			}
		))?;

		let d_inp = out.reuse_or_new_like()?;

		d_inp.assign(custom_kernel!(
			[d_out: &d_out, out: &out, g: &g, magn_recip: &magn_recip], (), {
				(d_out - (out * g)) * magn_recip
			}
		))?;

		queue.add(inp_backward, d_inp);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
