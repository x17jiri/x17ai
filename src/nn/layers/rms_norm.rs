//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::autograd::{Autograd, AutogradNode, BackwardFn, StraightThroughBackwardFn};
use crate::nn::param::Param;
use crate::tensor::math::{RSqrt, Sum};
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

use super::Layer;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RMSNormGradientMode {
	Precise,
	NormGradients,
	StraightThrough,
}

#[derive(Clone, Copy)]
struct CalcScale {
	sum_to_mean: f64,
	eps: f64,
}

impl CalcScale {
	pub fn calc(&self, inp: &Tensor) -> Result<Tensor, ErrPack<TensorOpError>> {
		let scale = inp.new_replace_tail(1, &[1])?;
		let sum_square = (inp * inp).sum();
		let mean_square = sum_square * self.sum_to_mean;
		scale.assign(mean_square.rsqrt(self.eps))?;
		Ok(scale)
	}
}

pub struct RMSNorm {
	shape: [usize; 1],
	calc_scale: CalcScale,
	gradient_mode: RMSNormGradientMode,
}

impl RMSNorm {
	pub fn new(n_inputs: usize, eps: f64) -> Self {
		Self {
			shape: [n_inputs],
			calc_scale: CalcScale {
				sum_to_mean: 1.0 / n_inputs.lossy_into(),
				eps,
			},
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

	fn forward(&self, inp_node: AutogradNode) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		let (inp, inp_backward) = inp_node.take();
		let scale = self.calc_scale.calc(&inp)?;

		let out = inp.reuse_or_new_like()?;
		out.assign(&inp * &scale)?;

		let backward_fn = inp_backward.map(|inp_backward| match self.gradient_mode {
			RMSNormGradientMode::Precise => {
				//
				Box::new(RMSNormBackwardFn_Precise {
					out: out.clone(),
					scale,
					sum_to_mean: self.calc_scale.sum_to_mean,
					inp_backward,
				}) as Box<dyn BackwardFn>
			},
			RMSNormGradientMode::NormGradients => {
				//
				Box::new(RMSNormBackwardFn_NormGradients {
					calc_scale: self.calc_scale,
					inp_backward,
				}) as Box<dyn BackwardFn>
			},
			RMSNormGradientMode::StraightThrough => {
				Box::new(StraightThroughBackwardFn::new(inp_backward)) as Box<dyn BackwardFn>
			},
		});

		Ok(AutogradNode::new(out, backward_fn))
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		// no parameters to randomize
		Ok(())
	}
}

pub struct RMSNormBackwardFn_Precise {
	out: Tensor,
	scale: Tensor,
	sum_to_mean: f64,
	inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for RMSNormBackwardFn_Precise {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { out, scale, sum_to_mean, inp_backward } = Box::into_inner(self);

		let g = scale.new_empty_like()?; // [..., 1]
		g.assign((&out * &d_out).sum() * sum_to_mean)?;

		let d_inp = out.reuse_or_new_like()?;

		// TODO - could we merge `mul, sub, mul` into a single kernel?
		d_inp.assign(&out * &g)?;
		d_inp.assign(&d_out - &d_inp)?;
		d_inp.assign(&d_inp * &scale)?;

		autograd.set_grad(inp_backward, d_inp);
		Ok(())
	}
}

pub struct RMSNormBackwardFn_NormGradients {
	calc_scale: CalcScale,
	inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for RMSNormBackwardFn_NormGradients {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { calc_scale, inp_backward } = Box::into_inner(self);

		let scale = calc_scale.calc(&d_out)?;

		let d_inp = d_out.reuse_or_new_like()?;
		d_inp.assign(&d_out * &scale)?;

		autograd.set_grad(inp_backward, d_inp);
		Ok(())
	}
}
