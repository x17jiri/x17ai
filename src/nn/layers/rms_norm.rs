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
	StraightThrough,
}

pub struct RMSNorm {
	shape: [usize; 1],
	gradient_mode: RMSNormGradientMode,
	calc: RMSCalc,
}

#[derive(Clone, Copy)]
pub struct RMSCalc {
	pub sum_to_mean: f64,
	pub eps: f64,
}

impl RMSCalc {
	pub fn new(n_inputs: usize, eps: f64) -> Self {
		Self {
			sum_to_mean: 1.0 / n_inputs.lossy_into(),
			eps,
		}
	}

	/// Calculates:
	///
	///     1.0 / sqrt(mean(inp * inp) + eps)
	///
	/// where `mean` is calculated over the last dimension of `inp`.
	pub fn rsqrt_mean_square(&self, inp: &Tensor) -> Result<Tensor, ErrPack<TensorOpError>> {
		let result = inp.new_replace_tail(1, &[1])?;
		let sum_square = (inp * inp).sum();
		let mean_square = sum_square * self.sum_to_mean;
		result.assign(mean_square.rsqrt(self.eps))?;
		Ok(result)
	}

	/// Calculates:
	///
	///     mean(inp * inp)
	///
	/// where `mean` is calculated over the last dimension of `inp`.
	pub fn sqrt_mean_square(&self, inp: &Tensor) -> Result<Tensor, ErrPack<TensorOpError>> {
		let result = inp.new_replace_tail(1, &[1])?;
		let sum_square = (inp * inp).sum();
		let mean_square = sum_square * self.sum_to_mean;
		result.assign(mean_square.sqrt())?;
		Ok(result)
	}
}

impl RMSNorm {
	pub fn new(n_inputs: usize, eps: f64) -> Self {
		Self {
			shape: [n_inputs],
			calc: RMSCalc::new(n_inputs, eps),
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
		let scale = self.calc.rsqrt_mean_square(&inp)?;

		let out = inp.reuse_or_new_like()?;
		out.assign(&inp * &scale)?;

		let backward_fn = inp_backward.map(|inp_backward| match self.gradient_mode {
			RMSNormGradientMode::Precise => {
				//
				Box::new(RMSNormBackwardFn_Precise {
					out: out.clone(),
					scale,
					sum_to_mean: self.calc.sum_to_mean,
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
	fn run(
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

/// This is no-op in the forward pass and normalizes the gradients in the backward pass.
pub struct BackwardRMSNorm {
	shape: [usize; 1],
	sum_to_mean: f64,
	eps: f64,
}

impl BackwardRMSNorm {
	pub fn new(n_inputs: usize, eps: f64) -> Self {
		let rms_norm = RMSNorm::new(n_inputs, eps);
		Self {
			shape: rms_norm.shape,
			sum_to_mean: rms_norm.calc.sum_to_mean,
			eps: rms_norm.calc.eps,
		}
	}
}

impl Layer for BackwardRMSNorm {
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

		let backward_fn = inp_backward.map(|inp_backward| {
			Box::new(BackwardRMSNormBackwardFn {
				sum_to_mean: self.sum_to_mean,
				eps: self.eps,
				inp_backward,
			}) as Box<dyn BackwardFn>
		});

		Ok(AutogradNode::new(inp, backward_fn))
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		// no parameters to randomize
		Ok(())
	}
}

pub struct BackwardRMSNormBackwardFn {
	sum_to_mean: f64,
	eps: f64,
	inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for BackwardRMSNormBackwardFn {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { sum_to_mean, eps, inp_backward } = Box::into_inner(self);
		let rms_norm = RMSNorm {
			shape: [0], // shape is not important; it will not be used
			calc: RMSCalc { sum_to_mean, eps },
			gradient_mode: RMSNormGradientMode::Precise,
		};
		let d_inp_node = rms_norm.forward(AutogradNode::new(d_out, None))?;
		let (d_inp, _) = d_inp_node.take();
		autograd.set_grad(inp_backward, d_inp);
		Ok(())
	}
}
