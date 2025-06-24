//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::hint::cold_path;
use std::rc::Rc;

use crate::autograd::{AutogradCtx, BackwardFn, StraightThroughBackwardFn};
use crate::nn::eval_context::EvalContext;
use crate::nn::optimizer::OptimizerError;
use crate::nn::param::Param;
use crate::tensor::math::Sum;
use crate::tensor::{Tensor, TensorOpError, math};
use crate::{ErrPack, autograd};

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
	pub fn new(n_inputs: usize) -> Self {
		Self {
			shape: [n_inputs],
			gradient_mode: SoftmaxGradientMode::Precise,
		}
	}

	pub fn set_gradient_mode(&mut self, mode: SoftmaxGradientMode) {
		self.gradient_mode = mode;
	}

	pub fn forward2(
		&self,
		mut inp_node: Box<autograd::Node>,
	) -> Result<Box<autograd::Node>, ErrPack<TensorOpError>> {
		let Some(inp) = inp_node.take_value() else {
			cold_path();
			return Err(TensorOpError::missing_value());
		};
		let out = inp.reuse_or_new_like()?;

		out.assign(math::softmax(&inp))?;

		let backward_fn: Option<Box<dyn BackwardFn>> = if inp_node.requires_grad() {
			match self.gradient_mode {
				SoftmaxGradientMode::Precise => {
					Some(Box::new(SoftmaxBackwardFn { out: out.clone(), inp_node }))
				},
				SoftmaxGradientMode::StraightThrough => {
					Some(Box::new(StraightThroughBackwardFn::new(inp_node)))
				},
			}
		} else {
			None
		};

		Ok(autograd::Node::new(out, backward_fn))
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

	fn forward(
		&self,
		inp: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let out = inp.reuse_or_new_like()?;

		out.assign(math::softmax(&inp))?;

		if ctx.is_training() {
			match self.gradient_mode {
				SoftmaxGradientMode::Precise => ctx.tensors.set([out.clone()]),
				SoftmaxGradientMode::StraightThrough => {},
			}
		}

		Ok(out)
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		// no parameters to randomize
		Ok(())
	}

	fn backward(
		&self,
		d_out: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<OptimizerError>> {
		match self.gradient_mode {
			SoftmaxGradientMode::Precise => {
				let [out] = ctx.tensors.get();

				let g = out.new_replace_tail(1, &[1])?; // [..., 1]
				g.assign((&out * &d_out).sum())?;

				let d_inp = d_out.reuse_or_new_like()?;

				// TODO - we could merge `-` and `*` into a single kernel
				d_inp.assign(&d_out - &g)?;
				d_inp.assign(&d_inp * &out)?;

				Ok(d_inp)
			},
			SoftmaxGradientMode::StraightThrough => Ok(d_out),
		}
	}
}

pub struct SoftmaxBackwardFn {
	pub out: Tensor,
	pub inp_node: Box<autograd::Node>,
}

impl BackwardFn for SoftmaxBackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		ctx: &mut AutogradCtx,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { out, inp_node } = Box::into_inner(self);

		let g = out.new_replace_tail(1, &[1])?; // [..., 1]
		g.assign((&out * &d_out).sum())?;

		let d_inp = d_out.reuse_or_new_like()?;

		// TODO - we could merge `-` and `*` into a single kernel
		d_inp.assign(&d_out - &g)?;
		d_inp.assign(&d_inp * &out)?;

		ctx.set_grad(inp_node, d_inp);
		Ok(())
	}
}
