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
use crate::tensor::math::Sum;
use crate::tensor::{Tensor, TensorOpError, math};

use super::Layer;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
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

	fn forward(&self, inp_node: AutogradNode) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		let (inp, inp_backward) = inp_node.take();
		let out = inp.reuse_or_new_like()?;

		out.assign(math::softmax(&inp))?;

		#[allow(clippy::option_if_let_else)]
		let backward_fn = match inp_backward {
			Some(inp_backward) => match self.gradient_mode {
				SoftmaxGradientMode::Precise => {
					Some(Box::new(SoftmaxBackwardFn_Precise { out: out.clone(), inp_backward })
						as Box<dyn BackwardFn>)
				},
				SoftmaxGradientMode::StraightThrough => {
					Some(Box::new(StraightThroughBackwardFn::new(inp_backward))
						as Box<dyn BackwardFn>)
				},
			},
			None => None,
		};

		Ok(AutogradNode::new(out, backward_fn))
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		// no parameters to randomize
		Ok(())
	}
}

pub struct SoftmaxBackwardFn_Precise {
	pub out: Tensor,
	pub inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for SoftmaxBackwardFn_Precise {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { out, inp_backward } = Box::into_inner(self);

		let g = out.new_replace_tail(1, &[1])?; // [..., 1]
		g.assign((&out * &d_out).sum())?;

		let d_inp = d_out.reuse_or_new_like()?;

		// TODO - we could merge `-` and `*` into a single kernel
		d_inp.assign(&d_out - &g)?;
		d_inp.assign(&d_inp * &out)?;

		autograd.set_grad(inp_backward, d_inp);
		Ok(())
	}
}
