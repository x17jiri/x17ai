//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::autograd::{self, AutogradNode, BackwardFn, LossFn};
use crate::nn::layers::Layer;
use crate::nn::layers::softmax::{Softmax, SoftmaxGradientMode};
use crate::tensor::device::kernel::expr::TensorOps;
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

pub struct CrossEntropy {
	softmax: Softmax,
}

impl CrossEntropy {
	pub fn new(n_inputs: usize) -> Self {
		let mut softmax = Softmax::new(n_inputs);
		softmax.set_gradient_mode(SoftmaxGradientMode::StraightThrough);
		Self { softmax }
	}

	pub fn forward_with_target(
		&self,
		inp_node: AutogradNode,
		target: Tensor,
	) -> Result<Box<dyn LossFn>, ErrPack<TensorOpError>> {
		let out_node = self.softmax.forward(inp_node)?;
		let (value, inp_backward) = out_node.take();
		Ok(Box::new(CrossEntropyLossFn { value, target, inp_backward }))
	}
}

pub struct CrossEntropyLossFn {
	pub value: Tensor,
	pub target: Tensor,
	pub inp_backward: Option<Box<dyn BackwardFn>>,
}

impl LossFn for CrossEntropyLossFn {
	fn value(&self) -> Tensor {
		self.value.clone()
	}

	fn target(&self) -> Tensor {
		self.target.clone()
	}

	fn loss(&self) -> Result<Tensor, ErrPack<TensorOpError>> {
		let value = &self.value;
		let target = &self.target;

		let err_sums = value.new_replace_tail(1, &[1])?;
		err_sums.assign((target * value.ln_clamped()).sum())?;

		let err_sums = err_sums.merge_all_dims()?;

		let result = err_sums.new_empty(&[1], value.dtype())?;
		result.assign(err_sums.mean())?;

		Ok(result)
	}

	fn backward(self: Box<Self>) -> Result<(), ErrPack<TensorOpError>> {
		let Self { value, target, inp_backward } = Box::into_inner(self);
		let d_inp = value.new_empty_like()?;
		d_inp.assign(&value - &target)?;
		autograd::run(inp_backward, d_inp)?;
		Ok(())
	}
}
