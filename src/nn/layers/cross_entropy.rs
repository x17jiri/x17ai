//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::autograd::{self, AutogradNode, BackwardFn, LossFn};
use crate::tensor::device::kernel::expr::TensorOps;
use crate::tensor::math::{self, sum_all};
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

pub struct CrossEntropy;

impl Default for CrossEntropy {
	fn default() -> Self {
		Self::new()
	}
}

impl CrossEntropy {
	pub fn new() -> Self {
		Self
	}

	pub fn forward_with_target(
		&self,
		inp_node: AutogradNode,
		target: Tensor,
	) -> Result<Box<dyn LossFn>, ErrPack<TensorOpError>> {
		let (inp, inp_backward) = inp_node.take();
		let value = inp.reuse_or_new_like()?;

		value.assign(math::softmax(&inp))?;

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

	fn loss(&self) -> Result<f64, ErrPack<TensorOpError>> {
		let value = &self.value;
		let target = &self.target;

		// Remove the feature dimension (-1). All other dimensions are batch dimensions,
		// so `elems()` will give us the batch size.
		let batch_size = value.select(-1, 0)?.elems();

		let tmp = value.new_empty_like()?;
		tmp.assign(target * value.ln_clamped())?;

		Ok(sum_all(&tmp)? / -batch_size.lossy_into())
	}

	fn backward(self: Box<Self>) -> Result<(), ErrPack<TensorOpError>> {
		let Self { value, target, inp_backward } = Box::into_inner(self);
		let d_inp = value.new_empty_like()?;
		d_inp.assign(&value - &target)?;
		autograd::run(inp_backward, d_inp)?;
		Ok(())
	}
}
