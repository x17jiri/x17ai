//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::autograd::{self, AutogradTensor, BackwardFn, LossFn};
use crate::nn::fragments::UnaryFragment;
use crate::nn::fragments::softmax::{Softmax, SoftmaxGradMode};
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::{DType, Tensor, TensorOpError};
use crate::{ErrPack, custom_kernel};

pub struct CrossEntropy {
	softmax: Softmax,
}

impl CrossEntropy {
	pub fn new(internal_dtype: DType) -> Self {
		Self {
			softmax: Softmax::new(internal_dtype, SoftmaxGradMode::StraightThrough),
		}
	}

	pub fn forward_with_target(
		&self,
		inp_node: AutogradTensor,
		target: Tensor,
	) -> Result<Box<dyn LossFn>, ErrPack<TensorOpError>> {
		let out_node = self.softmax.forward(inp_node)?;
		let (value, inp_backward) = out_node.into_parts();
		Ok(Box::new(CrossEntropyLossFn {
			value,
			target,
			internal_dtype: self.softmax.internal_dtype(),
			inp_backward,
		}))
	}
}

pub struct CrossEntropyLossFn {
	pub value: Tensor,
	pub target: Tensor,
	pub internal_dtype: DType,
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

		let internal_dtype =
			common_dtype(common_dtype(value.dtype(), target.dtype()), self.internal_dtype);
		let err_sums = value.new_replace_tail(1, &[1], internal_dtype)?;
		err_sums.assign(custom_kernel!(
			internal_dtype,
			[target: &target, value: &value], (), {
				(target * value.ln()).sum()
			}
		))?;

		// We have a sum for each batch. Let's merge batch dimensions
		let err_sums = err_sums.merge_all_dims()?;

		let result = err_sums.new_empty(&[1], value.dtype())?;
		result.assign(custom_kernel!(
			internal_dtype,
			[err_sums: &err_sums], (sum_to_mean: err_sums.sum_to_mean()), {
				err_sums.sum() * sum_to_mean
			}
		))?;

		Ok(result)
	}

	fn backward(self: Box<Self>, grad_dtype: DType) -> Result<(), ErrPack<TensorOpError>> {
		let Self {
			value,
			target,
			internal_dtype,
			inp_backward,
		} = Box::into_inner(self);
		let d_inp = value.new_empty_like(grad_dtype)?;
		d_inp.assign(custom_kernel!(
			internal_dtype,
			[value: &value, target: &target], (), {
				value - target
			}
		))?;
		autograd::run(inp_backward, d_inp)?;
		Ok(())
	}
}
