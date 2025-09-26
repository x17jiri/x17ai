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
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::{DType, Tensor, TensorOpError};
use crate::{ErrPack, custom_kernel};

use super::Fragment;

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SoftmaxGradMode {
	Precise,
	Simplified,
	StraightThrough,
}

pub struct Softmax {
	internal_dtype: DType,
	grad_mode: SoftmaxGradMode,
}

impl Softmax {
	pub fn new(internal_dtype: DType, grad_mode: SoftmaxGradMode) -> Self {
		Self { internal_dtype, grad_mode }
	}
}

impl Fragment for Softmax {
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

impl UnaryFragment for Softmax {
	fn forward(&self, inp_node: AutogradTensor) -> Result<AutogradTensor, ErrPack<TensorOpError>> {
		let (inp, inp_backward) = inp_node.into_parts();
		let out = inp.reuse_or_new_like()?;
		let internal_dtype = common_dtype(inp.dtype(), self.internal_dtype)?;

		let max = inp.new_replace_tail(1, &[1], inp.dtype())?; // [..., 1]
		let sum_recip = max.new_empty_like(internal_dtype)?;

		max.assign(custom_kernel!(
			[inp: &inp], (), {
				inp.max()
			}
		))?;
		sum_recip.assign(custom_kernel!(
			[inp: &inp, max: &max], (), {
				(inp - max).exp().sum().recip()
			}
		))?;
		out.assign(custom_kernel!(
			[inp: &inp, max: &max, sum_recip: &sum_recip], (), {
				(inp - max).exp() * sum_recip
			}
		))?;

		let backward_fn = inp_backward.map(|inp_backward| match self.grad_mode {
			SoftmaxGradMode::Precise => Box::new(SoftmaxBackwardFn_Precise {
				out: out.clone(),
				internal_dtype: self.internal_dtype,
				inp_backward,
			}) as Box<dyn BackwardFn>,
			SoftmaxGradMode::Simplified => {
				Box::new(SoftmaxBackwardFn_Simplified { out: out.clone(), inp_backward })
					as Box<dyn BackwardFn>
			},
			SoftmaxGradMode::StraightThrough => {
				// TODO - could I just use inp_backward directly?
				Box::new(StraightThroughBackwardFn::new(inp_backward)) as Box<dyn BackwardFn>
			},
		});

		Ok(AutogradTensor::new(out, backward_fn))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SoftmaxBackwardFn_Precise {
	pub out: Tensor,
	pub internal_dtype: DType,
	pub inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for SoftmaxBackwardFn_Precise {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { out, internal_dtype, inp_backward } = Box::into_inner(self);
		let internal_dtype = common_dtype(d_out.dtype(), internal_dtype)?;

		let g = out.new_replace_tail(1, &[1], internal_dtype)?; // [..., 1]
		g.assign(custom_kernel!(
			[out: &out, d_out: &d_out], (), {
				(out * d_out).sum()
			}
		))?;

		let d_inp = d_out.reuse_or_new_like()?;

		d_inp.assign(custom_kernel!(
			[d_out: &d_out, g: &g, out: &out], (), {
				(d_out - g) * out
			}
		))?;

		queue.add(inp_backward, d_inp);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SoftmaxBackwardFn_Simplified {
	pub out: Tensor,
	pub inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for SoftmaxBackwardFn_Simplified {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { out, inp_backward } = Box::into_inner(self);

		let d_inp = d_out.reuse_or_new_like()?;

		d_inp.assign(custom_kernel!(
			[d_out: &d_out, out: &out], (), {
				d_out * out
			}
		))?;

		queue.add(inp_backward, d_inp);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
