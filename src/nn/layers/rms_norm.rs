//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::autograd::{self, AutogradNode, BackwardFn, StraightThroughBackwardFn};
use crate::nn::param::Param;
use crate::tensor::device::kernel::lookup::tsr;
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

use super::Layer;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RMSNormGradientMode {
	Precise,
	StraightThrough,
}

pub struct RMSNorm {
	shape: [usize; 1],
	gradient_mode: RMSNormGradientMode,
	eps: f64,
}

impl RMSNorm {
	pub fn new(n_inputs: usize, eps: f64) -> Self {
		Self {
			shape: [n_inputs],
			gradient_mode: RMSNormGradientMode::Precise,
			eps,
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
		let kernels = inp.builtin_kernel_library();
		let magn_recip = inp.new_replace_tail(1, &[1])?;
		magn_recip.assign(kernels.rms_recip(&inp, self.eps))?;

		let out = inp.reuse_or_new_like()?;
		out.assign2(tsr(&inp) * tsr(&magn_recip))?;

		let backward_fn = inp_backward.map(|inp_backward| match self.gradient_mode {
			RMSNormGradientMode::Precise => {
				//
				Box::new(RMSNormBackwardFn_Precise {
					out: out.clone(),
					magn_recip,
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

		let n = out.size(-1).unwrap_or(1);
		let sum_to_mean = 1.0 / n.lossy_into();

		let g = magn_recip.new_empty_like()?; // [..., 1]
		g.assign2((tsr(&out) * &d_out).sum() * sum_to_mean)?;

		let d_inp = out.reuse_or_new_like()?;

		d_inp.assign2((tsr(&d_out) - (tsr(&out) * &g)) * &magn_recip)?;

		queue.add(inp_backward, d_inp);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
