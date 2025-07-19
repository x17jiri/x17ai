//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

use crate::ErrPack;
use crate::autograd::{self, AutogradNode, BackwardFn, StraightThroughBackwardFn};
use crate::nn::param::Param;
use crate::tensor::device::kernel::Kernel;
use crate::tensor::device::kernel_builder::KernelBuilder;
use crate::tensor::math::{EvaluatesToTensor, Sum};
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

use super::Layer;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct RMSKernel {
	kernel: &'static Kernel<0, 1, 1>,
}

impl RMSKernel {
	pub fn new() -> RMSKernel {
		static instance: OnceLock<Kernel<0, 1, 1>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [], [inp], [sum_to_mean]) =
				KernelBuilder::new("rms", [], ["inp"], ["sum_to_mean"]);
			builder.build(((inp.clone() * inp).sum() * sum_to_mean).sqrt())
		});
		RMSKernel { kernel }
	}

	pub fn call<'a>(self, inp: &'a Tensor, sum_to_mean: f64) -> RMSKernelCall<'a> {
		RMSKernelCall { rms_kernel: self, inp, sum_to_mean }
	}
}

pub struct RMSKernelCall<'a> {
	rms_kernel: RMSKernel,
	inp: &'a Tensor,
	sum_to_mean: f64,
}

impl<'a> EvaluatesToTensor for RMSKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.rms_kernel.kernel.run(to, [], [self.inp], [self.sum_to_mean])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct RMSRecipKernel {
	kernel: &'static Kernel<0, 1, 2>,
}

impl RMSRecipKernel {
	pub fn new() -> RMSRecipKernel {
		static instance: OnceLock<Kernel<0, 1, 2>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [], [inp], [sum_to_mean, eps]) =
				KernelBuilder::new("rms", [], ["inp"], ["sum_to_mean", "eps"]);
			builder.build(((inp.clone() * inp).sum() * sum_to_mean).sqrt().recip(eps))
		});
		RMSRecipKernel { kernel }
	}

	pub fn call<'a>(self, inp: &'a Tensor, sum_to_mean: f64, eps: f64) -> RMSRecipKernelCall<'a> {
		RMSRecipKernelCall {
			rms_recip_kernel: self,
			inp,
			sum_to_mean,
			eps,
		}
	}
}

pub struct RMSRecipKernelCall<'a> {
	rms_recip_kernel: RMSRecipKernel,
	inp: &'a Tensor,
	sum_to_mean: f64,
	eps: f64,
}

impl<'a> EvaluatesToTensor for RMSRecipKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.rms_recip_kernel.kernel.run(to, [], [self.inp], [self.sum_to_mean, self.eps])
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RMSNormGradientMode {
	Precise,
	StraightThrough,
}

pub struct RMSNorm {
	shape: [usize; 1],
	gradient_mode: RMSNormGradientMode,
	sum_to_mean: f64,
	eps: f64,
	rms_recip_kernel: RMSRecipKernel,
}

impl RMSNorm {
	pub fn new(n_inputs: usize, eps: f64) -> Self {
		Self {
			shape: [n_inputs],
			gradient_mode: RMSNormGradientMode::Precise,
			sum_to_mean: 1.0 / n_inputs.lossy_into(),
			eps,
			rms_recip_kernel: RMSRecipKernel::new(),
		}
	}

	fn root_mean_square_recip<'a>(&self, inp: &'a Tensor) -> RMSRecipKernelCall<'a> {
		self.rms_recip_kernel.call(&inp, self.sum_to_mean, self.eps)
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
		let magn_recip = inp.new_replace_tail(1, &[1])?;
		magn_recip.assign(self.root_mean_square_recip(&inp))?;

		let out = inp.reuse_or_new_like()?;
		out.assign(&inp * &magn_recip)?;

		let backward_fn = inp_backward.map(|inp_backward| match self.gradient_mode {
			RMSNormGradientMode::Precise => {
				//
				Box::new(RMSNormBackwardFn_Precise {
					out: out.clone(),
					magn_recip,
					sum_to_mean: self.sum_to_mean,
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
	sum_to_mean: f64,
	inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for RMSNormBackwardFn_Precise {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self {
			out,
			magn_recip,
			sum_to_mean,
			inp_backward,
		} = Box::into_inner(self);

		let g = magn_recip.new_empty_like()?; // [..., 1]
		g.assign((&out * &d_out).sum() * sum_to_mean)?;

		let d_inp = out.reuse_or_new_like()?;

		// TODO - could we merge `mul, sub, mul` into a single kernel?
		d_inp.assign(&out * &g)?;
		d_inp.assign(&d_out - &d_inp)?;
		d_inp.assign(&d_inp * &magn_recip)?;

		queue.add(inp_backward, d_inp);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
