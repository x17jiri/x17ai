//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::OnceLock;

use crate::ErrPack;
use crate::tensor::math::EvaluatesToTensor;
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

use super::Kernel;
use super::builder::KernelBuilder;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct RMSRecipKernel {
	kernel: &'static Kernel<0, 1, 2>,
}

impl RMSRecipKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<0, 1, 2>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [], [inp], [sum_to_mean, eps]) =
				KernelBuilder::new("rms", [], ["inp"], ["sum_to_mean", "eps"]);
			builder.build(((inp.clone() * inp).sum() * sum_to_mean).sqrt().recip(eps))
		});
		Self { kernel }
	}

	pub fn calc<'a>(self, inp: &'a Tensor, eps: f64) -> RMSRecipKernelCall<'a> {
		let n_inputs = inp.size(-1).unwrap_or(1);
		let sum_to_mean = 1.0 / n_inputs.lossy_into();
		RMSRecipKernelCall {
			rms_recip_kernel: self,
			inp,
			sum_to_mean,
			eps,
		}
	}

	pub fn calc2<'a>(self, inp: &'a Tensor, sum_to_mean: f64, eps: f64) -> RMSRecipKernelCall<'a> {
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
