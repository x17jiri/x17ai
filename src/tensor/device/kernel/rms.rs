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
pub struct RMSKernel {
	kernel: &'static Kernel<0, 1, 1>,
}

impl RMSKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<0, 1, 1>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [], [inp], [sum_to_mean]) =
				KernelBuilder::new("rms", [], ["inp"], ["sum_to_mean"]);
			builder.build(((inp.clone() * inp).sum() * sum_to_mean).sqrt())
		});
		Self { kernel }
	}

	pub fn call<'a>(self, inp: &'a Tensor) -> RMSKernelCall<'a> {
		let n_inputs = inp.size(-1).unwrap_or(1);
		let sum_to_mean = 1.0 / n_inputs.lossy_into();
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
