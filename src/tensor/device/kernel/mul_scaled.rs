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

use super::Kernel;
use super::builder::KernelBuilder;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct MulScaledKernel {
	kernel: &'static Kernel<2, 0, 1>,
}

impl MulScaledKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<2, 0, 1>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [a, b], [], [scale]) =
				KernelBuilder::new("mul", ["a", "b"], [], ["scale"]);
			builder.build(a * b * scale)
		});
		Self { kernel }
	}

	pub fn calc<'a>(self, a: &'a Tensor, b: &'a Tensor, scale: f64) -> MulScaledKernelCall<'a> {
		MulScaledKernelCall { kernel: self, a, b, scale }
	}
}

pub struct MulScaledKernelCall<'a> {
	kernel: MulScaledKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	scale: f64,
}

impl<'a> EvaluatesToTensor for MulScaledKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b], [], [self.scale])
	}
}

//--------------------------------------------------------------------------------------------------
