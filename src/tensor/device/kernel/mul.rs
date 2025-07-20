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
pub struct MulKernel {
	kernel: &'static Kernel<2, 0, 0>,
}

impl MulKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<2, 0, 0>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [a, b], [], []) = KernelBuilder::new("mul", ["a", "b"], [], []);
			builder.build(a * b)
		});
		Self { kernel }
	}

	pub fn calc<'a>(self, a: &'a Tensor, b: &'a Tensor) -> MulKernelCall<'a> {
		MulKernelCall { kernel: self, a, b }
	}
}

pub struct MulKernelCall<'a> {
	kernel: MulKernel,
	a: &'a Tensor,
	b: &'a Tensor,
}

impl<'a> EvaluatesToTensor for MulKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b], [], [])
	}
}

//--------------------------------------------------------------------------------------------------
