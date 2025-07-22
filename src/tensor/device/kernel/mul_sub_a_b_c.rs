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
pub struct MulSubABCKernel {
	kernel: &'static Kernel<3, 0, 0>,
}

impl MulSubABCKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<3, 0, 0>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [a, b, c], [], []) =
				KernelBuilder::new("(a - b) * c", ["a", "b", "c"], [], []);
			builder.build((a - b) * c)
		});
		Self { kernel }
	}

	pub fn call<'a>(self, a: &'a Tensor, b: &'a Tensor, c: &'a Tensor) -> MulSubABCKernelCall<'a> {
		MulSubABCKernelCall { kernel: self, a, b, c }
	}
}

pub struct MulSubABCKernelCall<'a> {
	kernel: MulSubABCKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	c: &'a Tensor,
}

impl<'a> EvaluatesToTensor for MulSubABCKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b, self.c], [], [])
	}
}

//--------------------------------------------------------------------------------------------------
