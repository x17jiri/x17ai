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
pub struct MulXLnYKernel {
	kernel: &'static Kernel<2, 0, 0>,
}

impl MulXLnYKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<2, 0, 0>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [x, y], [], []) = KernelBuilder::new("x * ln(y)", ["x", "y"], [], []);
			builder.build(x * y.ln_clamped())
		});
		Self { kernel }
	}

	pub fn call<'a>(self, x: &'a Tensor, y: &'a Tensor) -> MulXLnYKernelCall<'a> {
		MulXLnYKernelCall { kernel: self, x, y }
	}
}

pub struct MulXLnYKernelCall<'a> {
	kernel: MulXLnYKernel,
	x: &'a Tensor,
	y: &'a Tensor,
}

impl<'a> EvaluatesToTensor for MulXLnYKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.x, self.y], [], [])
	}
}

//--------------------------------------------------------------------------------------------------
