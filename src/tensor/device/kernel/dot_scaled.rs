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
pub struct DotScaledKernel {
	kernel: &'static Kernel<0, 2, 1>,
}

impl DotScaledKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<0, 2, 1>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [], [a, b], [scale]) =
				KernelBuilder::new("dot", [], ["a", "b"], ["scale"]);
			builder.build((a * b).sum() * scale)
		});
		Self { kernel }
	}

	pub fn call<'a>(self, a: &'a Tensor, b: &'a Tensor, scale: f64) -> DotScaledKernelCall<'a> {
		DotScaledKernelCall { kernel: self, a, b, scale }
	}
}

pub struct DotScaledKernelCall<'a> {
	kernel: DotScaledKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	scale: f64,
}

impl<'a> EvaluatesToTensor for DotScaledKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [], [self.a, self.b], [self.scale])
	}
}

//--------------------------------------------------------------------------------------------------
