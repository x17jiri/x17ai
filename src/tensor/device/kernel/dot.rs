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
pub struct DotKernel {
	kernel: &'static Kernel<0, 2, 0>,
}

impl DotKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<0, 2, 0>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [], [a, b], []) = KernelBuilder::new("dot", [], ["a", "b"], []);
			builder.build((a * b).sum())
		});
		Self { kernel }
	}

	pub fn call<'a>(self, a: &'a Tensor, b: &'a Tensor) -> DotKernelCall<'a> {
		DotKernelCall { kernel: self, a, b }
	}
}

pub struct DotKernelCall<'a> {
	kernel: DotKernel,
	a: &'a Tensor,
	b: &'a Tensor,
}

impl<'a> EvaluatesToTensor for DotKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [], [self.a, self.b], [])
	}
}

//--------------------------------------------------------------------------------------------------
