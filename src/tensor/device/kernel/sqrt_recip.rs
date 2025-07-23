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
pub struct SqrtRecipKernel {
	kernel: &'static Kernel<1, 0, 1>,
}

impl SqrtRecipKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<1, 0, 1>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [a], [], [eps]) = KernelBuilder::new("1.0 / sqrt", ["a"], [], ["eps"]);
			builder.build(a.sqrt().recip(eps))
		});
		Self { kernel }
	}

	pub fn call<'a>(self, a: &'a Tensor, eps: f64) -> SqrtRecipKernelCall<'a> {
		SqrtRecipKernelCall { kernel: self, a, eps }
	}
}

pub struct SqrtRecipKernelCall<'a> {
	kernel: SqrtRecipKernel,
	a: &'a Tensor,
	eps: f64,
}

impl<'a> EvaluatesToTensor for SqrtRecipKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a], [], [self.eps])
	}
}

//--------------------------------------------------------------------------------------------------
