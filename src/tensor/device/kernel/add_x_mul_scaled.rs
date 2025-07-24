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
pub struct AddXMulScaledKernel {
	kernel: &'static Kernel<3, 0, 1>,
}

impl AddXMulScaledKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<3, 0, 1>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [x, a, b], [], [c]) =
				KernelBuilder::new("x + (a * b * c)", ["x", "a", "b"], [], ["c"]);
			builder.build(x + (a * b * c))
		});
		Self { kernel }
	}

	pub fn call<'a>(
		self,
		x: &'a Tensor,
		a: &'a Tensor,
		b: &'a Tensor,
		c: f64,
	) -> AddXMulScaledKernelCall<'a> {
		AddXMulScaledKernelCall { kernel: self, x, a, b, c }
	}
}

pub struct AddXMulScaledKernelCall<'a> {
	kernel: AddXMulScaledKernel,
	x: &'a Tensor,
	a: &'a Tensor,
	b: &'a Tensor,
	c: f64,
}

impl<'a> EvaluatesToTensor for AddXMulScaledKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.x, self.a, self.b], [], [self.c])
	}
}

//--------------------------------------------------------------------------------------------------
