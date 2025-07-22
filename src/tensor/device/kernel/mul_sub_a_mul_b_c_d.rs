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
pub struct MulSubAMulBCDKernel {
	kernel: &'static Kernel<4, 0, 0>,
}

impl MulSubAMulBCDKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<4, 0, 0>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [a, b, c, d], [], []) =
				KernelBuilder::new("(a - (b * c)) * d", ["a", "b", "c", "d"], [], []);
			builder.build((a - (b * c)) * d)
		});
		Self { kernel }
	}

	pub fn call<'a>(
		self,
		a: &'a Tensor,
		b: &'a Tensor,
		c: &'a Tensor,
		d: &'a Tensor,
	) -> MulSubAMulBCDKernelCall<'a> {
		MulSubAMulBCDKernelCall { kernel: self, a, b, c, d }
	}
}

pub struct MulSubAMulBCDKernelCall<'a> {
	kernel: MulSubAMulBCDKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	c: &'a Tensor,
	d: &'a Tensor,
}

impl<'a> EvaluatesToTensor for MulSubAMulBCDKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b, self.c, self.d], [], [])
	}
}

//--------------------------------------------------------------------------------------------------
