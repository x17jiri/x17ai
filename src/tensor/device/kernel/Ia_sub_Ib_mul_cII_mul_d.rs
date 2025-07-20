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
pub struct Ia_sub_Ib_mul_cII_mul_d_Kernel {
	kernel: &'static Kernel<4, 0, 0>,
}

impl Ia_sub_Ib_mul_cII_mul_d_Kernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<4, 0, 0>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [a, b, c, d], [], []) =
				KernelBuilder::new("(a - (b * c)) * d", ["a", "b", "c", "d"], [], []);
			builder.build((a - (b * c)) * d)
		});
		Self { kernel }
	}

	pub fn calc<'a>(
		self,
		a: &'a Tensor,
		b: &'a Tensor,
		c: &'a Tensor,
		d: &'a Tensor,
	) -> Ia_sub_Ib_mul_cII_mul_d_KernelCall<'a> {
		Ia_sub_Ib_mul_cII_mul_d_KernelCall { kernel: self, a, b, c, d }
	}
}

pub struct Ia_sub_Ib_mul_cII_mul_d_KernelCall<'a> {
	kernel: Ia_sub_Ib_mul_cII_mul_d_Kernel,
	a: &'a Tensor,
	b: &'a Tensor,
	c: &'a Tensor,
	d: &'a Tensor,
}

impl<'a> EvaluatesToTensor for Ia_sub_Ib_mul_cII_mul_d_KernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b, self.c, self.d], [], [])
	}
}

//--------------------------------------------------------------------------------------------------
