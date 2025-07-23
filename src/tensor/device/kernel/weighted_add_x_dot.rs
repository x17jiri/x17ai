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
pub struct WeightedAddXDotKernel {
	kernel: &'static Kernel<1, 2, 2>,
}

impl WeightedAddXDotKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<1, 2, 2>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [x], [a, b], [x_weight, dot_weight]) = KernelBuilder::new(
				"weighted_add(x, dot(...))",
				["x"],
				["a", "b"],
				["x_weight", "dot_weight"],
			);
			builder.build((x * x_weight) + ((a * b).sum() * dot_weight))
		});
		Self { kernel }
	}

	pub fn call<'a>(
		self,
		x: &'a Tensor,
		x_weight: f64,
		a: &'a Tensor,
		b: &'a Tensor,
		dot_weight: f64,
	) -> WeightedAddXDotKernelCall<'a> {
		WeightedAddXDotKernelCall {
			kernel: self,
			x,
			a,
			b,
			x_weight,
			dot_weight,
		}
	}
}

pub struct WeightedAddXDotKernelCall<'a> {
	kernel: WeightedAddXDotKernel,
	x: &'a Tensor,
	a: &'a Tensor,
	b: &'a Tensor,
	x_weight: f64,
	dot_weight: f64,
}

impl<'a> EvaluatesToTensor for WeightedAddXDotKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.x], [self.a, self.b], [self.x_weight, self.dot_weight])
	}
}

//--------------------------------------------------------------------------------------------------
