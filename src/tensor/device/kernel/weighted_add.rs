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
pub struct WeightedAddKernel {
	kernel: &'static Kernel<2, 0, 2>,
}

impl WeightedAddKernel {
	pub fn instance() -> Self {
		static instance: OnceLock<Kernel<2, 0, 2>> = OnceLock::new();
		let kernel = instance.get_or_init(|| {
			let (builder, [a, b], [], [a_weight, b_weight]) =
				KernelBuilder::new("weighted add", ["a", "b"], [], ["a_weight", "b_weight"]);
			builder.build((a * a_weight) + (b * b_weight))
		});
		Self { kernel }
	}

	pub fn call<'a>(
		self,
		a: &'a Tensor,
		a_weight: f64,
		b: &'a Tensor,
		b_weight: f64,
	) -> WeightedAddKernelCall<'a> {
		WeightedAddKernelCall { kernel: self, a, b, a_weight, b_weight }
	}
}

pub struct WeightedAddKernelCall<'a> {
	kernel: WeightedAddKernel,
	a: &'a Tensor,
	b: &'a Tensor,
	a_weight: f64,
	b_weight: f64,
}

impl<'a> EvaluatesToTensor for WeightedAddKernelCall<'a> {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.kernel.kernel.run(to, [self.a, self.b], [], [self.a_weight, self.b_weight])
	}
}

//--------------------------------------------------------------------------------------------------
