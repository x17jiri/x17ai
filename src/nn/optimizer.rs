// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

// Optimizer. Inspired by Adam-mini: https://arxiv.org/abs/2406.16793

use crate::tensor::math::{Accumulable, Savable};
use crate::tensor::{self, Tensor, TensorSize};

pub struct OptCoef {
	pub(crate) momentum_decay: f64, // beta1
	pub(crate) velocity_decay: f64, // beta2
	pub(crate) eps: f64,            // epsilon
	pub(crate) learning_rate: f64,  // alpha
}

impl Default for OptCoef {
	fn default() -> Self {
		OptCoef {
			momentum_decay: 0.9,
			velocity_decay: 0.99,
			eps: 1e-8,
			learning_rate: 0.001,
		}
	}
}

pub struct OptParam {
	pub(crate) parts: TensorSize,
	pub(crate) part_elems: TensorSize,
	pub(crate) part_elems_recip: f64, // 1.0 / (part_elems as f64)
	pub(crate) already_have_grad: bool,

	pub(crate) value: Tensor,          // shape: [parts, part_elems]
	pub(crate) grad: Tensor,           // shape: [parts, part_elems]
	pub(crate) momentum: Tensor,       // shape: [parts, part_elems]
	pub(crate) velocity: Tensor,       // shape: [parts, 1]
	pub(crate) velocity_recip: Tensor, // shape: [parts, 1]

	pub(crate) grad_reshaped: Tensor, // shape: what's expected by the user
}

impl OptParam {
	pub fn new(value: Tensor, parts: TensorSize, part_elems: TensorSize) -> OptParam {
		let elems = parts.checked_mul(part_elems).expect("Overflow in multiplication");
		assert!(value.elems() == elems, "Tensor size mismatch");

		let grad_reshaped = value.new_empty_like();

		let value = value.merge_all_dims(); // if fail, tensor is not contiguous
		let value = value.reshape_last_dim([parts, part_elems]);

		let grad = grad_reshaped.clone().merge_all_dims();
		let grad = grad.reshape_last_dim([parts, part_elems]);

		let momentum = value.new_empty_like();

		let velocity = value.new_empty(&[parts, 1], value.dtype());
		let velocity_recip = velocity.new_empty_like();

		OptParam {
			parts,
			part_elems,
			part_elems_recip: 1.0 / (part_elems as f64),
			already_have_grad: false,

			value,
			grad,
			momentum,
			velocity,
			velocity_recip,

			grad_reshaped,
		}
	}

	pub fn update_grad(&mut self, update: impl FnOnce(&Tensor, bool)) {
		update(&self.grad_reshaped, self.already_have_grad);
		self.already_have_grad = true;
	}

	pub fn zero_grad(&mut self) {
		self.already_have_grad = false;
	}

	pub fn step(&mut self, coef: &OptCoef) {
		// update momentum
		self.grad.acc_to(&self.momentum, coef.momentum_decay, 1.0 - coef.momentum_decay);

		// update velocity
		// `dot(grad, grad)` calculates the sum of squares.
		// Dividing the sum by `part_elems` gives the mean of squares.
		tensor::math::dot(&self.grad, &self.grad).acc_to(
			&self.velocity,
			coef.velocity_decay,
			(1.0 - coef.velocity_decay) * self.part_elems_recip,
		);
		tensor::math::rsqrt(&self.velocity, coef.eps).save_to(&self.velocity_recip);

		// update value
		tensor::math::mul(&self.momentum, &self.velocity_recip).acc_to(
			&self.value,
			1.0,
			-coef.learning_rate,
		);
	}

	pub fn value(&self) -> &Tensor {
		&self.value
	}

	pub fn parts(&self) -> TensorSize {
		self.parts
	}

	pub fn part_elems(&self) -> TensorSize {
		self.part_elems
	}
}
