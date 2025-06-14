//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

// Optimizer inspired by Adam-mini.
//
//     Original Adam: https://arxiv.org/abs/1412.6980
//     Adam-mini: https://arxiv.org/abs/2406.16793

use crate::Result;
use crate::tensor::Tensor;
use crate::tensor::math::{RSqrt, Sum};

pub struct OptCoef {
	pub(crate) m_decay: f64,       // beta1
	pub(crate) v_decay: f64,       // beta2
	pub(crate) eps: f64,           // epsilon
	pub(crate) learning_rate: f64, // alpha
}

impl Default for OptCoef {
	fn default() -> Self {
		Self {
			m_decay: 0.9,
			v_decay: 0.995,
			eps: 1e-8,
			learning_rate: 0.001,
		}
	}
}

pub struct OptParam {
	pub(crate) parts: usize,
	pub(crate) part_elems: usize,
	pub(crate) part_elems_recip: f64, // 1.0 / (part_elems as f64)
	pub(crate) already_have_grad: bool,

	pub(crate) value: Tensor,   // shape: [parts, part_elems]
	pub(crate) grad: Tensor,    // shape: [parts, part_elems]
	pub(crate) m: Tensor,       // shape: [parts, part_elems]
	pub(crate) v: Tensor,       // shape: [parts, 1]
	pub(crate) v_rsqrt: Tensor, // shape: [parts, 1]

	pub(crate) grad_reshaped: Tensor, // shape: what's expected by the user
}

impl OptParam {
	pub fn new(value: Tensor, parts: usize, part_elems: usize) -> Result<OptParam> {
		let elems = parts.checked_mul(part_elems).expect("Overflow in multiplication");
		assert!(value.elems() == elems, "Tensor size mismatch");

		let grad_reshaped = value.new_empty_like()?;

		let value = value.merge_all_dims()?; // if fails, tensor is not contiguous
		let value = value.reshape_last_dim([parts, part_elems])?;

		let grad = grad_reshaped.clone().merge_all_dims()?;
		let grad = grad.reshape_last_dim([parts, part_elems])?;

		let m = value.new_empty_like()?;

		let v = value.new_empty(&[parts, 1], value.dtype())?;
		let v_rsqrt = v.new_empty_like()?;

		Ok(OptParam {
			parts,
			part_elems,
			part_elems_recip: 1.0 / (part_elems as f64),
			already_have_grad: false,

			value,
			grad,
			m,
			v,
			v_rsqrt,

			grad_reshaped,
		})
	}

	pub fn update_grad(&mut self, update: impl FnOnce(&Tensor, bool) -> Result<()>) -> Result<()> {
		let result = update(&self.grad_reshaped, self.already_have_grad);
		self.already_have_grad = true;
		result
	}

	pub fn zero_grad(&mut self) -> Result<()> {
		self.already_have_grad = false;
		Ok(())
	}

	pub fn step(&mut self, coef: &OptCoef) -> Result<()> {
		let grad = &self.grad;

		// The original Adam uses just `grad * grad`. Adam-mini saves space
		// required for `v` by computing the mean of `grad * grad` for each part
		// of the parameter tensor.
		// Dividing the sum by `part_elems` gives the mean.
		let grad_squared = (grad * grad).sum() * self.part_elems_recip;

		// Update the first moment estimate
		let m_decayed = &self.m * coef.m_decay;
		let m_update = grad * (1.0 - coef.m_decay);
		let new_m = m_decayed + m_update;
		self.m.assign(new_m)?;

		// Update the second moment estimate
		let v_decayed = &self.v * coef.v_decay;
		let v_update = grad_squared * (1.0 - coef.v_decay);
		let new_v = v_decayed + v_update;
		self.v.assign(new_v)?;

		// Update value
		self.v_rsqrt.assign(self.v.rsqrt(coef.eps))?;
		let update = &self.m * &self.v_rsqrt;
		let new_value = &self.value - coef.learning_rate * update;
		self.value.assign(new_value)?;

		Ok(())
	}

	pub fn value(&self) -> &Tensor {
		&self.value
	}

	pub fn parts(&self) -> usize {
		self.parts
	}

	pub fn part_elems(&self) -> usize {
		self.part_elems
	}
}
