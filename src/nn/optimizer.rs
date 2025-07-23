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

use std::hint::cold_path;

use crate::tensor::device::kernel::lookup::tsr;
use crate::tensor::math::{Recip, Sqrt, Sum};
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;
use crate::{ErrExtra, ErrPack};

//--------------------------------------------------------------------------------------------------

pub enum CurrentGradValue<'a> {
	Uninit(&'a Tensor),
	Value(&'a Tensor),
}

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
			v_decay: 0.99,
			eps: 1e-8,
			learning_rate: 0.001,
		}
	}
}

pub struct OptParam {
	pub(crate) parts: usize,
	pub(crate) part_elems: usize,
	pub(crate) part_elems_recip: f64, // 1.0 / (part_elems as f64)

	pub(crate) value: Tensor,   // shape: [parts, part_elems]
	pub(crate) m: Tensor,       // shape: [parts, part_elems]
	pub(crate) v: Tensor,       // shape: [parts, 1]
	pub(crate) v_rsqrt: Tensor, // shape: [parts, 1]

	pub(crate) value_orig_shape: Tensor, // shape: user defined
	pub(crate) grad: Option<Tensor>,     // shape: same as value_orig_shape
	pub(crate) grad_error: bool,
}

impl OptParam {
	/// # Panics
	/// - when the value tensor is not contiguous
	/// - when the value tensor cannot be reshaped to [parts, part_elems]
	#[allow(clippy::unwrap_used)]
	pub fn new(
		value: Tensor,
		parts: usize,
		part_elems: usize,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let value_orig_shape = value;
		let value = value_orig_shape.merge_all_dims().unwrap(); // if fails, tensor is not contiguous
		let value = value.reshape_last_dim([parts, part_elems]).unwrap();

		let m = value.new_empty_like()?;

		let v = value.new_empty(&[parts, 1], value.dtype())?;
		let v_rsqrt = v.new_empty_like()?;

		Ok(Self {
			parts,
			part_elems,
			part_elems_recip: 1.0 / part_elems.lossy_into(),

			value,
			m,
			v,
			v_rsqrt,

			value_orig_shape,
			grad: None,
			grad_error: false,
		})
	}

	pub fn zero_grad(&mut self) {
		self.grad = None;
		self.grad_error = false;
	}

	#[cold]
	#[inline(never)]
	fn grad_error(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		Err(ErrPack {
			code: TensorOpError::InvalidValue,
			extra: Some(Box::new(ErrExtra {
				message: "There was an error while computing the gradient. \
					We don't have a valid value to use"
					.to_string(),
				nested: None,
			})),
		})
	}

	#[allow(clippy::panic_in_result_fn)]
	pub fn step(&mut self, coef: &OptCoef) -> Result<(), ErrPack<TensorOpError>> {
		if self.grad_error {
			cold_path();
			return self.grad_error();
		}

		let Some(grad) = &self.grad else {
			return Ok(());
		};
		let grad = grad.merge_all_dims()?;
		let grad = grad.reshape_last_dim([self.parts, self.part_elems])?;

		// Update the first moment estimate
		let m_decayed = tsr(&self.m) * coef.m_decay;
		let m_update = tsr(&grad) * (1.0 - coef.m_decay);
		let new_m = m_decayed + m_update;
		self.m.assign2(new_m)?;

		// Update the second moment estimate
		// The original Adam uses just `grad * grad`. Adam-mini saves space
		// required for `v` by computing the mean of `grad * grad` for each part
		// of the parameter tensor.
		// Dividing the sum by `part_elems` gives the mean.
		let grad_squared = (&grad * &grad).sum() * self.part_elems_recip;
		let v_decayed = tsr(&self.v) * coef.v_decay;
		let v_update = grad_squared * (1.0 - coef.v_decay);
		let new_v = v_decayed + v_update;
		self.v.assign(new_v)?;

		// Update value
		self.v_rsqrt.assign(self.v.sqrt().recip(coef.eps))?;
		let update = &self.m * &self.v_rsqrt;
		let new_value = &self.value - coef.learning_rate * update;
		self.value.assign(new_value)?;

		Ok(())
	}

	pub fn value(&self) -> &Tensor {
		&self.value
	}

	pub fn grad(&self) -> Option<&Tensor> {
		if self.grad_error {
			cold_path();
			return None;
		}
		self.grad.as_ref()
	}

	pub fn parts(&self) -> usize {
		self.parts
	}

	pub fn part_elems(&self) -> usize {
		self.part_elems
	}

	pub fn update_grad(
		&mut self,
		mut f: impl FnMut(CurrentGradValue) -> Result<(), ErrPack<TensorOpError>>,
	) -> Result<(), ErrPack<TensorOpError>> {
		if let Some(grad) = &self.grad {
			let result = f(CurrentGradValue::Value(grad));
			self.grad_error |= result.is_err();
			result
		} else {
			match self.value_orig_shape.new_empty_like() {
				Ok(grad) => {
					let grad = self.grad.insert(grad);
					let result = f(CurrentGradValue::Uninit(grad));
					self.grad_error |= result.is_err();
					result
				},
				Err(err) => {
					cold_path();
					self.grad_error = true;
					Err(err)
				},
			}
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PartitionError;

impl PartitionError {
	#[cold]
	#[inline(never)]
	pub fn new(parts: usize, part_elems: usize, total_elems: usize) -> ErrPack<Self> {
		let message = format!(
			"Invalid parameter partition: parts * part_elems must equal the total number of elements in the tensor. parts = {parts}, part_elems = {part_elems}, total_elems = {total_elems}"
		);
		ErrPack {
			code: Self,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}
}

//--------------------------------------------------------------------------------------------------
