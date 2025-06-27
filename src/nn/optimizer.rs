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

use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::{Autograd, BackwardFn};
use crate::nn::Param;
use crate::tensor::math::{RSqrt, Sum};
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;
use crate::{ErrExtra, ErrPack};

//--------------------------------------------------------------------------------------------------

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

	pub(crate) value: Tensor,        // shape: [parts, part_elems]
	pub(crate) grad: Option<Tensor>, // shape: [parts, part_elems]
	pub(crate) m: Tensor,            // shape: [parts, part_elems]
	pub(crate) v: Tensor,            // shape: [parts, 1]
	pub(crate) v_rsqrt: Tensor,      // shape: [parts, 1]
}

impl OptParam {
	pub fn new(
		value: &Tensor,
		parts: usize,
		part_elems: usize,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let value = value.merge_all_dims().unwrap(); // if fails, tensor is not contiguous
		let value = value.reshape_last_dim([parts, part_elems]).unwrap();

		let m = value.new_empty_like()?;

		let v = value.new_empty(&[parts, 1], value.dtype())?;
		let v_rsqrt = v.new_empty_like()?;

		Ok(Self {
			parts,
			part_elems,
			part_elems_recip: 1.0 / part_elems.lossy_into(),

			value,
			grad: None,
			m,
			v,
			v_rsqrt,
		})
	}

	pub fn zero_grad(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.grad = None;
		Ok(())
	}

	pub fn step(&mut self, coef: &OptCoef) -> Result<(), ErrPack<TensorOpError>> {
		let Some(grad) = &self.grad else {
			return Ok(());
		};

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

	pub fn add_grad(&mut self, grad: Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let grad_reshaped = grad.merge_all_dims()?;
		let grad_reshaped = grad_reshaped.reshape_last_dim([self.parts, self.part_elems])?;
		std::mem::drop(grad);
		if let Some(existing_grad) = &self.grad {
			existing_grad.assign(existing_grad + &grad_reshaped)?;
		} else {
			if grad_reshaped.owns_buffer() {
				self.grad = Some(grad_reshaped);
			} else {
				todo!("need to copy");
			}
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ParamUpdateBackwardFn {
	pub param: Rc<RefCell<Param>>,
}

impl BackwardFn for ParamUpdateBackwardFn {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		_autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		self.param.borrow_mut().add_grad(d_out)
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
