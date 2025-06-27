//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::intrinsics::cold_path;
use std::rc::Rc;

use crate::ErrPack;
use crate::nn::optimizer::PartitionError;
use crate::tensor::{DType, Device, Tensor, TensorOpError};

use super::optimizer::{OptCoef, OptParam};

pub struct Param {
	value: Tensor,
	opt_param: Option<OptParam>,
	parts: usize,
	part_elems: usize,
}

impl Param {
	pub fn new(
		shape: &[usize],
		dtype: DType,
		device: Rc<dyn Device>,
	) -> Result<Rc<RefCell<Self>>, ErrPack<TensorOpError>> {
		let value = Tensor::new_empty_on(shape, dtype, device)?;
		let opt_param = None;
		let parts = 1;
		let part_elems = value.elems();
		Ok(Rc::new(RefCell::new(Self { value, opt_param, parts, part_elems })))
	}

	pub fn value(&self) -> &Tensor {
		&self.value
	}

	/// Defines the partitioning of the parameter.
	///
	/// All elements in a part will share the same second moment estimate (the `v` parameter in
	/// Adam).
	///
	/// # Errors
	/// - if `parts * part_elems != self.value().elems()`
	pub fn partition(
		&mut self,
		parts: usize,
		part_elems: usize,
	) -> Result<(), ErrPack<PartitionError>> {
		let total_elems = self.value.elems();
		match parts.checked_mul(part_elems) {
			Some(partition_elems) if partition_elems == total_elems => {},
			_ => {
				cold_path();
				return Err(PartitionError::new(parts, part_elems, total_elems));
			},
		}
		self.parts = parts;
		self.part_elems = part_elems;
		Ok(())
	}

	#[inline(never)]
	fn init_opt_param(&mut self) -> Result<&mut OptParam, ErrPack<TensorOpError>> {
		let opt_param = OptParam::new(&self.value, self.parts, self.part_elems)?;
		self.opt_param = Some(opt_param);

		// We just assigned `self.opt_param`, so `unwrap()` can never fail.
		#[allow(clippy::unwrap_used)]
		Ok(self.opt_param.as_mut().unwrap())
	}

	fn opt_param(&mut self) -> Result<&mut OptParam, ErrPack<TensorOpError>> {
		if let Some(ref mut opt_param) = self.opt_param {
			Ok(opt_param)
		} else {
			cold_path();
			self.init_opt_param()
		}
	}

	pub fn zero_grad(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		Ok(self.opt_param()?.zero_grad()?)
	}

	pub fn add_grad(&mut self, grad: Tensor) -> Result<(), ErrPack<TensorOpError>> {
		Ok(self.opt_param()?.add_grad(grad)?)
	}

	pub fn step(&mut self, opt_coef: &OptCoef) -> Result<(), ErrPack<TensorOpError>> {
		Ok(self.opt_param()?.step(opt_coef)?)
	}
}
