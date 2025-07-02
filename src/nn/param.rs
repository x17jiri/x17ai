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
use crate::nn::optimizer::{CurrentGradValue, PartitionError};
use crate::tensor::{DType, Device, Tensor, TensorOpError};

use super::optimizer::{OptCoef, OptParam};

pub struct Param {
	value: Tensor,
	opt_param: Option<Box<OptParam>>,
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
		let parts = value.elems();
		let part_elems = 1;
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
	pub fn init_optimizer(&mut self) -> Result<&mut OptParam, ErrPack<TensorOpError>> {
		let opt_param = Box::new(OptParam::new(self.value.clone(), self.parts, self.part_elems)?);
		Ok(self.opt_param.insert(opt_param))
	}

	#[inline(never)]
	pub fn deinit_optimizer(&mut self) {
		self.opt_param.take();
	}

	pub fn requires_grad(&self) -> bool {
		self.opt_param.is_some()
	}

	pub fn zero_grad(&mut self) {
		if let Some(opt_param) = &mut self.opt_param {
			opt_param.zero_grad();
		}
	}

	pub fn update_grad(
		&mut self,
		f: impl FnMut(CurrentGradValue) -> Result<(), ErrPack<TensorOpError>>,
	) -> Result<(), ErrPack<TensorOpError>> {
		#[allow(clippy::option_if_let_else)]
		if let Some(opt_param) = &mut self.opt_param {
			opt_param.update_grad(f) //
		} else {
			Ok(())
		}
	}

	pub fn grad(&self) -> Option<&Tensor> {
		self.opt_param.as_ref().and_then(|opt_param| opt_param.grad())
	}

	pub fn step(&mut self, opt_coef: &OptCoef) -> Result<(), ErrPack<TensorOpError>> {
		#[allow(clippy::option_if_let_else)]
		if let Some(opt_param) = &mut self.opt_param {
			opt_param.step(opt_coef) //
		} else {
			Ok(())
		}
	}
}
