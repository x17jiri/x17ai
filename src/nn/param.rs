//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::intrinsics::cold_path;
use std::rc::Rc;

use crate::tensor::{DType, Device, Tensor};
use crate::{Error, Result};

use super::optimizer::{OptCoef, OptParam};

pub struct Param {
	value: Tensor,
	opt_param: Option<OptParam>,
	parts: usize,
	part_elems: usize,
}

impl Param {
	pub fn new(shape: &[usize], dtype: DType, device: Rc<dyn Device>) -> Rc<RefCell<Self>> {
		let value = Tensor::new_empty_on(shape, dtype, device);
		let opt_param = None;
		let parts = 1;
		let part_elems = value.elems();
		Rc::new(RefCell::new(Self { value, opt_param, parts, part_elems }))
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
	pub fn partition(&mut self, parts: usize, part_elems: usize) -> Result<()> {
		let total_elems = self.value.elems();
		if parts * part_elems != total_elems {
			#[cold]
			fn err_invalid_param_partition(
				parts: usize, part_elems: usize, total_elems: usize,
			) -> Error {
				format!("Invalid parameter partition: parts * part_elems must equal the total number of elements in the tensor. parts = {parts}, part_elems = {part_elems}, total_elems = {total_elems}").into()
			}
			return Err(err_invalid_param_partition(parts, part_elems, total_elems));
		}
		self.parts = parts;
		self.part_elems = part_elems;
		Ok(())
	}

	#[inline(never)]
	fn init_opt_param(&mut self) -> Result<&mut OptParam> {
		let opt_param = OptParam::new(self.value.clone(), self.parts, self.part_elems)?;
		self.opt_param = Some(opt_param);

		// We just assigned `self.opt_param`, so `unwrap()` can never fail.
		#[allow(clippy::unwrap_used)]
		Ok(self.opt_param.as_mut().unwrap())
	}

	fn opt_param(&mut self) -> Result<&mut OptParam> {
		if let Some(ref mut opt_param) = self.opt_param {
			Ok(opt_param)
		} else {
			cold_path();
			self.init_opt_param()
		}
	}

	pub fn zero_grad(&mut self) -> Result<()> {
		self.opt_param()?.zero_grad()
	}

	pub fn update_grad(&mut self, update: impl FnOnce(&Tensor, bool) -> Result<()>) -> Result<()> {
		self.opt_param()?.update_grad(update)
	}

	pub fn step(&mut self, opt_coef: &OptCoef) -> Result<()> {
		self.opt_param()?.step(opt_coef)
	}
}
