//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::nn::optimizer::OptimizerError;
use crate::tensor::{DType, Device, TensorOpError};

use super::optimizer::OptCoef;
use super::param::Param;

pub struct ModelContext {
	pub opt_coef: OptCoef,
	pub params: Vec<Rc<RefCell<Param>>>,
	pub device: Rc<dyn Device>,
}

impl ModelContext {
	pub fn new(device: Rc<dyn Device>) -> Self {
		Self {
			opt_coef: OptCoef::default(),
			params: Vec::new(),
			device,
		}
	}

	pub fn new_param(
		&mut self,
		shape: &[usize],
		dtype: DType,
	) -> Result<Rc<RefCell<Param>>, ErrPack<TensorOpError>> {
		let param = Param::new(shape, dtype, self.device.clone())?;
		self.params.push(param.clone());
		Ok(param)
	}

	pub fn zero_grad(&mut self) -> Result<(), ErrPack<OptimizerError>> {
		for param in &self.params {
			param.borrow_mut().zero_grad()?;
		}
		Ok(())
	}

	pub fn step(&mut self) -> Result<(), ErrPack<OptimizerError>> {
		for param in &self.params {
			param.borrow_mut().step(&self.opt_coef)?;
		}
		Ok(())
	}
}
