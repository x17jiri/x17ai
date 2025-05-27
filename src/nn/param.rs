// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::cell::RefCell;
use std::intrinsics::cold_path;
use std::rc::Rc;

use crate::tensor::{DType, Device, Tensor};

use super::optimizer::{OptCoef, OptParam};

pub struct Param {
	value: Tensor,
	opt_param: Option<OptParam>,
	parts: usize,
	part_elems: usize,
}

impl Param {
	pub fn new(shape: &[usize], dtype: DType, device: Rc<dyn Device>) -> Rc<RefCell<Param>> {
		let value = Tensor::new_empty_on(shape, dtype, device);
		let opt_param = None;
		let parts = 1;
		let part_elems = value.elems();
		Rc::new(RefCell::new(Param { value, opt_param, parts, part_elems }))
	}

	pub fn value(&self) -> &Tensor {
		&self.value
	}

	pub fn partition(&mut self, parts: usize, part_elems: usize) {
		assert!(self.value.elems() == parts * part_elems, "Tensor size mismatch");
		self.parts = parts;
		self.part_elems = part_elems;
	}

	#[inline(never)]
	fn init_opt_param(&mut self) -> &mut OptParam {
		self.opt_param = Some(OptParam::new(self.value.clone(), self.parts, self.part_elems));
		self.opt_param.as_mut().unwrap()
	}

	fn opt_param(&mut self) -> &mut OptParam {
		if let Some(ref mut opt_param) = self.opt_param {
			opt_param
		} else {
			cold_path();
			self.init_opt_param()
		}
	}

	pub fn zero_grad(&mut self) {
		self.opt_param().zero_grad();
	}

	pub fn update_grad(&mut self, update: impl FnOnce(&Tensor, bool)) {
		self.opt_param().update_grad(update);
	}

	pub fn step(&mut self, opt_coef: &OptCoef) {
		self.opt_param().step(opt_coef);
	}
}
