use crate::{DType, Device, OptCoef, OptParam, Tensor, TensorSize};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

pub struct Context {
	pub opt_coef: OptCoef,
	pub params: Vec<Rc<RefCell<OptParam>>>,
	pub device: Rc<dyn Device>,
}

impl Context {
	pub fn new(device: Rc<dyn Device>) -> Context {
		Context {
			opt_coef: OptCoef::default(),
			params: Vec::new(),
			device,
		}
	}

	pub fn add_param(
		&mut self, dtype: DType, parts: TensorSize, part_elems: TensorSize,
	) -> Rc<RefCell<OptParam>> {
		let param = OptParam::new(self.device.clone(), dtype, parts, part_elems);
		self.params.push(param.clone());
		param
	}

	pub fn zero_grad(&mut self) {
		for param in &self.params {
			param.borrow_mut().zero_grad();
		}
	}

	pub fn step(&mut self) {
		for param in &self.params {
			param.borrow_mut().step(&self.opt_coef);
		}
	}
}
