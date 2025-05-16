use crate::param::Param;
use crate::{DType, Device, OptCoef, OptParam, Tensor, TensorSize};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

pub struct ModelContext {
	pub opt_coef: OptCoef,
	pub params: Vec<Rc<RefCell<Param>>>,
	pub device: Rc<dyn Device>,
}

impl ModelContext {
	pub fn new(device: Rc<dyn Device>) -> ModelContext {
		ModelContext {
			opt_coef: OptCoef::default(),
			params: Vec::new(),
			device,
		}
	}

	pub fn new_param(&mut self, shape: &[TensorSize], dtype: DType) -> Rc<RefCell<Param>> {
		let param = Param::new(shape, dtype, self.device.clone());
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
