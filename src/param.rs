use crate::device::Device;
use crate::dtype::DType;
use crate::optimizer::{OptCoef, OptParam};
use crate::tensor::{Tensor, TensorSize};
use std::cell::RefCell;
use std::rc::Rc;

pub struct Param {
	value: Tensor,
	opt_param: Option<OptParam>,
}

impl Param {
	pub fn new(shape: &[TensorSize], dtype: DType, device: Rc<dyn Device>) -> Rc<RefCell<Param>> {
		Rc::new(RefCell::new(Param {
			value: Tensor::new_empty_on(shape, dtype, device),
			opt_param: None,
		}))
	}

	pub fn value(&self) -> &Tensor {
		&self.value
	}

	pub fn init_optimizer(&mut self, parts: TensorSize, part_elems: TensorSize) {
		assert!(self.opt_param.is_none(), "Optimizer already initialized for this parameter");
		let elems = parts.checked_mul(part_elems).expect("Overflow in multiplication");
		assert!(self.value.elems() == elems, "Tensor size mismatch");

		self.opt_param = Some(OptParam::new(self.value.clone().reshape_all(&[parts, part_elems])));
	}

	pub fn zero_grad(&mut self) {
		self.opt_param //
			.as_mut()
			.expect("Optimizer not initialized")
			.zero_grad();
	}

	pub fn update_grad(&mut self, update: impl FnOnce(&Tensor, bool)) {
		self.opt_param //
			.as_mut()
			.expect("Optimizer not initialized")
			.update_grad(update);
	}

	pub fn step(&mut self, opt_coef: &OptCoef) {
		self.opt_param //
			.as_mut()
			.expect("Optimizer not initialized")
			.step(opt_coef);
	}
}
