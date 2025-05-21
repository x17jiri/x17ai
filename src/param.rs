use crate::device::Device;
use crate::dtype::DType;
use crate::optimizer::{OptCoef, OptParam};
use crate::tensor::{Tensor, TensorSize};
use std::cell::RefCell;
use std::intrinsics::cold_path;
use std::rc::Rc;

pub struct Param {
	value: Tensor,
	opt_param: Option<OptParam>,
	parts: TensorSize,
	part_elems: TensorSize,
}

impl Param {
	pub fn new(shape: &[TensorSize], dtype: DType, device: Rc<dyn Device>) -> Rc<RefCell<Param>> {
		let value = Tensor::new_empty_on(shape, dtype, device);
		let opt_param = None;
		let parts = 1;
		let part_elems = value.elems();
		Rc::new(RefCell::new(Param { value, opt_param, parts, part_elems }))
	}

	pub fn value(&self) -> &Tensor {
		&self.value
	}

	pub fn partition(&mut self, parts: TensorSize, part_elems: TensorSize) {
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
