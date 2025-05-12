// Optimizer. Inspired by Adam-mini: https://arxiv.org/abs/2406.16793

use crate::expr::*;
use crate::{DType, Device, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub struct OptCoef {
	pub(crate) momentum_decay: f64, // beta1
	pub(crate) velocity_decay: f64, // beta2
	pub(crate) eps: f64,            // epsilon
	pub(crate) learning_rate: f64,  // alpha
}

impl Default for OptCoef {
	fn default() -> Self {
		OptCoef {
			momentum_decay: 0.9,
			velocity_decay: 0.99,
			eps: 1e-8,
			learning_rate: 0.001,
		}
	}
}

pub struct OptParam {
	pub(crate) parts: usize,
	pub(crate) part_elems: usize,

	pub(crate) value: Tensor,          // shape: [parts, part_elems]
	pub(crate) grad: Tensor,           // shape: [parts, part_elems]
	pub(crate) momentum: Tensor,       // shape: [parts, part_elems]
	pub(crate) velocity: Tensor,       // shape: [parts, 1]
	pub(crate) velocity_recip: Tensor, // shape: [parts, 1]

	pub(crate) stored_tensors: Vec<Tensor>,
}

impl OptParam {
	pub fn new(
		device: Rc<dyn Device>, dtype: DType, parts: usize, part_elems: usize,
	) -> Rc<RefCell<OptParam>> {
		let value = Tensor::new_empty_on(&[parts, part_elems], dtype, device);
		let grad = value.new_empty_like();
		let momentum = value.new_empty_like();

		let velocity = value.new_empty(&[parts, 1], dtype);
		let velocity_recip = velocity.new_empty_like();

		Rc::new(RefCell::new(OptParam {
			parts,
			part_elems,

			value,
			grad,
			momentum,
			velocity,
			velocity_recip,

			stored_tensors: Vec::new(),
		}))
	}

	pub fn grad(&self) -> &Tensor {
		&self.grad
	}

	pub fn zero_grad(&mut self) {
		zeros().save_to(&self.grad);
	}

	pub fn step(&mut self, coef: &OptCoef) {
		// update momentum
		self.grad.acc_to(&self.momentum, coef.momentum_decay, 1.0 - coef.momentum_decay);

		// update velocity
		// `vec_mul(grad, grad)` calculates the sum of squares.
		// Dividing the sum by `part_elems` gives the mean of squares.
		vec_mul(&self.grad, &self.grad).acc_to(
			&self.velocity,
			coef.velocity_decay,
			(1.0 - coef.velocity_decay) / (self.part_elems as f64),
		);
		rsqrt(&self.velocity, coef.eps).save_to(&self.velocity_recip);

		// update value
		mul(&self.momentum, &self.velocity_recip).acc_to(&self.value, 1.0, -coef.learning_rate);
	}

	pub fn save_tensors<const N: usize>(&mut self, tensors: [Tensor; N]) {
		assert!(self.stored_tensors.is_empty());
		self.stored_tensors.extend(tensors.into_iter());
	}

	pub fn load_tensors<const N: usize>(&mut self) -> [Tensor; N] {
		assert!(self.stored_tensors.len() == N);
		let mut iter = self.stored_tensors.drain(..);
		std::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() })
	}

	pub fn value(&self) -> &Tensor {
		&self.value
	}
}
