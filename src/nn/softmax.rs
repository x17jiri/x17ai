use std::cell::RefCell;
use std::rc::Rc;

use crate::eval_context::EvalContext;
use crate::expr::{self, Accumulable, Savable};
use crate::nn::{Layer, LossLayer};
use crate::param::Param;
use crate::tensor::{Tensor, TensorSize};

use super::BackpropLayer;

pub struct Softmax {
	shape: [TensorSize; 1],
}

impl Softmax {
	pub fn new(classes: TensorSize) -> Softmax {
		Softmax { shape: [classes] }
	}
}

impl Layer for Softmax {
	fn randomize(&mut self) {
		// no parameters to randomize
	}

	fn input_shape(&self) -> &[TensorSize] {
		&self.shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.shape
	}

	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn forward(&self, inp: Tensor, _ctx: &mut EvalContext) -> Tensor {
		let out = inp.new_empty_like();
		expr::softmax(&inp).save_to(&out);
		out
	}
}

impl BackpropLayer for Softmax {
	fn init_optimizer(&self) {
		// no parameters to optimize
	}

	fn zero_grad(&self) {
		// no parameters to update
	}

	fn step(&self, _opt_coef: &crate::optimizer::OptCoef) {
		// no parameters to update
	}

	fn backward(&self, d_out: Tensor, _ctx: &mut EvalContext) -> Tensor {
		// TODO
		// idea to simplify the softmax gradient:
		// - treat softmax as e^x followed by rescaling
		// - treat the rescaling as straight-through estimator,
		// . i.e., ignore it during the backward pass
		// - so just calculate the gradient of e^x
		d_out
	}

	fn backward_first(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}
