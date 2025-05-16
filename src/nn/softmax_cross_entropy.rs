use std::cell::RefCell;
use std::rc::Rc;

use crate::eval_context::EvalContext;
use crate::expr::{self, Accumulable, Savable};
use crate::nn::{Layer, LossLayer};
use crate::param::Param;
use crate::tensor::{Tensor, TensorSize};

pub struct SoftmaxCrossEntropy {
	shape: [TensorSize; 1],
}

impl SoftmaxCrossEntropy {
	pub fn new(classes: TensorSize) -> SoftmaxCrossEntropy {
		SoftmaxCrossEntropy { shape: [classes] }
	}
}

impl Layer for SoftmaxCrossEntropy {
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

impl LossLayer for SoftmaxCrossEntropy {
	fn backward(&self, out: Tensor, expected_out: Tensor, _ctx: &mut EvalContext) -> Tensor {
		let d_inp = out.new_empty_like();
		expr::sub(&out, &expected_out).save_to(&d_inp);
		d_inp
	}

	fn loss(&self, out: Tensor, expected_out: Tensor) -> f64 {
		let tmp = out.new_empty_like();
		expr::log_clamped(&out).save_to(&tmp);
		expr::mul(&tmp, &expected_out).save_to(&tmp);

		expr::sum_all(&tmp) / -(tmp.batch_size(1) as f64)
	}
}
