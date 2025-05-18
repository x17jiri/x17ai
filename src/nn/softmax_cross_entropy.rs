use std::cell::RefCell;
use std::rc::Rc;

use crate::eval_context::EvalContext;
use crate::expr::{self, Accumulable, Savable};
use crate::nn::{BackpropLayer, Layer, LossLayer, Softmax};
use crate::param::Param;
use crate::tensor::{Tensor, TensorSize};

pub struct SoftmaxCrossEntropy {
	softmax: Softmax,
}

impl SoftmaxCrossEntropy {
	pub fn new(classes: TensorSize) -> SoftmaxCrossEntropy {
		SoftmaxCrossEntropy { softmax: Softmax::new(classes) }
	}
}

impl Layer for SoftmaxCrossEntropy {
	fn randomize(&mut self) {
		self.softmax.randomize();
	}

	fn input_shape(&self) -> &[TensorSize] {
		self.softmax.input_shape()
	}

	fn output_shape(&self) -> &[TensorSize] {
		self.softmax.output_shape()
	}

	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.softmax.collect_params(_f);
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.softmax.collect_named_params(_prefix, _f);
	}

	fn forward(&self, inp: Tensor, _ctx: &mut EvalContext) -> Tensor {
		self.softmax.forward(inp, _ctx)
	}
}

impl BackpropLayer for SoftmaxCrossEntropy {
	fn init_optimizer(&self) {
		self.softmax.init_optimizer();
	}

	fn zero_grad(&self) {
		self.softmax.zero_grad();
	}

	fn step(&self, opt_coef: &crate::optimizer::OptCoef) {
		self.softmax.step(opt_coef);
	}

	fn backward(&self, d_out: Tensor, _ctx: &mut EvalContext) -> Tensor {
		self.softmax.backward(d_out, _ctx)
	}

	fn backward_first(&self, d_out: Tensor, _ctx: &mut EvalContext) {
		self.softmax.backward_first(d_out, _ctx)
	}

	fn as_loss_layer(&self) -> Option<&dyn LossLayer> {
		Some(self)
	}
}

impl LossLayer for SoftmaxCrossEntropy {
	fn backward_last(&self, out: Tensor, expected_out: Tensor, _ctx: &mut EvalContext) -> Tensor {
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
