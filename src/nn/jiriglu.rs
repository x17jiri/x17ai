use std::cell::RefCell;
use std::rc::Rc;

use crate::eval_context::EvalContext;
use crate::expr::{self, Accumulable, Savable};
use crate::nn::Layer;
use crate::param::Param;
use crate::tensor::{Tensor, TensorSize};

pub struct JiriGLU {
	input_shape: [TensorSize; 2],
	output_shape: [TensorSize; 1],
}

impl JiriGLU {
	pub fn new(n_outputs: TensorSize) -> JiriGLU {
		JiriGLU {
			input_shape: [2, n_outputs],
			output_shape: [n_outputs],
		}
	}
}

impl Layer for JiriGLU {
	fn input_shape(&self) -> &[TensorSize] {
		&self.input_shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.output_shape
	}

	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		let out = inp.new_replace_tail(2, &self.output_shape);
		let lin = inp.clone().slice(-2, 0..1);
		let gate = inp.slice(-2, 1..2);

		expr::jiri_glu(&lin, &gate).save_to(&out);

		if ctx.is_training() {
			ctx.tensors.set([lin, gate]);
		}

		out
	}

	fn randomize(&mut self) {
		// no parameters to randomize
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		let [lin, gate] = ctx.tensors.get();

		let d_inp = d_out.new_replace_tail(1, &self.input_shape);
		let d_lin = d_inp.clone().slice(-2, 0..1);
		let d_gate = d_inp.clone().slice(-2, 1..2);

		expr::jiri_glu_backward(&d_out, &lin, &gate).save_to(&d_lin, &d_gate);

		d_inp
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}
