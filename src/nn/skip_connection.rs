use std::cell::RefCell;
use std::rc::Rc;

use crate::eval_context::EvalContext;
use crate::expr::{self, Accumulable, Savable};
use crate::nn::Layer;
use crate::param::Param;
use crate::tensor::{Tensor, TensorSize};

pub struct SkipConnection<Nested: Layer> {
	nested: Nested,
}

impl<Nested: Layer> SkipConnection<Nested> {
	pub fn new(nested: Nested) -> Self {
		Self { nested }
	}

	fn add_residual(&self, inp: Tensor, nested_out: Tensor) -> Tensor {
		// try to reuse `nested_out` for `out` if possible
		let (out, out_ref);
		if inp.owns_buffer() {
			out = None;
			out_ref = &nested_out;
		} else {
			out = Some(nested_out.new_empty_like());
			out_ref = out.as_ref().unwrap();
		}

		expr::add(&inp, &nested_out).save_to(out_ref);

		out.unwrap_or(nested_out)
	}
}

impl<Nested: Layer> Layer for SkipConnection<Nested> {
	fn input_shape(&self) -> &[TensorSize] {
		self.nested.input_shape()
	}

	fn output_shape(&self) -> &[TensorSize] {
		self.nested.output_shape()
	}

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.nested.collect_params(f);
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.nested.collect_named_params(prefix, f);
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		let nested_out = self.nested.forward(inp.clone(), ctx);
		self.add_residual(inp, nested_out)
	}

	fn randomize(&mut self) {
		self.nested.randomize();
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		let nested_out = self.nested.backward(d_out.clone(), ctx);
		self.add_residual(d_out, nested_out)
	}

	fn backward_finish(&self, d_out: Tensor, ctx: &mut EvalContext) {
		self.nested.backward_finish(d_out, ctx);
	}
}
