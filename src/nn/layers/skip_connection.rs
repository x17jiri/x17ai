//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::Result;
use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::Tensor;

use super::Layer;

pub struct SkipConnection<Nested: Layer> {
	nested: Nested,
}

impl<Nested: Layer> SkipConnection<Nested> {
	pub fn new(nested: Nested) -> Self {
		Self { nested }
	}

	pub fn add_residual(&self, inp: Tensor, nested_out: Tensor) -> Result<Tensor> {
		let out = nested_out.reuse_or_new_like()?;
		out.assign(&inp + &nested_out)?;
		Ok(out)
	}
}

impl<Nested: Layer> Layer for SkipConnection<Nested> {
	fn input_shape(&self) -> &[usize] {
		self.nested.input_shape()
	}

	fn output_shape(&self) -> &[usize] {
		self.nested.output_shape()
	}

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.nested.collect_params(f);
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.nested.collect_named_params(prefix, f);
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Result<Tensor> {
		let nested_out = self.nested.forward(inp.clone(), ctx)?;
		self.add_residual(inp, nested_out)
	}

	fn randomize(&mut self) -> Result<()> {
		self.nested.randomize()
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Result<Tensor> {
		let nested_out = self.nested.backward(d_out.clone(), ctx)?;
		self.add_residual(d_out, nested_out)
	}

	fn backward_finish(&self, d_out: Tensor, ctx: &mut EvalContext) -> Result<()> {
		self.nested.backward_finish(d_out, ctx)
	}
}
