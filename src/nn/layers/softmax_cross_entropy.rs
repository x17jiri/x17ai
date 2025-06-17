//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::math::{LnClamped, sum_all};
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

use super::{Layer, LossFunction, Softmax, SoftmaxGradientMode};

pub struct SoftmaxCrossEntropy {
	softmax: Softmax,
}

impl SoftmaxCrossEntropy {
	pub fn new(n_inputs: usize) -> Self {
		let mut softmax = Softmax::new(n_inputs);
		// Set the gradient mode to StraightThrough so that the forward pass
		// doesn't cache its output
		softmax.set_gradient_mode(SoftmaxGradientMode::StraightThrough);
		Self { softmax }
	}
}

impl Layer for SoftmaxCrossEntropy {
	fn input_shape(&self) -> &[usize] {
		self.softmax.input_shape()
	}

	fn output_shape(&self) -> &[usize] {
		self.softmax.output_shape()
	}

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.softmax.collect_params(f);
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.softmax.collect_named_params(prefix, f);
	}

	fn forward(
		&self,
		inp: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		self.softmax.forward(inp, ctx)
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.softmax.randomize()
	}

	fn backward(
		&self,
		d_out: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		self.softmax.backward(d_out, ctx)
	}

	fn backward_finish(
		&self,
		d_out: Tensor,
		ctx: &mut EvalContext,
	) -> Result<(), ErrPack<TensorOpError>> {
		self.softmax.backward_finish(d_out, ctx)
	}

	fn as_loss_function(&self) -> Option<&dyn LossFunction> {
		Some(self)
	}
}

impl LossFunction for SoftmaxCrossEntropy {
	fn backward_start(
		&self,
		out: Tensor,
		expected_out: Tensor,
		_ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let d_inp = out.new_empty_like()?;
		d_inp.assign(&out - &expected_out)?;
		Ok(d_inp)
	}

	fn loss(&self, out: Tensor, expected_out: Tensor) -> Result<f64, ErrPack<TensorOpError>> {
		// Remove the feature dimension (-1). All other dimensions are batch dimensions,
		// so `elems()` will give us the batch size.
		let batch_size = out.select(-1, 0)?.elems();

		let tmp = out.new_empty_like()?;
		tmp.assign(out.ln_clamped())?;
		tmp.assign(&tmp * &expected_out)?;

		Ok(sum_all(&tmp)? / -batch_size.lossy_into())
	}
}
