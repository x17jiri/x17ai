//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::autograd::{Autograd, AutogradNode, BackwardFn, GradientCapture};
use crate::nn::optimizer::OptimizerError;
use crate::nn::param::Param;
use crate::tensor::{Tensor, TensorOpError};

use super::Layer;

pub struct SkipConnection<Nested: Layer> {
	nested: Nested,
}

impl<Nested: Layer> SkipConnection<Nested> {
	pub fn new(nested: Nested) -> Self {
		Self { nested }
	}
}

pub fn add_residual(inp: Tensor, nested_out: Tensor) -> Result<Tensor, ErrPack<TensorOpError>> {
	let out = nested_out.reuse_or_new_like()?;
	out.assign(&inp + &nested_out)?;
	Ok(out)
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

	fn forward(&self, inp_node: AutogradNode) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		let (inp, backward_fn) = inp_node.take();
		if let Some(backward_fn) = backward_fn {
			let gradient_capture = GradientCapture::new();
			let nested_gradient = gradient_capture.storage();
			let nested_inp_node = AutogradNode::new(inp.clone(), Some(gradient_capture));
			let (nested_out, nested_fn) = self.nested.forward(nested_inp_node)?.take();
			let out = add_residual(inp, nested_out)?;
			Ok(AutogradNode::new(
				out,
				Some(Box::new(SkipConnectionBackwardFn {
					nested_fn: nested_fn.unwrap(),
					nested_gradient,
					backward_fn,
				})),
			))
		} else {
			let nested_inp_node = AutogradNode::new(inp.clone(), None);
			let (nested_out, nested_fn) = self.nested.forward(nested_inp_node)?.take();
			let out = add_residual(inp, nested_out)?;
			Ok(AutogradNode::new(out, nested_fn))
		}
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.nested.randomize()
	}
}

pub struct SkipConnectionBackwardFn {
	pub nested_fn: Box<dyn BackwardFn>,
	pub nested_gradient: Rc<RefCell<Option<Tensor>>>,
	pub backward_fn: Box<dyn BackwardFn>,
}

impl BackwardFn for SkipConnectionBackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		Autograd::run(Some(self.nested_fn), d_out.clone())?;
		if let Some(d_nested) = self.nested_gradient.borrow_mut().take() {
			let d_inp = add_residual(d_out, d_nested)?;
			autograd.set_grad(self.backward_fn, d_inp);
		}
		Ok(())
	}
}
