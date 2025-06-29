//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::AutogradNode;
use crate::nn::param::Param;
use crate::tensor::TensorOpError;
use crate::{ErrPack, autograd};

use super::Layer;

pub struct SkipConnection<Nested: Layer> {
	nested: Nested,
	nested_weight: f64,
	skip_weight: f64,
}

impl<Nested: Layer> SkipConnection<Nested> {
	pub fn new(nested: Nested) -> Self {
		Self {
			nested,
			nested_weight: 1.0,
			skip_weight: 1.0,
		}
	}

	/// We can weigh the gradients during the backward pass.
	/// - `nested_weight` is the weight of the gradient from the nested layer
	/// - `skip_weight` is the weight of the gradient from the skip connection
	pub fn set_weights(&mut self, nested_weight: f64, skip_weight: f64) -> &mut Self {
		self.nested_weight = nested_weight;
		self.skip_weight = skip_weight;
		self
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

	fn forward(&self, inp_node: AutogradNode) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		let [a, b] = autograd::split::split(inp_node, [self.nested_weight, self.skip_weight]);
		let nested_out = self.nested.forward(a)?;
		autograd::add::add(nested_out, b)
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.nested.randomize()
	}
}
