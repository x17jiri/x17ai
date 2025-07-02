//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::hint::cold_path;
use std::rc::Rc;

use crate::autograd::{AutogradNode, BackwardFn};
use crate::nn::layers::rms_norm::RMSCalc;
use crate::nn::param::Param;
use crate::tensor::{Tensor, TensorOpError};
use crate::{ErrPack, autograd};

use super::Layer;

pub struct SkipConRMSNorm<Nested: Layer> {
	nested: Nested,
	calc: RMSCalc,
}

impl<Nested: Layer> SkipConRMSNorm<Nested> {
	pub fn new(nested: Nested, eps: f64) -> Option<Self> {
		let input_shape = nested.input_shape();
		let output_shape = nested.output_shape();
		if input_shape == output_shape
			&& let Some(&n_inputs) = input_shape.last()
		{
			Some(Self {
				nested,
				calc: RMSCalc::new(n_inputs, eps),
			})
		} else {
			cold_path();
			None
		}
	}
}

impl<Nested: Layer> Layer for SkipConRMSNorm<Nested> {
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

		let inp_rms = self.calc.root_mean_square(&inp)?;
		let value = inp.new_empty_like()?;
		value.assign(&inp * &inp_rms)?;

		if let Some(backward_fn) = backward_fn {
			let rc_grad = Rc::new(RefCell::new(None));
			let merge_fn = Box::new(SkipConRMSNormBackwardFn_Merge { rc_grad, backward_fn })
				as Box<dyn BackwardFn>;
			let nested_out = self.nested.forward(AutogradNode::new(value, Some(merge_fn)))?;
			// TODO
			1
		} else {
			let nested_out = self.nested.forward(AutogradNode::new(value, None))?;
			autograd::add::add(nested_out, AutogradNode::new(inp, None))
		}
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.nested.randomize()
	}
}

pub struct SkipConRMSNormBackwardFn_Split {
	rc_grad: Rc<RefCell<Option<Tensor>>>,
}

pub struct SkipConRMSNormBackwardFn_Merge {
	rc_grad: Rc<RefCell<Option<Tensor>>>,
	backward_fn: Box<dyn BackwardFn>,
}

impl BackwardFn for SkipConRMSNormBackwardFn_Split {
	//
}

impl BackwardFn for SkipConRMSNormBackwardFn_Merge {
	//
}
