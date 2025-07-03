//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::f128::consts::E;
use std::hint::cold_path;
use std::rc::Rc;

use crate::autograd::{Autograd, AutogradNode, BackwardFn};
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
		let (mut inp, backward_fn) = inp_node.take();

		let inp_scale = self.calc.root_mean_square(&inp)?;
		let nested_inp = inp.new_empty_like()?;
		nested_inp.assign(&inp * &inp_scale)?;

		let (mut nested_out, merge_fn) = if let Some(backward_fn) = backward_fn {
			let rc_grad = Rc::new(RefCell::new(None));

			let split_fn =
				Box::new(SkipConRMSNormBackwardFn_Split { rc_grad: rc_grad.clone(), backward_fn })
					as Box<dyn BackwardFn>;

			let nested_out_node =
				self.nested.forward(AutogradNode::new(nested_inp, Some(split_fn)))?;
			let (nested_out, nested_fn) = nested_out_node.take();

			let merge_fn = if let Some(nested_fn) = nested_fn {
				let nested_scale = self.calc.root_mean_square(&nested_out)?;
				// TODO: backward_scale = inp_scale / nested_scale
				// backward_scale will be used to multiply the gradient before propagating
				// to the nested layer
				Some(Box::new(SkipConRMSNormBackwardFn_Merge { rc_grad, nested_fn })
					as Box<dyn BackwardFn>)
			} else {
				None
			};

			(nested_out, merge_fn)
		} else {
			let nested_node = self.nested.forward(AutogradNode::new(nested_inp, None))?;
			nested_node.take()
		};

		if !inp.owns_buffer() {
			std::mem::swap(&mut inp, &mut nested_out);
		}
		let out = if inp.owns_buffer() { inp.clone() } else { inp.new_empty_like()? };
		out.assign(&inp + &nested_out)?;

		Ok(autograd::AutogradNode::new(out, merge_fn))
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.nested.randomize()
	}
}

pub struct SkipConRMSNormBackwardFn_Split {
	rc_grad: Rc<RefCell<Option<Tensor>>>,
	backward_fn: Box<dyn BackwardFn>,
}

pub struct SkipConRMSNormBackwardFn_Merge {
	rc_grad: Rc<RefCell<Option<Tensor>>>,
	nested_fn: Box<dyn BackwardFn>,
}

impl BackwardFn for SkipConRMSNormBackwardFn_Split {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
	}
}

impl BackwardFn for SkipConRMSNormBackwardFn_Merge {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
	}
}
