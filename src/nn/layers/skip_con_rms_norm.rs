//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::hint::cold_path;
use std::rc::Rc;

use crate::ErrPack;
use crate::autograd::{self, AutogradNode};
use crate::nn::layers::rms_norm::BackwardRMSNormBackwardFn;
use crate::nn::param::Param;
use crate::tensor::TensorOpError;
use crate::tensor::math::RMSCalc;

use super::Layer;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum NormPosition {
	/// ```
	///         +------+    +--------+    +-----+
	/// ---+--->| norm |--->| nested |--->| add |---->
	///    |    +------+    +--------+    +-----+
	///    |                                 ^
	///    |                                 |
	///    +---------------------------------+
	/// ```
	Inside,

	/// ```
	///     +------+        +--------+    +-----+
	/// --->| norm |---+--->| nested |--->| add |---->
	///     +------+   |    +--------+    +-----+
	///                |                     ^
	///                |                     |
	///                +---------------------+
	/// ```
	Outside,
}

pub struct SkipConRMSNorm<Nested: Layer> {
	nested: Nested,
	calc: RMSCalc,
	norm_pos: NormPosition,
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
				norm_pos: NormPosition::Inside,
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
		let (inp, inp_fn) = inp_node.take();
		let [nested_inp_fn, residual_fn] = autograd::split::split_fn(inp_fn);

		let inp_magn_recip = inp.new_replace_tail(1, &[1])?;
		inp_magn_recip.assign(self.calc.rsqrt_mean_square(&inp))?;
		let rms_norm = inp.new_empty_like()?;
		rms_norm.assign(&inp * &inp_magn_recip)?;

		let residual = match self.norm_pos {
			NormPosition::Inside => AutogradNode::new(inp, residual_fn),
			NormPosition::Outside => AutogradNode::new(rms_norm.clone(), residual_fn),
		};

		let nested_out;
		if let Some(nested_inp_fn) = nested_inp_fn {
			let required_magn = inp_magn_recip.new_empty_like()?;
			let grad_rescale_fn = Box::new(BackwardRMSNormBackwardFn {
				calc: self.calc,
				inp_backward: nested_inp_fn,
				required_magn: Some(required_magn.clone()),
			});

			let nested_inp = AutogradNode::new(rms_norm, Some(grad_rescale_fn));
			nested_out = self.nested.forward(nested_inp)?;

			required_magn.assign(self.calc.sqrt_mean_square(&nested_out.value))?;
			if self.norm_pos == NormPosition::Inside {
				required_magn.assign(&required_magn * &inp_magn_recip)?;
			}
		} else {
			let nested_inp = AutogradNode::new(rms_norm, None);
			nested_out = self.nested.forward(nested_inp)?;
		}

		let out = autograd::add::add(nested_out, residual)?;

		Ok(out.map_backward_fn(|f| {
			Box::new(BackwardRMSNormBackwardFn {
				calc: self.calc,
				inp_backward: f,
				required_magn: None,
			})
		}))
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.nested.randomize()
	}
}
