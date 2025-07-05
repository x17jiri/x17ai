//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::hint::cold_path;
use std::rc::Rc;

use crate::autograd::{Autograd, AutogradNode, BackwardFn};
use crate::nn::layers::rms_norm::RMSCalc;
use crate::nn::param::Param;
use crate::tensor::{Tensor, TensorOpError};
use crate::{ErrPack, autograd};

use super::Layer;

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

	fn forward_norm_inside(
		&self,
		inp_node: AutogradNode,
	) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		let (mut inp, inp_fn) = inp_node.take();
		let [nested_inp_fn, residual_fn] = autograd::split::split_fn(inp_fn);

		let rescale = self.calc.rsqrt_mean_square(&inp)?;
		let nested_inp = inp.new_empty_like()?;
		nested_inp.assign(&inp * &rescale)?;

		let t = self.nested.forward(AutogradNode::new(nested_inp, nested_inp_fn))?;
		let (mut nested_out, nested_out_fn) = t.take();
		let out_fn = if let Some(residual_fn) = residual_fn
			&& let Some(nested_out_fn) = nested_out_fn
		{
			// backward_scale will be used to multiply the gradient before propagating
			// to the nested layer
			let backward_scale = self.calc.sqrt_mean_square(&nested_out)?;
			backward_scale.assign(&backward_scale * &rescale)?;
			Some(Box::new(SkipConRMSNormBackwardFn {
				residual_fn,
				nested_fn: nested_out_fn,
				backward_rescale: backward_scale,
				rms_calc: self.calc,
			}) as Box<dyn BackwardFn>)
		} else {
			nested_out_fn
		};

		if !inp.owns_buffer() {
			std::mem::swap(&mut inp, &mut nested_out);
		}
		let out = if inp.owns_buffer() { inp.clone() } else { inp.new_empty_like()? };
		out.assign(&inp + &nested_out)?;

		Ok(autograd::AutogradNode::new(out, out_fn))
	}

	fn forward_norm_outside(
		&self,
		_inp_node: AutogradNode,
	) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		todo!("SkipConRMSNorm::forward_norm_outside is not implemented yet");
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
		match self.norm_pos {
			NormPosition::Inside => self.forward_norm_inside(inp_node),
			NormPosition::Outside => self.forward_norm_outside(inp_node),
		}
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.nested.randomize()
	}
}

pub struct SkipConRMSNormBackwardFn {
	residual_fn: Box<dyn BackwardFn>,
	nested_fn: Box<dyn BackwardFn>,
	backward_rescale: Tensor,
	rms_calc: RMSCalc,
}

impl BackwardFn for SkipConRMSNormBackwardFn {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self {
			residual_fn,
			nested_fn,
			backward_rescale,
			rms_calc,
		} = Box::into_inner(self);

		let grad_rescale = rms_calc.rsqrt_mean_square(&d_out)?;
		debug_assert!(grad_rescale.owns_buffer());
		debug_assert!(backward_rescale.owns_buffer());
		backward_rescale.assign(&backward_rescale * &grad_rescale)?;

		// We need to calculate d_nested before the next step updates d_out
		let d_nested = d_out.new_empty_like()?;
		d_nested.assign(&d_out * &backward_rescale)?;
		autograd.set_grad(nested_fn, d_nested);

		let d_residual = d_out.reuse_or_new_like()?;
		d_residual.assign(&d_out * &grad_rescale)?;
		residual_fn.run(d_residual, autograd)
	}
}
