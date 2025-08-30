//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::{self, AutogradNode, BackwardFn};
use crate::nn::param::Param;
use crate::rng::Rng;
use crate::tensor::{Tensor, TensorOpError};
use crate::{ErrPack, custom_kernel};

use super::Layer;

pub struct SwiGLU {
	input_shape: [usize; 2],
	output_shape: [usize; 1],
}

impl SwiGLU {
	pub fn new(n_outputs: usize) -> Self {
		Self {
			input_shape: [2, n_outputs],
			output_shape: [n_outputs],
		}
	}
}

impl Layer for SwiGLU {
	fn input_shape(&self) -> &[usize] {
		&self.input_shape
	}

	fn output_shape(&self) -> &[usize] {
		&self.output_shape
	}

	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn forward(&self, inp_node: AutogradNode) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		let (inp, inp_backward) = inp_node.take();
		let out = inp.new_replace_tail(2, &self.output_shape)?;
		let lin = inp.select(-2, 0)?;
		let gate = inp.select(-2, 1)?;

		out.assign(custom_kernel!(
			[lin: &lin, gate: &gate], (ONE: 1.0), {
				lin * gate * (ONE + (-gate).exp()).recip()
			}
		))?;

		let backward_fn = inp_backward.map(|inp_backward| {
			Box::new(SwiGLUBackwardFn {
				lin,
				gate,
				inp_backward,
				output_shape: self.output_shape,
			}) as Box<dyn BackwardFn>
		});

		Ok(AutogradNode::new(out, backward_fn))
	}

	fn randomize(&mut self, _rng: &mut Rng) -> Result<(), ErrPack<TensorOpError>> {
		// no parameters to randomize
		Ok(())
	}
}

pub struct SwiGLUBackwardFn {
	pub lin: Tensor,
	pub gate: Tensor,
	pub inp_backward: Box<dyn BackwardFn>,
	pub output_shape: [usize; 1],
}

impl BackwardFn for SwiGLUBackwardFn {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { lin, gate, inp_backward, output_shape } = Box::into_inner(self);

		let input_shape = [2, output_shape[0]];
		let d_inp = d_out.new_replace_tail(1, &input_shape)?;
		let d_lin = d_inp.select(-2, 0)?;
		let d_gate = d_inp.select(-2, 1)?;

		d_lin.assign(custom_kernel!(
			[d_out: &d_out, gate: &gate], (ONE: 1.0), {
				d_out * gate * (ONE + (-gate).exp()).recip()
			}
		))?;

		d_gate.assign(custom_kernel!(
			[d_out: &d_out, lin: &lin, gate: &gate], (ONE: 1.0), {
				let sigmoid = (ONE + (-gate).exp()).recip();
				let swish = gate * sigmoid;
				let result = d_out * lin * (sigmoid + swish - (sigmoid * swish));
				result
			}
		))?;

		queue.add(inp_backward, d_inp);
		Ok(())
	}
}
