//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::autograd::{self, AutogradNode, BackwardFn};
use crate::nn::model_context::ModelContext;
use crate::nn::optimizer::CurrentGradValue;
use crate::nn::param::Param;
use crate::tensor::math::{self, col, mat, row};
use crate::tensor::{self, DType, Tensor, TensorOpError};
use crate::util::LossyInto;

use super::Layer;

//--------------------------------------------------------------------------------------------------

/// Linear Layer transforming inputs to outputs
///
/// This is basically a thin wrapper around matrix multiplication.
/// It does not include a bias term.
///
///     input: [..., inputs]
///     output: [..., outputs]
pub struct Linear {
	input_shape: [usize; 1],
	output_shape: [usize; 1],

	weights: Rc<RefCell<Param>>,

	forward_scale: f64,
	backward_scale: f64,
}

impl Linear {
	pub fn new(
		inputs: usize,
		outputs: usize,
		dtype: DType,
		ctx: &mut ModelContext,
	) -> Result<Self, ErrPack<TensorOpError>> {
		Ok(Self {
			input_shape: [inputs],
			output_shape: [outputs],

			weights: ctx.new_param(&[outputs, inputs], dtype)?,

			forward_scale: 1.0 / inputs.lossy_into().sqrt(),
			backward_scale: 1.0,
		})
	}

	pub fn weights(&self) -> Rc<RefCell<Param>> {
		self.weights.clone()
	}
}

impl Layer for Linear {
	fn input_shape(&self) -> &[usize] {
		&self.input_shape
	}

	fn output_shape(&self) -> &[usize] {
		&self.output_shape
	}

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		f(self.weights.clone());
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		f(format!("{prefix}.weights"), self.weights.clone());
	}

	fn forward(&self, inp_node: AutogradNode) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		let (inp, inp_backward) = inp_node.take();

		// [..., inputs] -> [..., outputs]
		let out = inp.new_replace_tail(1, &self.output_shape)?;

		let weights = self.weights.borrow();
		let w = mat(weights.value())?;
		let i = col(&inp)?;
		let o = col(&out)?;
		o.assign((w * i).scale(self.forward_scale))?;

		let backward_fn = if inp_backward.is_some() || weights.requires_grad() {
			Some(Box::new(LinearBackwardFn {
				weights: self.weights.clone(),
				inp: if weights.requires_grad() { Some(inp) } else { None },
				inp_backward,
				backward_scale: self.backward_scale,
				input_shape: self.input_shape,
			}) as Box<dyn BackwardFn>)
		} else {
			None
		};

		Ok(AutogradNode::new(out, backward_fn))
	}

	fn randomize(&mut self) -> std::result::Result<(), ErrPack<tensor::TensorOpError>> {
		let w = self.weights.borrow();
		w.value().assign(math::randn_clamped())
	}
}

pub struct LinearBackwardFn {
	weights: Rc<RefCell<Param>>,

	/// should be `Some` if we should compute `d_w`
	inp: Option<Tensor>,

	/// should be `Some` if we should compute `d_inp`
	inp_backward: Option<Box<dyn BackwardFn>>,

	backward_scale: f64,

	input_shape: [usize; 1],
}

impl BackwardFn for LinearBackwardFn {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self {
			weights,
			inp,
			inp_backward,
			backward_scale,
			input_shape,
		} = Box::into_inner(self);

		let mut weights = weights.borrow_mut();
		let d_o = col(&d_out)?;

		// d_w
		if let Some(inp) = inp
			&& weights.requires_grad()
		{
			let i = row(&inp)?;
			let d_w = (d_o * i).scale(1.0);
			weights.update_grad(|current_grad| match current_grad {
				CurrentGradValue::Uninit(tensor) => {
					let grad = mat(tensor)?;
					grad.clear_acc(d_w)
				},
				CurrentGradValue::Value(_tensor) => {
					todo!("Linear layer backward with existing gradient is not implemented yet");
				},
			})?;
		}

		// d_inp
		if let Some(inp_backward) = inp_backward {
			let w = mat(weights.value())?;
			let d_i = (w.T() * d_o).scale(backward_scale);
			// [... , outputs] -> [... , inputs]
			let d_inp = d_out.new_replace_tail(1, &input_shape)?;
			col(&d_inp)?.assign(d_i)?;
			queue.add(inp_backward, d_inp);
		}

		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

/// Multihead Linear Layer.
///
/// It works similarly to linear layer, but the same inputs are transformed
/// by multiple heads.
///
///     input: [..., inputs]
///     output: [..., head, outputs]
pub struct MultiheadLinear {
	pub(crate) linear: Linear,
	output_shape: [usize; 2],
}

impl MultiheadLinear {
	pub fn new(
		inputs: usize,
		outputs: usize,
		heads: usize,
		dtype: DType,
		ctx: &mut ModelContext,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let linear = Linear::new(inputs, heads * outputs, dtype, ctx)?;
		linear.weights.borrow_mut().partition(heads, inputs * outputs).unwrap(); // TODO: unwrap

		// TODO - should we change the backward scale?
		//linear.backward_scale = 1.0 / (outputs as f64).sqrt();

		Ok(Self { linear, output_shape: [heads, outputs] })
	}
}

impl Layer for MultiheadLinear {
	fn input_shape(&self) -> &[usize] {
		&self.linear.input_shape
	}

	fn output_shape(&self) -> &[usize] {
		&self.output_shape
	}

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.linear.collect_params(f);
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.linear.collect_named_params(format!("{prefix}.linear").as_str(), f);
	}

	#[inline(never)]
	fn forward(&self, inp_node: AutogradNode) -> Result<AutogradNode, ErrPack<TensorOpError>> {
		let out_node = self.linear.forward(inp_node)?;
		let (out, backward_fn) = out_node.take();

		// [..., heads * outputs] -> [..., heads, outputs]
		let out = out.reshape_last_dim(self.output_shape)?;

		let backward_fn = backward_fn.map(|inp_backward| {
			Box::new(MultiheadLinearBackwardFn { inp_backward }) as Box<dyn BackwardFn>
		});

		Ok(AutogradNode::new(out, backward_fn))
	}

	fn randomize(&mut self) -> std::result::Result<(), ErrPack<tensor::TensorOpError>> {
		self.linear.randomize()
	}
}

pub struct MultiheadLinearBackwardFn {
	inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for MultiheadLinearBackwardFn {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { inp_backward } = Box::into_inner(self);
		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.merge_dims::<2>()?;
		queue.add(inp_backward, d_out);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
