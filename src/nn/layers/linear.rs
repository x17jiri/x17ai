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
use crate::autograd::AutogradNode;
use crate::nn::model_context::ModelContext;
use crate::nn::param::Param;
use crate::tensor::math::{self, Scalable, col, mat, row};
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
	backward_scale: f64, // TODO - do we need to scale the backward pass?
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
			backward_scale: 1.0 / outputs.lossy_into().sqrt(),
		})
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

		let w = self.weights.borrow();
		let w = mat(w.value())?;
		let i = col(&inp)?;
		let o = col(&out)?;
		o.assign((w * i).scale(self.forward_scale))?;

		if ctx.is_training() {
			ctx.tensors.set([inp]);
		}

		Ok(out)
	}

	fn randomize(&mut self) -> std::result::Result<(), ErrPack<tensor::TensorOpError>> {
		let w = self.weights.borrow();
		w.value().assign(math::randn_clamped())
	}
}

/*
	fn backward(
		&self,
		d_out: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<OptimizerError>> {
		let [inp] = ctx.tensors.get();

		let mut weights = self.weights.borrow_mut();
		let w = mat(weights.value())?;

		// d_inp
		let d_o = col(&d_out)?;
		let d_i = (w.T() * d_o).scale(self.backward_scale);
		// [... , outputs] -> [... , inputs]
		let d_inp = d_out.new_replace_tail(1, &self.input_shape)?;
		col(&d_inp)?.assign(d_i)?;

		// d_w
		let i = row(&inp)?;
		let d_w = (d_o * i).scale(1.0);
		weights.update_grad(|grad, already_have_grad| {
			let grad = mat(grad)?;
			if already_have_grad {
				cold_path();
				todo!("Linear layer backward with existing gradient is not implemented yet");
			}
			grad.clear_acc(d_w)
		})?;

		Ok(d_inp)
	}
}*/

//--------------------------------------------------------------------------------------------------
/*
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
	) -> Result<Self, ErrPack<OptimizerError>> {
		let linear = Linear::new(inputs, heads * outputs, dtype, ctx)?;
		linear.weights.borrow_mut().partition(heads, inputs * outputs)?;

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
	fn forward(
		&self,
		inp: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let out = self.linear.forward(inp, ctx)?;

		// [..., heads * outputs] -> [..., heads, outputs]
		Ok(out.reshape_last_dim(self.output_shape)?)
	}

	fn randomize(&mut self) -> std::result::Result<(), ErrPack<tensor::TensorOpError>> {
		self.linear.randomize()
	}

	fn backward(
		&self,
		d_out: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<OptimizerError>> {
		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.merge_dims::<2>()?;
		self.linear.backward(d_out, ctx)
	}
}
*/
//--------------------------------------------------------------------------------------------------
