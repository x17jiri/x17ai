//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::nn::model_context::ModelContext;
use crate::nn::param::Param;
use crate::tensor::math::{self, Scalable, col, mat};
use crate::tensor::{self, DType, Tensor, TensorOpError};
use crate::util::LossyInto;

use super::{EvalContext, Layer};

//--------------------------------------------------------------------------------------------------

/// Linear Layer transforming inputs to outputs
///
/// This is basically a thin wrapper around a matrix multiplication.
/// It does not include a bias term.
///
///     input: [..., inputs]
///     output: [..., outputs]
pub struct Linear {
	input_shape: [usize; 1],
	output_shape: [usize; 1],

	pub(crate) weights: Rc<RefCell<Param>>,

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

	/*
	fn calc_d_weights(&self, d_out: Tensor, inp: Tensor) {
		if d_out.ndim() <= 2 {
			let (d_o, i) = if d_out.ndim() == 2 {
				(matrix(&d_out).T(), matrix(&inp))
			} else {
				(col_matrix(&d_out), row_matrix(&inp))
			};

			let d_w = tensor::math::mm(d_o, i).scale(1.0);

			let mut weights = self.weights.borrow_mut();
			weights.update_grad(|grad, already_have_grad| {
				if already_have_grad {
					cold_path();
					d_w.acc_to(matrix(grad), 1.0, 1.0);
				} else {
					d_w.save_to(matrix(grad));
				}
			});
		} else {
			cold_path();
			todo!("merge batch dimensions");
		}
	}

	fn calc_d_inp(&self, d_out: &Tensor) -> Tensor {
		// [... , outputs] -> [... , inputs]
		let d_inp = d_out.new_replace_tail(1, &self.input_shape);
		let d_i = col_matrix(&d_inp);

		let d_o = col_matrix(d_out);

		let w = self.weights.borrow();
		let w = matrix(w.value());
		tensor::math::mm(w.T(), d_o).scale(self.backward_scale).save_to(d_i);
		d_inp
	}
	*/
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

	#[inline(never)]
	fn forward(
		&self,
		inp: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<TensorOpError>> {
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

	fn backward(
		&self,
		d_out: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<tensor::TensorOpError>> {
		/*let [inp] = ctx.tensors.get();
		let d_inp = self.calc_d_inp(&d_out);
		self.calc_d_weights(d_out, inp);
		d_inp*/
		todo!();
	}

	fn backward_finish(
		&self,
		d_out: Tensor,
		ctx: &mut EvalContext,
	) -> Result<(), ErrPack<tensor::TensorOpError>> {
		/*let [inp] = ctx.tensors.get();
		self.calc_d_weights(d_out, inp);*/
		todo!();
	}
}

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
	) -> MultiheadLinear {
		let mut linear = Linear::new(inputs, heads * outputs, dtype, ctx);
		linear.weights.borrow_mut().partition(heads, inputs * outputs);

		// TODO - should we change the backward scale?
		linear.backward_scale = 1.0 / (outputs as f64).sqrt();

		MultiheadLinear { linear, output_shape: [heads, outputs] }
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
		self.linear.collect_named_params(format!("{}.linear", prefix).as_str(), f);
	}

	#[inline(never)]
	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		let out = self.linear.forward(inp, ctx);

		// [..., heads * outputs] -> [..., heads, outputs]
		out.reshape_last_dim(self.output_shape)
	}

	fn randomize(&mut self) {
		self.linear.randomize();
	}

	fn backward_finish(&self, d_out: Tensor, ctx: &mut EvalContext) {
		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.merge_dims::<2>();

		self.linear.backward_finish(d_out, ctx)
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.merge_dims::<2>();

		self.linear.backward(d_out, ctx)
	}
}

//--------------------------------------------------------------------------------------------------
*/
