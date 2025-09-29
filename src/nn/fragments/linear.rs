//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::autograd::{self, AutogradTensor, BackwardFn};
use crate::nn::fragments::UnaryFragment;
use crate::nn::model_context::ModelContext;
use crate::nn::optimizer::CurrentGradValue;
use crate::nn::param::Param;
use crate::rng::Rng;
use crate::tensor::math::{col, mat, row};
use crate::tensor::{self, DType, Tensor, TensorOpError};
use crate::util::LossyFrom;

use super::Fragment;

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
	internal_dtype: DType,
	forward_scale: f64,
	backward_scale: f64,
}

impl Linear {
	pub fn new(
		inputs: usize,
		outputs: usize,
		dtype: DType,
		internal_dtype: DType,
		ctx: &mut ModelContext,
	) -> Result<Self, ErrPack<TensorOpError>> {
		Ok(Self {
			input_shape: [inputs],
			output_shape: [outputs],
			weights: ctx.new_param(&[outputs, inputs], dtype)?,
			internal_dtype,
			forward_scale: 1.0 / f64::lossy_from(inputs).sqrt(),
			backward_scale: 1.0,
		})
	}

	pub fn weights(&self) -> Rc<RefCell<Param>> {
		self.weights.clone()
	}
}

impl Fragment for Linear {
	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		f(self.weights.clone());
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		f(format!("{prefix}.weights"), self.weights.clone());
	}

	fn randomize(
		&mut self,
		rng: &mut Rng,
	) -> std::result::Result<(), ErrPack<tensor::TensorOpError>> {
		let w = self.weights.borrow();
		w.value().randn_(rng)
	}
}

impl UnaryFragment for Linear {
	fn forward(&self, inp_node: AutogradTensor) -> Result<AutogradTensor, ErrPack<TensorOpError>> {
		let (inp, inp_backward) = inp_node.into_parts();

		// [..., inputs] -> [..., outputs]
		let out = inp.new_replace_tail(1, &self.output_shape, inp.dtype())?;

		let weights = self.weights.borrow();
		let w = mat(weights.value())?;
		let i = col(&inp)?;
		let o = col(&out)?;
		o.assign((w * i).scale(self.forward_scale), self.internal_dtype)?;

		let backward_fn = if inp_backward.is_some() || weights.requires_grad() {
			Some(Box::new(LinearBackwardFn {
				weights: self.weights.clone(),
				inp: if weights.requires_grad() { Some(inp) } else { None },
				inp_backward,
				backward_scale: self.backward_scale,
				internal_dtype: self.internal_dtype,
				input_shape: self.input_shape,
			}) as Box<dyn BackwardFn>)
		} else {
			None
		};

		Ok(AutogradTensor::new(out, backward_fn))
	}
}

pub struct LinearBackwardFn {
	weights: Rc<RefCell<Param>>,

	/// should be `Some` if we should compute `d_w`
	inp: Option<Tensor>,

	/// should be `Some` if we should compute `d_inp`
	inp_backward: Option<Box<dyn BackwardFn>>,

	backward_scale: f64,
	internal_dtype: DType,
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
			internal_dtype,
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
			weights.update_grad(d_out.dtype(), |current_grad| match current_grad {
				CurrentGradValue::Uninit(tensor) => {
					let grad = mat(tensor)?;
					grad.clear_acc(d_w, internal_dtype)
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
			let d_inp = d_out.new_replace_tail(1, &input_shape, d_out.dtype())?;
			col(&d_inp)?.assign(d_i, internal_dtype)?;
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
		internal_dtype: DType,
		ctx: &mut ModelContext,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let linear = Linear::new(inputs, heads * outputs, dtype, internal_dtype, ctx)?;

		// This `unwrap()` should never fail unsless we have a bug
		#[allow(clippy::missing_panics_doc)]
		#[allow(clippy::unwrap_used)]
		linear.weights.borrow_mut().partition(heads, inputs * outputs).unwrap();

		// TODO - should we change the backward scale?
		//linear.backward_scale = 1.0 / (outputs as f64).sqrt();

		Ok(Self { linear, output_shape: [heads, outputs] })
	}
}

impl Fragment for MultiheadLinear {
	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.linear.collect_params(f);
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.linear.collect_named_params(format!("{prefix}.linear").as_str(), f);
	}

	fn randomize(
		&mut self,
		rng: &mut Rng,
	) -> std::result::Result<(), ErrPack<tensor::TensorOpError>> {
		self.linear.randomize(rng)
	}
}

impl UnaryFragment for MultiheadLinear {
	#[inline(never)]
	fn forward(&self, inp_node: AutogradTensor) -> Result<AutogradTensor, ErrPack<TensorOpError>> {
		let out_node = self.linear.forward(inp_node)?;
		let (out, backward_fn) = out_node.into_parts();

		// [..., heads * outputs] -> [..., heads, outputs]
		let out = out.reshape_last_dim(self.output_shape)?;

		let backward_fn = backward_fn.map(|inp_backward| {
			Box::new(MultiheadLinearBackwardFn { inp_backward }) as Box<dyn BackwardFn>
		});

		Ok(AutogradTensor::new(out, backward_fn))
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
