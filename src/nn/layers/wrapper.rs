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
use crate::autograd::{self, AutogradNode, BackwardFn};
use crate::nn::param::Param;
use crate::tensor::device::kernel::lookup::{scalar, tsr};
use crate::tensor::{Tensor, TensorOpError};

use super::Layer;

//--------------------------------------------------------------------------------------------------

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

//--------------------------------------------------------------------------------------------------

pub struct Wrapper<Nested: Layer> {
	pub nested: Nested,
	eps: f64,
	norm_pos: NormPosition,
}

impl<Nested: Layer> Wrapper<Nested> {
	pub fn new(nested: Nested, eps: f64) -> Option<Self> {
		let input_shape = nested.input_shape();
		let output_shape = nested.output_shape();
		if input_shape == output_shape
			&& let Some(&n_inputs) = input_shape.last()
		{
			Some(Self {
				nested,
				eps,
				norm_pos: NormPosition::Inside,
			})
		} else {
			cold_path();
			None
		}
	}
}

impl<Nested: Layer> Layer for Wrapper<Nested> {
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

		let inp_magn_recip = inp.new_replace_tail(1, &[1])?;
		inp_magn_recip.assign((tsr(&inp) * &inp).mean().sqrt().recip(scalar(self.eps)))?;

		let rms_norm = if self.norm_pos != NormPosition::Inside && inp.owns_buffer() {
			inp.clone()
		} else {
			inp.new_empty_like()?
		};
		rms_norm.assign(tsr(&inp) * &inp_magn_recip)?;

		let mut residual = if self.norm_pos == NormPosition::Inside {
			inp
		} else {
			std::mem::drop(inp);
			rms_norm.clone()
		};

		let (mut nested_out, nested_out_fn) = if let Some(backward_fn) = backward_fn {
			let rc_inner = Rc::new(RefCell::new(WrapperBackwardFn_Inner {
				d_residual: None,
				ratio: inp_magn_recip.new_empty_like()?,
			}));
			let inner = rc_inner.borrow();

			let nested_inp = AutogradNode::new(
				rms_norm,
				Some(Box::new(WrapperBackwardFn_Split {
					rc_inner: rc_inner.clone(),
					eps: self.eps,
					backward_fn,
				})),
			);
			let (nested_out, nested_out_fn) = self.nested.forward(nested_inp)?.take();

			inner.ratio.assign((tsr(&nested_out) * &nested_out).mean().sqrt())?;
			if self.norm_pos == NormPosition::Inside {
				inner.ratio.assign(tsr(&inner.ratio) * &inp_magn_recip)?;
			}
			std::mem::drop(inner);

			(
				nested_out,
				if let Some(nested_out_fn) = nested_out_fn {
					Some(Box::new(WrapperBackwardFn_Merge { rc_inner, nested_out_fn })
						as Box<dyn BackwardFn>)
				} else {
					cold_path();
					std::mem::drop(rc_inner);
					None
				},
			)
		} else {
			std::mem::drop(backward_fn);
			let nested_inp = AutogradNode::new(rms_norm, None);
			self.nested.forward(nested_inp)?.take()
		};

		if !nested_out.owns_buffer() {
			std::mem::swap(&mut nested_out, &mut residual);
		}
		let out = nested_out.reuse_or_new_like()?;
		out.assign(tsr(&nested_out) + &residual)?;
		Ok(AutogradNode::new(out, nested_out_fn))
	}

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>> {
		self.nested.randomize()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct WrapperBackwardFn_Inner {
	d_residual: Option<Tensor>,

	/// Magnitude of the signal coming from the nested block
	/// divided by magnitude of the residual signal
	ratio: Tensor,
}

pub struct WrapperBackwardFn_Split {
	rc_inner: Rc<RefCell<WrapperBackwardFn_Inner>>,
	eps: f64,
	backward_fn: Box<dyn BackwardFn>,
}

pub struct WrapperBackwardFn_Merge {
	rc_inner: Rc<RefCell<WrapperBackwardFn_Inner>>,
	nested_out_fn: Box<dyn BackwardFn>,
}

impl BackwardFn for WrapperBackwardFn_Split {
	#[allow(clippy::unwrap_used)]
	fn run(
		self: Box<Self>,
		d_nested: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { rc_inner, eps, backward_fn } = Box::into_inner(self);
		let refcell = Rc::into_inner(rc_inner).unwrap();
		let inner = refcell.into_inner();
		let WrapperBackwardFn_Inner { d_residual, ratio } = inner;
		let d_out = d_residual.unwrap();

		let d_nested_magn_recip = ratio.new_empty_like()?;
		d_nested_magn_recip
			.assign((tsr(&d_nested) * &d_nested).mean().sqrt().recip(scalar(eps)))?;
		ratio.assign(tsr(&ratio) * &d_nested_magn_recip)?;

		let d_out_magn = d_nested_magn_recip; // reuse tensor with different variable name
		d_out_magn.assign((tsr(&d_out) * &d_out).mean().sqrt())?;
		ratio.assign(tsr(&ratio) * &d_out_magn)?;

		let d_inp = d_out.reuse_or_new_like()?;
		d_inp.assign(tsr(&d_out) + (tsr(&d_nested) * &ratio))?;
		std::mem::drop(d_out);
		std::mem::drop(d_nested);

		let d_inp_magn_recip = ratio; // reuse tensor with different variable name
		d_inp_magn_recip.assign((tsr(&d_inp) * &d_inp).mean().sqrt().recip(scalar(eps)))?;
		d_out_magn.assign(tsr(&d_out_magn) * &d_inp_magn_recip)?;
		std::mem::drop(d_inp_magn_recip);

		d_inp.assign(tsr(&d_inp) * &d_out_magn)?;
		std::mem::drop(d_out_magn);

		queue.add(backward_fn, d_inp);
		Ok(())
	}
}

impl BackwardFn for WrapperBackwardFn_Merge {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { rc_inner, nested_out_fn } = Box::into_inner(self);
		let mut inner = rc_inner.borrow_mut();
		inner.d_residual = Some(d_out.clone());
		queue.add(nested_out_fn, d_out);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
