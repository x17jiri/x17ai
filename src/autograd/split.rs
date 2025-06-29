//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::rc::Rc;

use crate::ErrPack;
use crate::autograd::{Autograd, AutogradNode, BackwardFn};
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

pub fn split<const N: usize>(inp_node: AutogradNode, weights: [f64; N]) -> [AutogradNode; N] {
	let mut output = [const { MaybeUninit::uninit() }; N];
	if N > 0 {
		let (inp, inp_fn) = inp_node.take();
		if let Some(inp_fn) = inp_fn {
			let rc_inner =
				Rc::new(RefCell::new(SplitBackwardFn_Inner { grad: None, backward_fn: inp_fn }));
			for i in 0..N - 1 {
				let inp = inp.clone();
				let rc_inner = rc_inner.clone();
				output[i].write(AutogradNode::new(
					inp,
					Some(Box::new(SplitBackwardFn { rc_inner, weight: weights[i] })
						as Box<dyn BackwardFn>),
				));
			}
			output[N - 1].write(AutogradNode::new(
				inp,
				Some(Box::new(SplitBackwardFn { rc_inner, weight: weights[N - 1] })
					as Box<dyn BackwardFn>),
			));
		} else {
			for i in 0..N - 1 {
				let inp = inp.clone();
				output[i].write(AutogradNode::new(inp, None));
			}
			output[N - 1].write(AutogradNode::new(inp, None));
		}
	}
	unsafe { MaybeUninit::array_assume_init(output) }
}

//--------------------------------------------------------------------------------------------------

pub struct GradWithWeight {
	tensor: Tensor,
	weight: f64,
}

pub struct SplitBackwardFn_Inner {
	grad: Option<GradWithWeight>,
	backward_fn: Box<dyn BackwardFn>,
}

pub struct SplitBackwardFn {
	rc_inner: Rc<RefCell<SplitBackwardFn_Inner>>,
	weight: f64,
}

impl BackwardFn for SplitBackwardFn {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { rc_inner, weight } = Box::into_inner(self);
		let mut d_out = GradWithWeight { tensor: d_out, weight };

		// accumulate gradient
		{
			let mut inner = rc_inner.borrow_mut();
			if let Some(ref mut grad) = inner.grad {
				if !grad.tensor.owns_buffer() {
					std::mem::swap(grad, &mut d_out);
				}
				if grad.tensor.owns_buffer() {
					grad.tensor
						.assign(&grad.tensor * grad.weight + &d_out.tensor * d_out.weight)?;
				} else {
					let mut new_grad = grad.tensor.new_empty_like()?;
					std::mem::swap(&mut grad.tensor, &mut new_grad);
					grad.tensor.assign(&new_grad * grad.weight + &d_out.tensor * d_out.weight)?;
					std::mem::drop(new_grad);
				}
				// Both gradients have already been scaled by their weights.
				// When we get another gradient, we don't want to scale them again.
				grad.weight = 1.0;
				std::mem::drop(d_out);
			} else {
				// first gradient
				inner.grad = Some(d_out);
			}
		}

		// propagate if we have the last Rc
		if let Ok(refcell) = Rc::try_unwrap(rc_inner) {
			let SplitBackwardFn_Inner { grad, backward_fn } = refcell.into_inner();
			// The value 1.0 is not result of any computation, so it should be exact.
			#[allow(clippy::float_cmp)]
			if let Some(grad) = grad {
				debug_assert!(grad.weight == 1.0, "GradWithWeight.weight should be 1.0");
				autograd.set_grad(backward_fn, grad.tensor);
			} else {
				// TODO - return some error when grad is None ?
			}
		}

		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
