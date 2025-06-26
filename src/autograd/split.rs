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

pub fn split<const N: usize>(inp_node: AutogradNode) -> [AutogradNode; N] {
	let mut output = [const { MaybeUninit::uninit() }; N];
	let (inp, inp_fn) = inp_node.take();
	if let Some(inp_fn) = inp_fn {
		let rc_inner =
			Rc::new(RefCell::new(SplitBackwardFn_Inner { grad: None, backward_fn: inp_fn }));
		for elem in output.iter_mut().take(N.saturating_sub(1)) {
			let inp = inp.clone();
			let rc_inner = rc_inner.clone();
			elem.write(AutogradNode::new(
				inp,
				Some(Box::new(SplitBackwardFn { rc_inner }) as Box<dyn BackwardFn>),
			));
		}
		if let Some(last) = output.last_mut() {
			last.write(AutogradNode::new(
				inp,
				Some(Box::new(SplitBackwardFn { rc_inner }) as Box<dyn BackwardFn>),
			));
		}
	} else {
		for elem in output.iter_mut().take(N.saturating_sub(1)) {
			let inp = inp.clone();
			elem.write(AutogradNode::new(inp, None));
		}
		if let Some(last) = output.last_mut() {
			last.write(AutogradNode::new(inp, None));
		}
	}
	unsafe { MaybeUninit::array_assume_init(output) }
}

//--------------------------------------------------------------------------------------------------

pub struct SplitBackwardFn_Inner {
	grad: Option<Tensor>,
	backward_fn: Box<dyn BackwardFn>,
}

pub struct SplitBackwardFn {
	rc_inner: Rc<RefCell<SplitBackwardFn_Inner>>,
}

impl BackwardFn for SplitBackwardFn {
	fn backward(
		self: Box<Self>,
		mut d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { rc_inner } = Box::into_inner(self);

		// accumulate gradient
		{
			let mut inner = rc_inner.borrow_mut();
			if let Some(ref mut grad) = inner.grad {
				if !grad.owns_buffer() {
					std::mem::swap(grad, &mut d_out);
				}
				if grad.owns_buffer() {
					let grad: &Tensor = grad;
					grad.assign(grad + &d_out)?;
				} else {
					let mut new_grad = grad.new_empty_like()?;
					std::mem::swap(grad, &mut new_grad);
					grad.assign(&new_grad + &d_out)?;
					std::mem::drop(new_grad);
				}
				std::mem::drop(d_out);
			} else {
				// first gradient
				inner.grad = Some(d_out);
			}
		}

		// propagate if we have the last Rc
		if let Ok(refcell) = Rc::try_unwrap(rc_inner) {
			let SplitBackwardFn_Inner { grad, backward_fn } = refcell.into_inner();
			if let Some(grad) = grad {
				backward_fn.backward(grad, autograd)?;
			} else {
				// TODO - return some error when grad is None ?
			}
		}

		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
