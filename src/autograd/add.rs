//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::autograd::{Autograd, AutogradNode, BackwardFn};
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

pub fn add(
	a_node: AutogradNode,
	b_node: AutogradNode,
) -> Result<AutogradNode, ErrPack<TensorOpError>> {
	let (mut a, a_fn) = a_node.take();
	let (mut b, b_fn) = b_node.take();
	if !a.owns_buffer() {
		// Note: we don't swap `a_fn` and `b_fn`, but that's ok. Their order is not important
		std::mem::swap(&mut a, &mut b);
	}
	let c = if a.owns_buffer() { a.clone() } else { a.new_empty_like()? };
	c.assign(&a + &b)?;

	#[allow(clippy::collapsible_else_if)]
	#[allow(clippy::option_if_let_else)]
	Ok(AutogradNode::new(
		c,
		if let Some(a_fn) = a_fn {
			if let Some(b_fn) = b_fn {
				Some(Box::new(AddBackwardFn { a_fn, b_fn }))
			} else {
				Some(a_fn)
			}
		} else {
			if let Some(b_fn) = b_fn {
				Some(b_fn)
			} else {
				None //
			}
		},
	))
}

pub struct AddBackwardFn {
	a_fn: Box<dyn BackwardFn>,
	b_fn: Box<dyn BackwardFn>,
}

impl BackwardFn for AddBackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { a_fn, b_fn } = Box::into_inner(self);
		autograd.set_grad(a_fn, d_out.clone());
		autograd.set_grad(b_fn, d_out);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
