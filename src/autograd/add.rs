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
	mut a_node: AutogradNode,
	mut b_node: AutogradNode,
) -> Result<AutogradNode, ErrPack<TensorOpError>> {
	if !a_node.value.owns_buffer() {
		std::mem::swap(&mut a_node, &mut b_node);
	}
	let (a, a_fn) = a_node.take();
	let (b, b_fn) = b_node.take();
	let c = if a.owns_buffer() { a.clone() } else { a.new_empty_like()? };
	c.assign(&a + &b)?;

	if a_fn.is_some() || b_fn.is_some() {
		let backward_fn = Box::new(AddBackwardFn { a: a_fn, b: b_fn });
		Ok(AutogradNode::new(c, Some(backward_fn)))
	} else {
		Ok(AutogradNode::new(c, None))
	}
}

pub struct AddBackwardFn {
	a: Option<Box<dyn BackwardFn>>,
	b: Option<Box<dyn BackwardFn>>,
}

impl BackwardFn for AddBackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { a, b } = Box::into_inner(self);
		if let Some(a) = a {
			autograd.set_grad(a, d_out.clone());
		}
		if let Some(b) = b {
			autograd.set_grad(b, d_out);
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
