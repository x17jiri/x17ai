//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::likely;

use crate::autograd::{self, AutogradTensor, BackwardFn};
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::{Tensor, TensorOpError};
use crate::{ErrPack, custom_kernel};

//--------------------------------------------------------------------------------------------------

pub fn add(
	a_node: AutogradTensor,
	b_node: AutogradTensor,
) -> Result<AutogradTensor, ErrPack<TensorOpError>> {
	let (a, a_fn) = a_node.into_parts();
	let (b, b_fn) = b_node.into_parts();
	let c_dtype = common_dtype(a.dtype(), b.dtype());
	let c = if a.owns_buffer() && likely(c_dtype == a.dtype()) {
		a.clone()
	} else if b.owns_buffer() && likely(c_dtype == b.dtype()) {
		b.clone()
	} else {
		a.new_empty_like(c_dtype)?
	};
	c.assign(custom_kernel!(
		c_dtype,
		[a: &a, b: &b], (), {
			a + b
		}
	))?;

	#[allow(clippy::collapsible_else_if)]
	#[allow(clippy::option_if_let_else)]
	Ok(AutogradTensor::new(
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
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { a_fn, b_fn } = Box::into_inner(self);
		queue.add(a_fn, d_out.clone());
		queue.add(b_fn, d_out);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
