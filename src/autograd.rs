//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use smallvec::SmallVec;

use crate::ErrPack;
use crate::tensor::{Tensor, TensorOpError};

pub mod add;
pub mod split;

//--------------------------------------------------------------------------------------------------

pub struct AutogradNode {
	pub value: Tensor,
	pub backward_fn: Option<Box<dyn BackwardFn>>,
}

impl AutogradNode {
	pub fn new(value: Tensor, backward_fn: Option<Box<dyn BackwardFn>>) -> Self {
		Self { value, backward_fn }
	}

	pub fn requires_grad(&self) -> bool {
		self.backward_fn.is_some()
	}

	pub fn take(self) -> (Tensor, Option<Box<dyn BackwardFn>>) {
		let backward_fn = self.backward_fn;
		(self.value, backward_fn)
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Autograd {
	nodes: SmallVec<[(Box<dyn BackwardFn>, Tensor); 3]>,
}

impl Autograd {
	pub fn set_grad(&mut self, node: Box<dyn BackwardFn>, grad: Tensor) {
		self.nodes.push((node, grad));
	}

	pub fn run(
		backward_fn: Option<Box<dyn BackwardFn>>,
		grad: Tensor,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Some(backward_fn) = backward_fn else {
			return Ok(());
		};
		let mut ctx = Self { nodes: SmallVec::new() };
		backward_fn.backward(grad, &mut ctx)?;
		while let Some((node, d_out)) = ctx.nodes.pop() {
			node.backward(d_out, &mut ctx)?;
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub trait BackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>>;
}

//--------------------------------------------------------------------------------------------------

pub struct StraightThroughBackwardFn {
	pub inp_backward: Box<dyn BackwardFn>,
}

impl StraightThroughBackwardFn {
	pub fn new(inp_backward: Box<dyn BackwardFn>) -> Self {
		Self { inp_backward }
	}
}

impl BackwardFn for StraightThroughBackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		autograd.set_grad(self.inp_backward, d_out);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct GradientCapture {
	pub storage: Rc<RefCell<Option<Tensor>>>,
}

impl GradientCapture {
	pub fn new() -> Box<Self> {
		Box::new(Self { storage: Rc::new(RefCell::new(None)) })
	}

	pub fn storage(&self) -> Rc<RefCell<Option<Tensor>>> {
		self.storage.clone()
	}
}

impl BackwardFn for GradientCapture {
	#[allow(clippy::panic_in_result_fn)]
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		_autograd: &mut Autograd,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self { storage } = Box::into_inner(self);
		let mut storage = storage.borrow_mut();

		assert!(storage.is_none());
		*storage = Some(d_out);

		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
