//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use smallvec::SmallVec;

use crate::ErrPack;
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

pub struct Node {
	value: Option<Tensor>,
	backward_fn: Option<Box<dyn BackwardFn>>,
}

impl Node {
	pub fn new(value: Tensor, backward_fn: Option<Box<dyn BackwardFn>>) -> Box<Self> {
		Box::new(Self { value: Some(value), backward_fn })
	}

	pub fn requires_grad(&self) -> bool {
		self.backward_fn.is_some()
	}

	pub fn take_value(&mut self) -> Option<Tensor> {
		self.value.take()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct AutogradCtx {
	nodes: SmallVec<[(Box<Node>, Tensor); 3]>,
}

impl AutogradCtx {
	fn new() -> Self {
		Self { nodes: SmallVec::new() }
	}

	pub fn set_grad(&mut self, node: Box<Node>, grad: Tensor) {
		self.nodes.push((node, grad));
	}
}

//--------------------------------------------------------------------------------------------------

pub trait BackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		ctx: &mut AutogradCtx,
	) -> Result<(), ErrPack<TensorOpError>>;
}

//--------------------------------------------------------------------------------------------------

pub struct StraightThroughBackwardFn {
	pub inp_node: Box<Node>,
}

impl StraightThroughBackwardFn {
	pub fn new(inp_node: Box<Node>) -> Self {
		Self { inp_node }
	}
}

impl BackwardFn for StraightThroughBackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		ctx: &mut AutogradCtx,
	) -> Result<(), ErrPack<TensorOpError>> {
		ctx.set_grad(self.inp_node, d_out);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
