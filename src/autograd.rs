//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::ErrPack;
use crate::tensor::Tensor;

#[derive(Default)]
pub enum NodeValue {
	#[default]
	None,

	Value(Tensor),

	Grad(Tensor),
}

pub struct Node {
	pub value: NodeValue,
	pub backward_fn: Option<Box<dyn BackwardFn>>,
}

impl Node {
	pub fn value(&mut self) -> Option<Tensor> {
		if let NodeValue::Value(tensor) = std::mem::take(&mut self.value) {
			Some(tensor)
		} else {
			cold_path();
			None
		}
	}

	pub fn grad(&mut self) -> Option<Tensor> {
		if let NodeValue::Grad(tensor) = std::mem::take(&mut self.value) {
			Some(tensor)
		} else {
			cold_path();
			None
		}
	}
}

pub trait BackwardFn {
	fn backward(
		self: Box<Self>,
		d_out: Tensor,
		ctx: &mut AutogradCtx,
	) -> Result<(), ErrPack<AutogradError>>;
}

pub struct AutogradCtx {
	//
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AutogradError {
	MissingValue,
	MissingGrad,
}
