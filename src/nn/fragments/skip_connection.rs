//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

use crate::ErrPack;
use crate::autograd::AutogradTensor;
use crate::nn::fragments::add::add;
use crate::nn::fragments::split::split;
use crate::nn::fragments::{Fragment, UnaryFragment};
use crate::nn::param::Param;
use crate::rng::Rng;
use crate::tensor::TensorOpError;

//--------------------------------------------------------------------------------------------------

pub struct SkipConnection<Nested: UnaryFragment> {
	nested: Nested,
}

impl<Nested: UnaryFragment> SkipConnection<Nested> {
	pub fn new(nested: Nested) -> Self {
		Self { nested }
	}
}

impl<Nested: UnaryFragment> Fragment for SkipConnection<Nested> {
	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.nested.collect_params(f);
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.nested.collect_named_params(prefix, f);
	}

	fn randomize(&mut self, rng: &mut Rng) -> Result<(), ErrPack<TensorOpError>> {
		self.nested.randomize(rng)
	}
}

impl<Nested: UnaryFragment> UnaryFragment for SkipConnection<Nested> {
	fn forward(&self, inp_node: AutogradTensor) -> Result<AutogradTensor, ErrPack<TensorOpError>> {
		let [a, b] = split(inp_node);
		let nested_out = self.nested.forward(a)?;
		add(nested_out, b)
	}
}

//--------------------------------------------------------------------------------------------------
