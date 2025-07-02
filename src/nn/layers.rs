//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

//pub mod attention;
pub mod cross_entropy;
pub mod linear;
pub mod rms_norm;
pub mod skip_con_rms_norm;
pub mod skip_connection;
pub mod softmax;
pub mod swiglu;

#[cfg(test)]
mod tests;

use crate::ErrPack;
use crate::autograd::AutogradNode;
use crate::tensor::TensorOpError;

use super::Param;

//pub use linear::{Linear, MultiheadLinear};
pub use cross_entropy::CrossEntropy;
pub use rms_norm::{RMSNorm, RMSNormGradientMode};
pub use swiglu::SwiGLU;

pub trait Layer {
	fn input_shape(&self) -> &[usize];
	fn output_shape(&self) -> &[usize];

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>));
	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>));

	fn params(&self) -> Vec<Rc<RefCell<Param>>> {
		let mut params = Vec::new();
		self.collect_params(&mut |p| params.push(p));
		params
	}

	fn named_params(&self, prefix: &str) -> Vec<(String, Rc<RefCell<Param>>)> {
		let mut params = Vec::new();
		self.collect_named_params(prefix, &mut |name, p| params.push((name, p)));
		params
	}

	fn forward(&self, inp_node: AutogradNode) -> Result<AutogradNode, ErrPack<TensorOpError>>;

	fn randomize(&mut self) -> Result<(), ErrPack<TensorOpError>>;
}
