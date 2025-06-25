//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::RefCell;
use std::rc::Rc;

//pub mod attention;
//pub mod linear;
pub mod rms_norm;
//pub mod skip_connection;
pub mod softmax;
//pub mod softmax_cross_entropy;
//pub mod swiglu;

#[cfg(test)]
mod tests;

use crate::ErrPack;
use crate::autograd::AutogradNode;
use crate::tensor::TensorOpError;

use super::Param;

//pub use linear::{Linear, MultiheadLinear};
pub use rms_norm::{RMSNorm, RMSNormGradientMode};
pub use softmax::{Softmax, SoftmaxGradientMode};
//pub use softmax_cross_entropy::SoftmaxCrossEntropy;
//pub use swiglu::SwiGLU;

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

/*
pub trait LossFunction: Layer {
	/// This function is similar to `backward()`, but instead of derivatives with respect to
	/// the output, it takes expected value of the output.
	/// It would typically be used for the last layer in a model.
	fn backward_start(
		&self,
		out: Tensor,
		expected_out: Tensor,
		ctx: &mut EvalContext,
	) -> Result<Tensor, ErrPack<OptimizerError>>;

	fn loss(&self, out: Tensor, expected_out: Tensor) -> Result<f64, ErrPack<TensorOpError>>;
}
*/
