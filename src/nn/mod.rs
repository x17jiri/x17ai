use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub mod linear;
pub mod softmax;
pub mod softmax_cross_entropy;

use crate::eval_context::EvalContext;
use crate::optimizer::OptCoef;
use crate::param::Param;
use crate::tensor::{Tensor, TensorSize};

pub use linear::{Linear, MultiheadLinear};
pub use softmax::Softmax;
pub use softmax_cross_entropy::SoftmaxCrossEntropy;

pub trait Layer {
	fn randomize(&mut self);

	fn input_shape(&self) -> &[TensorSize];
	fn output_shape(&self) -> &[TensorSize];

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

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor;
}

pub trait BackpropLayer: Layer {
	fn init_optimizer(&self);
	fn zero_grad(&self);
	fn step(&self, opt_coef: &OptCoef);

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor;

	/// This function is similar to `backward()`. It should calculate derivatives of parameters
	/// used by the layer, but it doesn't calculate derivatives that could be used by previous
	/// layers. So it would typically only be used for the first layer in a model.
	fn backward_first(&self, d_out: Tensor, ctx: &mut EvalContext);

	fn as_loss_layer(&self) -> Option<&dyn LossLayer> {
		None
	}
}

pub trait LossLayer: BackpropLayer {
	/// This function is similar to `backward()`, but instead of derivatives with respect to
	/// the output, it takes expected value of the output.
	// It would typically be used for the last layer in a model.
	fn backward_last(&self, out: Tensor, expected_out: Tensor, ctx: &mut EvalContext) -> Tensor;

	fn loss(&self, out: Tensor, expected_out: Tensor) -> f64;
}
