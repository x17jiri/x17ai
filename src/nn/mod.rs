use crate::eval_ctx::EvalContext;
use crate::tensor::{Tensor, TensorSize};

pub mod cross_entropy;
pub mod linear;

pub trait Layer {
	fn randomize(&mut self);

	fn input_shape(&self) -> &[TensorSize];
	fn output_shape(&self) -> &[TensorSize];

	fn forward(&self, inp: &Tensor, out: &Tensor, ctx: &mut EvalContext);
}

pub trait BackpropLayer: Layer {
	fn backward(&self, d_out: &Tensor, d_inp: Option<&Tensor>, ctx: &mut EvalContext);
}

pub trait LossLayer: Layer {
	fn expect(&self, expected_out: &Tensor, d_inp: Option<&Tensor>, ctx: &mut EvalContext) -> f64;
}
