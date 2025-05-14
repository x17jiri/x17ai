use crate::tensor::{Tensor, TensorSize};

pub mod linear;

pub trait Layer {
	fn randomize(&mut self);

	fn input_shape(&self) -> &[TensorSize];
	fn output_shape(&self) -> &[TensorSize];

	fn forward(&self, inp: &Tensor, out: &Tensor);
	fn backward(&self, d_out: &Tensor, d_inp: Option<&Tensor>);
}
