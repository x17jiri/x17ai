use crate::tensor::{Tensor, TensorSize};

pub mod cross_entropy;
pub mod linear;

pub struct TensorStore {
	tensors: Vec<Tensor>,
}

impl TensorStore {
	pub fn new() -> TensorStore {
		TensorStore { tensors: Vec::new() }
	}

	pub fn set<const N: usize>(&mut self, tensors: [Tensor; N]) {
		assert!(self.tensors.is_empty());
		self.tensors.extend(tensors.into_iter());
	}

	pub fn get<const N: usize>(&mut self) -> [Tensor; N] {
		assert!(self.tensors.len() == N);
		let mut iter = self.tensors.drain(..);
		std::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() })
	}
}

pub trait Layer {
	fn randomize(&mut self);

	fn input_shape(&self) -> &[TensorSize];
	fn output_shape(&self) -> &[TensorSize];

	fn forward(&self, inp: &Tensor, out: &Tensor, tensor_store: Option<&mut TensorStore>);
}

pub trait BackpropLayer {
	fn backward(&self, d_out: &Tensor, d_inp: Option<&Tensor>, tensor_store: &mut TensorStore);
}

pub trait Loss {
	fn expect(
		&self, expected_out: &Tensor, d_inp: Option<&Tensor>, tensor_store: &mut TensorStore,
	) -> f64;
}
