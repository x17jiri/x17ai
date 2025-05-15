use crate::expr::{self, Accumulable, Savable};
use crate::nn::{Layer, Loss, TensorStore};
use crate::tensor::{Tensor, TensorSize};

pub struct CrossEntropy {
	shape: [TensorSize; 1],
}

impl CrossEntropy {
	pub fn new(classes: TensorSize) -> CrossEntropy {
		CrossEntropy { shape: [classes] }
	}
}

impl Layer for CrossEntropy {
	fn randomize(&mut self) {}

	fn input_shape(&self) -> &[TensorSize] {
		&self.shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.shape
	}

	fn forward(&self, inp: &Tensor, out: &Tensor, tensor_store: Option<&mut TensorStore>) {
		expr::softmax(inp).save_to(out);

		if let Some(tensor_store) = tensor_store {
			tensor_store.set([out.clone()]);
		}
	}
}

impl Loss for CrossEntropy {
	fn expect(
		&self, expected_out: &Tensor, d_inp: Option<&Tensor>, tensor_store: &mut TensorStore,
	) -> f64 {
		let [out] = tensor_store.get();
		if let Some(d_inp) = d_inp {
			expr::sub(&out, expected_out).save_to(d_inp);
		}

		0.0 // TODO: implement
	}
}
