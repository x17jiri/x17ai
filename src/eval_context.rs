use crate::tensor::{Tensor, TensorSize};

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

pub struct EvalContext {
	training: bool,
	pub tensors: TensorStore,
}

impl EvalContext {
	pub fn new() -> EvalContext {
		EvalContext {
			training: false,
			tensors: TensorStore::new(),
		}
	}

	pub fn is_training(&self) -> bool {
		self.training
	}
}
