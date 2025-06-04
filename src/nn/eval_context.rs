//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::Tensor;

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
	pub fn new(training: bool) -> EvalContext {
		EvalContext { training, tensors: TensorStore::new() }
	}

	pub fn is_training(&self) -> bool {
		self.training
	}
}
