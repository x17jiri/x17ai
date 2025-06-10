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

impl Default for TensorStore {
	fn default() -> Self {
		Self::new()
	}
}

impl TensorStore {
	pub fn new() -> Self {
		Self { tensors: Vec::new() }
	}

	/// Sets the store with a fixed-size array of tensors.
	///
	/// We assume that the store value is set once during the forward pass,
	/// and then the tensors are retrieved once during the backward pass.
	///
	/// So when calling this, the store should be empty.
	///
	/// # Panics
	/// - if the store is not empty.
	pub fn set<const N: usize>(&mut self, tensors: [Tensor; N]) {
		assert!(self.tensors.is_empty());
		self.tensors.extend(tensors);
	}

	/// Retrieves the tensors from the store as a fixed-size array.
	///
	/// We assume that the store value is set once during the forward pass,
	/// and then the tensors are retrieved once during the backward pass.
	///
	/// So when calling this, the store should contain exactly `N` tensors.
	///
	/// # Panics
	/// - if the store does not contain exactly `N` tensors.
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
	pub fn new(training: bool) -> Self {
		Self { training, tensors: TensorStore::new() }
	}

	pub fn is_training(&self) -> bool {
		self.training
	}
}
