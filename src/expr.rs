// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;

pub trait Savable {
	/// Calculate the result of the operation represented by `self`
	// and save it into the `to` tensor.
	fn save_to(&self, to: &Tensor);
}

pub trait Accumulable {
	/// Calculate the result of the operation represented by `self`
	/// and accumulate it into the `to` tensor.
	///
	///    to = to_weight * to + expr_weight * self
	fn acc_to(&self, to: &Tensor, expr_weight: f64, to_weight: f64);
}
