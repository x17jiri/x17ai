// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;

pub trait Buffer {
	fn capacity(&self) -> usize;

	fn zeros_(&self, tensor: &Tensor);
	fn randn_(&self, tensor: &Tensor);

	fn rms_norm(&self, a: &Tensor, out: &Tensor, params: &ReduceParams);

	//	fn mm(&self, a: &Tensor, b: &Tensor, c: &Tensor);

	fn format(
		&self,
		byte_offset: usize,
		dtype: DType,
		f: &mut fmt::Formatter,
		count: usize,
	) -> fmt::Result;
}
