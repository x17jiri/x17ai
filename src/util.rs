//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::convert::Infallible;

// pub mod rc; TODO
pub mod array;

pub trait LossyInto<T> {
	fn lossy_into(self) -> T;
}

#[allow(clippy::cast_precision_loss)]
impl LossyInto<f64> for usize {
	fn lossy_into(self) -> f64 {
		self as f64
	}
}

#[allow(clippy::cast_precision_loss)]
impl LossyInto<f64> for u64 {
	fn lossy_into(self) -> f64 {
		self as f64
	}
}

pub trait UnwrapInfallible<T> {
	fn unwrap_infallible(self) -> T;
}

impl<T> UnwrapInfallible<T> for Result<T, Infallible> {
	fn unwrap_infallible(self) -> T {
		self.unwrap()
	}
}
