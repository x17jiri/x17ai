//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::convert::Infallible;

pub mod array;
pub mod mycell;

//--------------------------------------------------------------------------------------------------

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

//--------------------------------------------------------------------------------------------------

pub trait FromToF64: Copy {
	const MIN: f64; // largest negative value of type

	fn from_f64(val: f64) -> Self;
	fn to_f64(&self) -> f64;
}

#[allow(clippy::use_self)]
impl FromToF64 for f32 {
	const MIN: f64 = f32::MIN as f64;

	fn from_f64(val: f64) -> Self {
		#[allow(clippy::cast_possible_truncation)]
		(val as f32)
	}

	fn to_f64(&self) -> f64 {
		f64::from(*self)
	}
}

#[allow(clippy::use_self)]
impl FromToF64 for f64 {
	const MIN: f64 = f64::MIN;

	fn from_f64(val: f64) -> Self {
		val
	}

	fn to_f64(&self) -> f64 {
		*self
	}
}

//--------------------------------------------------------------------------------------------------
