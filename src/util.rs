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

pub trait LossyFrom<T> {
	fn lossy_from(value: T) -> Self;
}

#[allow(clippy::cast_precision_loss, clippy::use_self)]
impl LossyFrom<usize> for f64 {
	fn lossy_from(value: usize) -> Self {
		value as f64
	}
}

#[allow(clippy::cast_precision_loss, clippy::use_self)]
impl LossyFrom<u64> for f64 {
	fn lossy_from(value: u64) -> Self {
		value as f64
	}
}

#[allow(clippy::cast_possible_truncation, clippy::use_self)]
impl LossyFrom<f64> for f32 {
	fn lossy_from(value: f64) -> Self {
		value as f32
	}
}

impl<T> LossyFrom<T> for T {
	fn lossy_from(value: T) -> Self {
		value
	}
}

//--------------------------------------------------------------------------------------------------

pub trait LossyInto<T> {
	fn lossy_into(self) -> T;
}

impl<T, U> LossyInto<U> for T
where
	U: LossyFrom<T>,
{
	fn lossy_into(self) -> U {
		U::lossy_from(self)
	}
}

//--------------------------------------------------------------------------------------------------

pub trait UnwrapInfallible<T> {
	fn unwrap_infallible(self) -> T;
}

impl<T> UnwrapInfallible<T> for Result<T, Infallible> {
	fn unwrap_infallible(self) -> T {
		self.unwrap()
	}
}

//--------------------------------------------------------------------------------------------------
