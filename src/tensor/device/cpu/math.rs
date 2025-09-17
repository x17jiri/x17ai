//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

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

pub trait Float:
	Copy
	+ Default
	+ std::ops::Mul<Output = Self>
	+ std::ops::Add<Output = Self>
	+ std::ops::Sub<Output = Self>
	+ FromToF64
{
	const ZERO: Self;
	const ONE: Self;
	const NEG_INFINITY: Self;

	fn max(self, other: Self) -> Self; // does not propagate NaN
	fn min(self, other: Self) -> Self; // does not propagate NaN
	fn clamp_to_finite(self) -> Self; // propagates NaN

	fn exp(self) -> Self;
	fn recip(self) -> Self;
}

#[allow(clippy::use_self)]
impl Float for f32 {
	const ZERO: Self = 0.0;
	const ONE: Self = 1.0;
	const NEG_INFINITY: Self = f32::NEG_INFINITY;

	fn max(self, other: Self) -> Self {
		f32::max(self, other)
	}

	fn min(self, other: Self) -> Self {
		f32::min(self, other)
	}

	fn clamp_to_finite(self) -> Self {
		f32::clamp(self, f32::MIN, f32::MAX)
	}

	fn exp(self) -> Self {
		f32::exp(self)
	}

	fn recip(self) -> Self {
		1.0 / self
	}
}

#[allow(clippy::use_self)]
impl Float for f64 {
	const ZERO: Self = 0.0;
	const ONE: Self = 1.0;
	const NEG_INFINITY: Self = f64::NEG_INFINITY;

	fn max(self, other: Self) -> Self {
		f64::max(self, other)
	}

	fn min(self, other: Self) -> Self {
		f64::min(self, other)
	}

	fn clamp_to_finite(self) -> Self {
		f64::clamp(self, f64::MIN, f64::MAX)
	}

	fn exp(self) -> Self {
		f64::exp(self)
	}

	fn recip(self) -> Self {
		1.0 / self
	}
}

//--------------------------------------------------------------------------------------------------

/// Calculates `(a * b) + c`
pub fn mul_add(a: f64, b: f64, c: f64) -> f64 {
	// Clippy recommends using `f64::mul_add()`, however I checked the assembly and
	// it generates `callq	*fma@GOTPCREL(%rip)`, which will probably be incredibly slow.
	#![allow(clippy::suboptimal_flops)]
	(a * b) + c
}

/// Calculates `(a * a_weight) + (b * b_weight)`
pub fn add_weighted(a: f64, a_weight: f64, b: f64, b_weight: f64) -> f64 {
	mul_add(a, a_weight, b * b_weight)
}

/// Linear interpolation between a and b. Equivalent to:
///
///     `add_weighted(a, 1.0 - t, b, t)`
///
/// When `t == 0.0`, returns `a`.
/// When `t == 1.0`, returns `b`.
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
	mul_add(t, b, mul_add(-t, a, a))
}

//--------------------------------------------------------------------------------------------------
