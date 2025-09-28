//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

mod attention;
mod kernel_eval;
mod mm;

pub use attention::attention;
pub use kernel_eval::run_kernel;
pub use mm::mm;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct KahanAcc<T: Float> {
	sum: T,
	c: T,
}

impl<T: Float> Default for KahanAcc<T> {
	fn default() -> Self {
		Self::new()
	}
}

impl<T: Float> KahanAcc<T> {
	pub fn new() -> Self {
		Self { sum: T::default(), c: T::default() }
	}

	pub fn acc_(&mut self, value: T) {
		let a = self.sum.max(value);
		let b = self.sum.min(value);
		let y = b - self.c;
		let t = a + y;
		self.c = (t - a) - y;
		self.sum = t;
	}

	pub fn scale_(&mut self, factor: T) {
		self.sum = self.sum * factor;
		self.c = self.c * factor;
	}

	pub fn value(&self) -> T {
		self.sum
	}
}

//--------------------------------------------------------------------------------------------------

#[allow(clippy::indexing_slicing)]
pub fn dot<T: Float, U: Float + From<T>>(a: &[T], b: &[T]) -> U {
	debug_assert!(a.len() == b.len());
	let L = a.len().min(b.len());
	match L {
		0 => U::ZERO,
		1 => {
			let a0: U = a[0].into();
			let b0: U = b[0].into();
			a0 * b0
		},
		2 => {
			let a0: U = a[0].into();
			let b0: U = b[0].into();
			let a1: U = a[1].into();
			let b1: U = b[1].into();
			a0 * b0 + a1 * b1
		},
		_ => {
			let mid = (a.len() / 2).next_power_of_two();
			let (a1, a2) = a.split_at(mid);
			let (b1, b2) = b.split_at(mid);
			dot::<T, U>(a1, b1) + dot::<T, U>(a2, b2)
		},
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
