//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;

//--------------------------------------------------------------------------------------------------

pub trait FromToF64 {
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

pub fn dot<T: Copy + FromToF64>(a: &[Cell<T>], b: &[Cell<T>]) -> f64 {
	let zip = a.iter().zip(b);
	zip.map(|(a, b)| a.get().to_f64() * b.get().to_f64()).sum()
}

pub fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
	(a - b).abs() < eps
}

pub fn rsqrt(a: f64) -> f64 {
	1.0 / a.sqrt()
}

pub fn sigmoid(x: f64) -> f64 {
	1.0 / (1.0 + (-x).exp())
}

pub fn swish(x: f64) -> f64 {
	let sigmoid = sigmoid(x);
	x * sigmoid
}

pub fn swiglu(lin: f64, gate: f64) -> f64 {
	let swish = swish(gate);
	lin * swish
}

pub fn swiglu_backward(lin: f64, gate: f64) -> (f64, f64) {
	let sigmoid = sigmoid(gate);
	let swish = gate * sigmoid;

	let d_lin = swish;

	let d_gate = lin * (swish + sigmoid * (1.0 - swish));

	(d_lin, d_gate)
}

pub fn softmax_part1<T: Copy + FromToF64, S: Copy + FromToF64>(
	inp: &[Cell<T>], scratch: &[Cell<S>],
) -> (f64, f64) {
	// TODO
	// - calculating `max` is one loop
	// - calculating `sum` is another loop
	// - there are online algorithms for calculating `max` and `sum` simultaneously
	// - would they be worth it?

	let max: f64 = inp.iter().map(|x| x.get().to_f64()).fold(f64::MIN, f64::max);

	let mut sum = 0.0;
	for (i, s) in inp.iter().zip(scratch) {
		let val = i.get().to_f64();
		let val = val - max;
		let e = val.exp();
		s.set(S::from_f64(e));

		sum += e;
	}

	(max, sum)
}

pub fn softmax_part2<S: Copy + FromToF64, T: Copy + FromToF64>(
	scratch: &[Cell<S>], sum: f64, dst: &[Cell<T>],
) {
	// NOTE:
	// Subtracting max in part1 ensures at least one of the exponents
	// is `exp(max - max) == 1.0`. So sum will be >= 1.0 and division by zero
	// is impossible.
	// This could only fail if all inputs are `-inf` or at least one input is `+inf`.
	// In that case, `sum == nan` and so all outputs will be `nan`.
	let sum_recip = 1.0 / sum;

	for (s, d) in scratch.iter().zip(dst) {
		let val = s.get().to_f64() * sum_recip;
		d.set(T::from_f64(val));
	}
}

pub fn softmax<T: Copy + FromToF64>(dst: &[Cell<T>], inp: &[Cell<T>]) {
	// use `dst` as scratch space between part1 and part2
	let scratch = dst;
	let (_, sum) = softmax_part1(inp, scratch);
	softmax_part2(scratch, sum, dst);
}
