//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::util::FromToF64;

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

pub fn dot<T: Copy + FromToF64>(a: &[T], b: &[T]) -> f64 {
	let zip = a.iter().zip(b);
	zip.fold(0.0, |acc, (a, b)| {
		let a = a.to_f64();
		let b = b.to_f64();
		mul_add(a, b, acc)
	})
}

pub fn softmax_part1_<T: Copy + FromToF64>(inp: &mut [T]) -> (f64, f64) {
	// TODO
	// - calculating `max` is one loop
	// - calculating `sum` is another loop
	// - there are online algorithms for calculating `max` and `sum` simultaneously
	// - would they be worth it?

	let max: f64 = inp.iter().map(T::to_f64).fold(f64::MIN, f64::max);

	let mut sum = 0.0;
	for i in inp {
		let val = i.to_f64();
		let val = val - max;
		let e = val.exp();
		*i = T::from_f64(e);

		sum += e;
	}

	(max, sum)
}

pub fn softmax_part2<S: Copy + FromToF64, T: Copy + FromToF64>(
	scratch: &[S],
	sum: f64,
	dst: &mut [T],
) {
	// NOTE:
	// Subtracting max in part1 ensures at least one of the exponents
	// is `exp(max - max) == 1.0`. So sum will be >= 1.0 and division by zero
	// is impossible.
	// This could only fail if all inputs are `-inf` or at least one input is `+inf`.
	// In that case, `sum == nan` and so all outputs will be `nan`.
	let sum_recip = 1.0 / sum;

	for (s, d) in scratch.iter().zip(dst) {
		let val = s.to_f64() * sum_recip;
		*d = T::from_f64(val);
	}
}

pub fn softmax_part2_<T: Copy + FromToF64>(sum: f64, inp: &mut [T]) {
	// NOTE:
	// Subtracting max in part1 ensures at least one of the exponents
	// is `exp(max - max) == 1.0`. So sum will be >= 1.0 and division by zero
	// is impossible.
	// This could only fail if all inputs are `-inf` or at least one input is `+inf`.
	// In that case, `sum == nan` and so all outputs will be `nan`.
	let sum_recip = 1.0 / sum;

	for i in inp {
		let val = i.to_f64() * sum_recip;
		*i = T::from_f64(val);
	}
}

pub fn softmax_<T: Copy + FromToF64>(inp: &mut [T]) {
	// use `dst` as scratch space between part1 and part2
	let (_, sum) = softmax_part1_(inp);
	softmax_part2_(sum, inp);
}

pub fn softmax_backward<T: Copy + FromToF64>(d_inp: &mut [T], out: &[T], d_out: &[T]) {
	let g = dot(out, d_out);
	for (d_i, (o, d_o)) in d_inp.iter_mut().zip(out.iter().zip(d_out)) {
		let o = o.to_f64();
		let d_o = d_o.to_f64();

		let v = (d_o - g) * o;

		*d_i = T::from_f64(v);
	}
}
