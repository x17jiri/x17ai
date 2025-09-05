//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::util::LossyInto;

//--------------------------------------------------------------------------------------------------

pub trait Float:
	Copy
	+ Default
	+ std::ops::Mul<Output = Self>
	+ std::ops::Add<Output = Self>
	+ std::ops::Sub<Output = Self>
{
	const ZERO: Self;
	const MAX_POSITIVE: Self;
	const MAX_NEGATIVE: Self;

	fn max(self, other: Self) -> Self; // does not propagate NaN
	fn min(self, other: Self) -> Self; // does not propagate NaN
	fn clamp(self, min: Self, max: Self) -> Self; // propagates NaN

	fn exp(self) -> Self;
	fn recip(self) -> Self;
}

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

#[derive(Clone, Copy)]
pub struct View3D<'a, T> {
	pub data: &'a [T],
	pub seq_len: usize,
	pub seq_stride: usize,
	pub head_shift: usize,
	pub heads: usize,
	pub features: usize,
}

impl<'a, T> View3D<'a, T> {
	#[allow(clippy::indexing_slicing)]
	pub fn slice(&self, i: usize, h: usize, f: std::ops::Range<usize>) -> &'a [T] {
		let head = h >> self.head_shift;
		let offset = i * self.seq_stride + head * self.features;
		&self.data[(offset + f.start)..(offset + f.end)]
	}
}

pub struct View3DMut<'a, T> {
	pub data: &'a mut [T],
	pub seq_len: usize,
	pub seq_stride: usize,
	pub head_shift: usize,
	pub heads: usize,
	pub features: usize,
}

impl<'a, T> View3DMut<'a, T> {
	#[allow(clippy::indexing_slicing)]
	pub fn slice_mut(&mut self, i: usize, h: usize, f: std::ops::Range<usize>) -> &mut [T] {
		let head = h >> self.head_shift;
		let offset = i * self.seq_stride + head * self.features;
		&mut self.data[(offset + f.start)..(offset + f.end)]
	}
}

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
		self.c = (t - b) - y;
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

#[allow(clippy::needless_range_loop)] // I want to see all the `for` loops in the algorithm
#[allow(clippy::indexing_slicing)]
fn attention_thread<
	T: Float,
	U: Float + From<T> + LossyInto<T>,
	const V_FEATURES: usize,
	const K_FEATURES: usize,
	const TILE_WIDTH: usize,  // number of inputs
	const TILE_HEIGHT: usize, // number of outputs
>(
	o: &mut View3DMut<T>,    // [output, head, V_FEATURES]
	q: &View3D<T>,           // [output, head, K_FEATURES]
	k: &View3D<T>,           // [input, head, K_FEATURES]
	v: &View3D<T>,           // [input, head, V_FEATURES]
	N: usize,                // sequence length
	j: usize,                // vertical position along a tile
	h: usize,                // head index
	acc: &mut [KahanAcc<U>], // [V_FEATURES] ------------------------------------------- acc - SRAM
) {
	let mut scores = [U::ZERO; TILE_WIDTH]; //-- uninit ------------------------------ scores - SRAM
	for tile_j in (0..N).step_by(TILE_HEIGHT) {
		if j >= N - tile_j {
			break;
		}
		let q = q.slice(tile_j + j, h, 0..K_FEATURES); //---------------------------------- Q - SRAM
		acc.fill(KahanAcc::<U>::new());
		let mut max = U::MAX_NEGATIVE;
		let mut sum = KahanAcc::<U>::new();
		for tile_i in (0..N).step_by(TILE_WIDTH) {
			let prev_max = max;
			for i in 0..TILE_WIDTH {
				let k = k.slice(tile_i + i, h, 0..K_FEATURES); //-------------------------- K - SRAM
				scores[i] = dot::<T, U>(q, k).clamp(U::MAX_NEGATIVE, U::MAX_POSITIVE);
				max = max.max(scores[i]);
			}
			let prev_weight = (prev_max - max).exp();
			sum.scale_(prev_weight);
			for f in 0..V_FEATURES {
				acc[f].scale_(prev_weight);
			}
			for i in 0..TILE_WIDTH {
				let w = (scores[i] - max).exp();
				sum.acc_(w);
				let v = v.slice(tile_i + i, h, 0..V_FEATURES); //-------------------------- V - SRAM
				for f in 0..V_FEATURES {
					acc[f].acc_(w * v[f].into());
				}
			}
		}
		let sum_recip = sum.value().recip();
		let o = o.slice_mut(tile_j + j, h, 0..V_FEATURES);
		for f in 0..V_FEATURES {
			o[f] = (acc[f].value() * sum_recip).lossy_into();
		}
	}
}

pub fn attention<T: Float + From<T> + LossyInto<T>, U: Float + From<T> + LossyInto<T>>(
	o: &mut View3DMut<T>, // [output, head, V_FEATURES]
	q: &View3D<T>,        // [output, head, K_FEATURES]
	k: &View3D<T>,        // [input, head, K_FEATURES]
	v: &View3D<T>,        // [input, head, V_FEATURES]
) {
	const TILE_WIDTH: usize = 32;
	const TILE_HEIGHT: usize = 32;
	let N = o.seq_len;
	let heads = o.heads;
	// These 2 `for` loops can run in parallel
	for inner_j in 0..TILE_HEIGHT {
		for h in 0..heads {
			attention_thread::<T, U, 64, 64, TILE_WIDTH, TILE_HEIGHT>(o, q, k, v, N, inner_j, h);
		}
	}
}

//--------------------------------------------------------------------------------------------------
