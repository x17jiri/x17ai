//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::device::cpu::math;
use crate::util::FromToF64;

//--------------------------------------------------------------------------------------------------

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
	pub fn slice_mut(&mut self, i: usize, h: usize, f: std::ops::Range<usize>) -> &mut [T] {
		let head = h >> self.head_shift;
		let offset = i * self.seq_stride + head * self.features;
		&mut self.data[(offset + f.start)..(offset + f.end)]
	}
}

//--------------------------------------------------------------------------------------------------

#[allow(clippy::needless_range_loop)] // I want to see all the `for` loops in the algorithm
#[allow(clippy::indexing_slicing)]
fn attention_thread<
	T: Copy + FromToF64,
	const V_FEATURES: usize,
	const K_FEATURES: usize,
	const TILE_WIDTH: usize,  // number of inputs
	const TILE_HEIGHT: usize, // number of outputs
>(
	o: &mut View3DMut<T>, // [output, head, V_FEATURES]
	q: &View3D<T>,        // [output, head, K_FEATURES]
	k: &View3D<T>,        // [input, head, K_FEATURES]
	v: &View3D<T>,        // [input, head, V_FEATURES]
	N: usize,             // sequence length
	j: usize,             // vertical position along a tile
	h: usize,             // head index
) {
	let mut scores: [f64; TILE_WIDTH] = [0.0; TILE_WIDTH]; //-- uninit ------------------------ SRAM
	for tile_j in (0..N).step_by(TILE_HEIGHT) {
		if j >= N - tile_j {
			break;
		}
		let q = q.slice(tile_j + j, h, 0..K_FEATURES); //-------------------------------------- SRAM
		let mut acc: [f64; V_FEATURES] = [0.0; V_FEATURES]; //--------------------------------- SRAM
		let mut max = f64::MIN;
		let mut sum = 0.0; // TODO - use Kahan sum for `acc` and `sum`
		for tile_i in (0..N).step_by(TILE_WIDTH) {
			let prev_max = max;
			for i in 0..TILE_WIDTH {
				let k = k.slice(tile_i + i, h, 0..K_FEATURES); //------------------------------ SRAM
				scores[i] = math::dot(q, k).clamp(f64::MIN, f64::MAX); // `clamp()` propagates NaN
				max = max.max(scores[i]); // `max()` doesn't propagate NaN
			}
			let prev_weight = (prev_max - max).exp();
			sum *= prev_weight;
			for f in 0..V_FEATURES {
				acc[f] *= prev_weight;
			}
			for i in 0..TILE_WIDTH {
				scores[i] = (scores[i] - max).exp();
				sum += scores[i];
				let v = v.slice(tile_i + i, h, 0..V_FEATURES); //------------------------------ SRAM
				for f in 0..V_FEATURES {
					acc[f] += scores[i] * v[f].to_f64();
				}
			}
		}
		let sum_recip = 1.0 / sum;
		let o = o.slice_mut(tile_j + j, h, 0..V_FEATURES);
		for f in 0..V_FEATURES {
			o[f] = T::from_f64(acc[f] * sum_recip);
		}
	}
}

pub fn attention<T: Copy + FromToF64>(
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
			attention_thread::<T, 64, 64, TILE_WIDTH, TILE_HEIGHT>(o, q, k, v, N, inner_j, h);
		}
	}
}

//--------------------------------------------------------------------------------------------------
