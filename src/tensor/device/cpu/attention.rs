//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::buffer::AttentionArgs;
use crate::tensor::device::cpu::math::Float;
use crate::tensor::generic::map::{Map, ND};
use crate::util::LossyInto;

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

#[derive(Clone, Copy)]
pub struct View3D<T> {
	pub data: *const T,
	pub data_len: usize,
	pub seq_stride: usize,
	pub head_stride: usize,
}

impl<T> View3D<T> {
	pub fn new(map: ND<3>, data: *const T) -> Self {
		let span = map.span();
		Self {
			data: unsafe { data.add(span.start) },
			data_len: span.len(),
			seq_stride: map.dim(0).stride,
			head_stride: map.dim(1).stride,
		}
	}

	/// # Panics
	/// - If the slice is out of bounds of the underlying data.
	pub fn slice(&self, i: usize, h: usize, f: std::ops::Range<usize>) -> &[T] {
		let offset = i * self.seq_stride + h * self.head_stride;
		let begin = offset + f.start;
		let end = offset + f.end;
		assert!(begin <= end && end <= self.data_len);
		unsafe { std::slice::from_raw_parts(self.data.add(begin), end - begin) }
	}
}

pub struct View3DMut<T> {
	pub data: *mut T,
	pub data_len: usize,
	pub head_stride: usize,
	pub seq_stride: usize,
}

impl<T> View3DMut<T> {
	pub fn new(map: ND<3>, data: *mut T) -> Self {
		let span = map.span();
		Self {
			data: unsafe { data.add(span.start) },
			data_len: span.len(),
			seq_stride: map.dim(0).stride,
			head_stride: map.dim(1).stride,
		}
	}

	/// # Panics
	/// - If the slice is out of bounds of the underlying data.
	pub fn slice_mut(&mut self, i: usize, h: usize, f: std::ops::Range<usize>) -> &mut [T] {
		let offset = i * self.seq_stride + h * self.head_stride;
		let begin = offset + f.start;
		let end = offset + f.end;
		assert!(begin <= end && end <= self.data_len);
		unsafe { std::slice::from_raw_parts_mut(self.data.add(begin), end - begin) }
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
#[allow(clippy::too_many_arguments)]
fn attention_thread<
	T: Float,
	U: Float + From<T> + LossyInto<T>,
	const TILE_WIDTH: usize,  // number of inputs
	const TILE_HEIGHT: usize, // number of outputs
>(
	o: &mut View3DMut<T>,    // [output, head, V_FEATURES]
	q: &View3D<T>,           // [output, head, K_FEATURES]
	k: &View3D<T>,           // [input, head, K_FEATURES]
	v: &View3D<T>,           // [input, head, V_FEATURES]
	QN: usize,               // number of queries and outputs
	KN: usize,               // number of keys/values
	qh: usize,               // q head index
	kvh: usize,              // k/v head index
	inner_j: usize,          // vertical position along a tile
	acc: &mut [KahanAcc<U>], // [V_FEATURES] ------------------------------------------- acc - SRAM
	V_FEATURES: usize,
	K_FEATURES: usize,
) {
	let mut scores = [U::ZERO; TILE_WIDTH]; //-- uninit ----------------------------- scores - SRAM
	for outer_j in (0..QN).step_by(TILE_HEIGHT) {
		if inner_j >= QN - outer_j {
			break; // This break is diverging
		}
		let j = outer_j + inner_j;

		let q = q.slice(j, qh, 0..K_FEATURES); //--------------------------------------------- Q - SRAM
		acc.fill(KahanAcc::<U>::new());
		let mut max = U::NEG_INFINITY.clamp_to_finite();
		let mut sum = KahanAcc::<U>::new();
		for outer_i in (0..KN).step_by(TILE_WIDTH) {
			let CNT = (KN - outer_i).min(TILE_WIDTH);
			let prev_max = max;
			for inner_i in 0..CNT {
				let i = outer_i + inner_i;
				let k = k.slice(i, kvh, 0..K_FEATURES); //------------------------------------- K - SRAM
				scores[i] = dot::<T, U>(q, k).clamp_to_finite();
				max = max.max(scores[i]);
			}
			let prev_weight = (prev_max - max).exp();
			sum.scale_(prev_weight);
			for f in 0..V_FEATURES {
				acc[f].scale_(prev_weight);
			}
			for inner_i in 0..CNT {
				let i = outer_i + inner_i;
				let w = (scores[i] - max).exp();
				sum.acc_(w);
				let v = v.slice(i, kvh, 0..V_FEATURES); //------------------------------------- V - SRAM
				for f in 0..V_FEATURES {
					acc[f].acc_(w * v[f].into());
				}
			}
		}
		let sum_recip = sum.value().recip();
		let o = o.slice_mut(j, qh, 0..V_FEATURES);
		for f in 0..V_FEATURES {
			o[f] = (acc[f].value() * sum_recip).lossy_into();
		}
	}
}

pub fn attention<T: Float, U: Float + From<T> + LossyInto<T>>(
	args: &AttentionArgs,
) -> Result<(), ErrPack<TensorOpError>> {
	const TILE_WIDTH: usize = 32;
	const TILE_HEIGHT: usize = 32;

	let q = View3D::<T>::new(args.q_map(), args.q.cast());
	let k = View3D::<T>::new(args.k_map(), args.k.cast());
	let v = View3D::<T>::new(args.v_map(), args.v.cast());
	let mut o = View3DMut::<T>::new(args.o_map(), args.o.cast());

	let mut acc = vec![KahanAcc::<U>::new(); args.v_width];

	// These 2 `for` loops can run in parallel
	for inner_j in 0..TILE_HEIGHT {
		for qh in 0..args.head_count {
			attention_thread::<T, U, TILE_WIDTH, TILE_HEIGHT>(
				&mut o,
				&q,
				&k,
				&v,
				args.q_count,
				args.k_count,
				qh,
				qh >> args.group_shift,
				inner_j,
				&mut acc,
				args.v_width,
				args.q_width,
			);
		}
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------
