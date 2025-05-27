// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::ptr::NonNull;
use std::rc::Rc;

use super::buffer::{Buffer, MatrixSet, SliceSet};
use super::dtype::DType;

pub mod cpu;

pub struct AttentionParams {
	pub inputs: usize,
	pub q_heads: usize,
	pub k_heads: usize,
	pub v_heads: usize,
	pub qk_size: usize,
	pub v_size: usize,
}

pub trait Device {
	fn name(&self) -> &str;

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: usize) -> Rc<Buffer>;

	fn drop_buffer(self: Rc<Self>, device_buffer: NonNull<u8>, size_bytes: usize);

	// If any of the slices represented by a SliceSet are not in bounds,
	// these functions will panic.

	fn load_data(&self, buffer: &Buffer, dtype: DType, offset: usize, len: usize, src: &[u8]);

	fn zeros(&self, dst: &SliceSet);

	fn randn(&self, dst: &SliceSet);

	fn copy(&self, dst: &SliceSet, src: &SliceSet);

	fn acc(&self, dst: &SliceSet, dst_weight: f64, b: &SliceSet, b_weight: f64);

	fn mul(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet);

	fn mul_acc(&self, dst: &SliceSet, dst_weight: f64, a: &SliceSet, b: &SliceSet, ab_weight: f64);

	fn sub(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet);
	fn add(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet);

	fn swiglu(&self, dst: &SliceSet, lin: &SliceSet, gate: &SliceSet);
	fn swiglu_backward(
		&self, d_lin: &SliceSet, d_gate: &SliceSet, lin: &SliceSet, gate: &SliceSet,
		d_out: &SliceSet,
	);

	fn dot(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet, ab_weight: f64);

	fn dot_acc(&self, dst: &SliceSet, dst_weight: f64, a: &SliceSet, b: &SliceSet, ab_weight: f64);

	fn sum_all(&self, a: &SliceSet) -> f64;
	fn approx_eq(&self, a: &SliceSet, b: &SliceSet, eps: f64) -> bool;

	fn rsqrt(&self, dst: &SliceSet, a: &SliceSet, eps: f64);

	/// Calculates:
	///
	///    dst = max(log(a), -1000, DType.MAX_NEGATIVE);
	///
	/// So the output is defined even for a <= 0.
	fn log_clamped(&self, dst: &SliceSet, a: &SliceSet);

	fn softmax(&self, dst: &SliceSet, a: &SliceSet);

	fn rms_norm(&self, dst: &SliceSet, a: &SliceSet, eps: f64, scale_storage: Option<&SliceSet>);

	fn gemm(&self, dst: &MatrixSet, dst_weight: f64, a: &MatrixSet, b: &MatrixSet, ab_weight: f64);

	fn attention(
		&self, dst: &SliceSet, q: &SliceSet, k: &SliceSet, v: &SliceSet, params: &AttentionParams,
	);

	fn format(
		&self, f: &mut std::fmt::Formatter, buffer: &Buffer, dtype: DType, offset: usize,
		len: usize, stride: usize,
	) -> std::fmt::Result;
}
