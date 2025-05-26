// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

pub mod cpu;

use std::fmt;
use std::rc::Rc;

use super::TensorSize;
use super::buffer::{Buffer, MatrixSet, SliceSet};
use super::dtype::DType;

pub struct AttentionParams {
	pub inputs: TensorSize,
	pub q_heads: TensorSize,
	pub k_heads: TensorSize,
	pub v_heads: TensorSize,
	pub qk_size: TensorSize,
	pub v_size: TensorSize,
}

pub trait Device {
	fn name(&self) -> &str;

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: TensorSize) -> Rc<Buffer>;

	// This function must manually drop buffer.device
	fn drop_buffer(&self, buffer: &mut Buffer);

	// If any of the slices represented by a SliceSet are not in bounds,
	// these functions will panic.

	fn load_data(
		&self, buffer: &Buffer, dtype: DType, offset: TensorSize, len: TensorSize, src: &[u8],
	);

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
		&self, f: &mut fmt::Formatter, dtype: DType, offset: TensorSize, len: TensorSize,
		stride: TensorSize,
	) -> fmt::Result;
}
