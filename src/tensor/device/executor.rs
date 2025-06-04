//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::generic::map::{CompactND, ND};
use crate::tensor::generic::{self};

use super::buffer::DeviceBuffer;

/// A single slice that doesn't have to be contiguous in memory.
pub type StridedSlice<'a> = generic::Tensor<ND<1>, &'a DeviceBuffer>;

/// A batch of slices.
///
/// They all have the same size and are contiguous in memory.
///
/// Stride is used to get from one slice in the batch to the next one.
pub type SliceBatch<'a> = generic::Tensor<CompactND<2>, &'a DeviceBuffer>;

/// A batch of matrices.
pub type MatrixBatch<'a> = generic::Tensor<ND<3>, &'a DeviceBuffer>;

pub struct AttentionParams {
	pub heads: usize,
	pub qk_features: usize,
	pub v_features: usize,

	/// `v_shift` and `k_shift` are used to implement Grouped Query Attention (GQA).
	///
	/// Each head has its own `Q`, but `K` and `V` can be shared by multiple heads.
	///
	/// For example:
	/// - we have 20 heads, so there is 20 `Q` inputs
	/// - each `K` is shared by 4 heads (k_shift = 2), so there are 5 `K` inputs
	/// - each `V` is shared by 2 heads (v_shift = 1), so there are 10 `V` inputs
	/// - for head H, we use Q[H], K[H >> k_shift], V[H >> v_shift]
	pub k_shift: usize,

	/// See `k_shift` for explanation.
	pub v_shift: usize,
}

pub trait Executor {
	// If any of the slices represented by a SliceSet are not in bounds,
	// these functions will panic.
	/*
		// These functions are designed to load/save data from files.
		// And in files, we always use little-endian format.
		// So it expects bytes to be in little-endian format.
		fn read_bin(&self, dst: &SliceBatch, src: &mut dyn std::io::Read) -> std::io::Result<()>;
		fn write_bin(&self, src: &SliceBatch, dst: &mut dyn std::io::Write) -> std::io::Result<()>;
	*/
	fn zeros(&self, dst: &SliceBatch);
	/*
		fn randn_clamped(&self, dst: &SliceBatch);

		fn copy(&self, dst: &SliceBatch, src: &SliceBatch);

		fn acc(&self, dst: &SliceBatch, dst_weight: f64, b: &SliceBatch, b_weight: f64);

		fn mul(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch);

		fn mul_acc(
			&self, dst: &SliceBatch, dst_weight: f64, a: &SliceBatch, b: &SliceBatch, ab_weight: f64,
		);

		fn sub(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch);
		fn add(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch);

		fn swiglu(&self, dst: &SliceBatch, lin: &SliceBatch, gate: &SliceBatch);
		fn swiglu_backward(
			&self, d_lin: &SliceBatch, d_gate: &SliceBatch, lin: &SliceBatch, gate: &SliceBatch,
			d_out: &SliceBatch,
		);

		fn dot(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64);

		fn dot_acc(
			&self, dst: &SliceBatch, dst_weight: f64, a: &SliceBatch, b: &SliceBatch, ab_weight: f64,
		);

		fn sum_all(&self, a: &SliceBatch) -> f64;
		fn approx_eq(&self, a: &SliceBatch, b: &SliceBatch, eps: f64) -> bool;

		fn rsqrt(&self, dst: &SliceBatch, a: &SliceBatch, eps: f64);

		/// Calculates:
		///
		///    dst = max(log(a), -1000, DType.MAX_NEGATIVE);
		///
		/// So the output is defined even for a <= 0.
		fn log_clamped(&self, dst: &SliceBatch, a: &SliceBatch);

		fn softmax(&self, dst: &SliceBatch, a: &SliceBatch);

		fn rms_norm(
			&self, dst: &SliceBatch, a: &SliceBatch, eps: f64, scale_storage: Option<&SliceBatch>,
		);

		fn gemm(
			&self, dst: &MatrixBatch, dst_weight: f64, a: &MatrixBatch, b: &MatrixBatch, ab_weight: f64,
		);

		fn attention(
			&self, dst: &SliceBatch, q: &SliceBatch, k: &SliceBatch, v: &SliceBatch,
			params: &AttentionParams,
		);

		fn format(&self, f: &mut std::fmt::Formatter, src: &StridedSlice) -> std::fmt::Result;
	*/
}
