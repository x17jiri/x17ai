//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::Result;
use crate::tensor::generic::map::ND;
use crate::tensor::generic::{self};

use super::buffer::DeviceBuffer;

/// A single slice that doesn't have to be contiguous in memory.
pub type StridedSlice<'a> = generic::Tensor<ND<1>, &'a DeviceBuffer>;

/// A batch of slices.
///
/// They all have the same size and are contiguous in memory.
///
/// Stride is used to get from one slice in the batch to the next one.
pub type SliceBatch<'a> = generic::Tensor<ND<2>, &'a DeviceBuffer>;

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
	/// - each `K` is shared by 4 heads `(k_shift = 2)`, so there are 5 `K` inputs
	/// - each `V` is shared by 2 heads `(v_shift = 1)`, so there are 10 `V` inputs
	/// - for head `H`, we use `Q[H]`, `K[H >> k_shift]`, `V[H >> v_shift]`
	pub k_shift: usize,

	/// See `k_shift` for explanation.
	pub v_shift: usize,
}

pub trait Executor {
	/*
		// These functions are designed to load/save data from files.
		// And in files, we always use little-endian format.
		// So it expects bytes to be in little-endian format.
		fn read_bin(&self, dst: &SliceBatch, src: &mut dyn std::io::Read) -> std::io::Result<()>;
		fn write_bin(&self, src: &SliceBatch, dst: &mut dyn std::io::Write) -> std::io::Result<()>;
	*/

	/// Fills the `dst` tensor with zeros.
	///
	/// # Requrements
	/// - The `dst` tensor has to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	///   - have contiguous dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn zeros(&self, dst: &SliceBatch) -> Result<()>;

	/// Fills the `dst` tensor with random values from a normal distribution
	/// with mean 0 and variance 1. The values are clamped to the range (-10.0, +10.0)
	///
	/// # Requrements
	/// - The `dst` tensor has to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	///   - have contiguous dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn randn_clamped(&self, dst: &SliceBatch) -> Result<()>;

	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `dst` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn copy(&self, dst: &SliceBatch, src: &SliceBatch) -> Result<()>;

	/// Element-wise accumulation:
	///
	///     dst[j, i] = dst[j, i] * dst_weight + upd[j, i] * upd_weight
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `dst` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn acc(
		&self, dst: &SliceBatch, dst_weight: f64, upd: &SliceBatch, upd_weight: f64,
	) -> Result<()>;

	/// Element-wise unary operation:
	///
	///    dst = 1.0 / sqrt(a + eps);
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `dst` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn rsqrt(&self, dst: &SliceBatch, a: &SliceBatch, eps: f64) -> Result<()>;

	/// Element-wise unary operation:
	///
	///     dst = max(ln(a), -1000, DType.MAX_NEGATIVE);
	///
	/// So the output is defined even for a <= 0.
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `dst` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn ln_clamped(&self, dst: &SliceBatch, a: &SliceBatch) -> Result<()>;

	/// Element-wise multiplication:
	///
	///     dst[j, i] = a[j, i] * b[j, i]
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `dst` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn mul(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()>;

	/// Element-wise multiplication with accumulation:
	///
	///     dst[j, i] = dst[j, i] * dst_weight + a[j, i] * b[j, i] * ab_weight
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `dst` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn mul_acc(
		&self, dst: &SliceBatch, dst_weight: f64, a: &SliceBatch, b: &SliceBatch, ab_weight: f64,
	) -> Result<()>;

	/// Element-wise subtraction:
	///
	///     dst[j, i] = a[j, i] - b[j, i]
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `dst` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn sub(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()>;

	/// Element-wise addition:
	///
	///     dst[j, i] = a[j, i] + b[j, i]
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `dst` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn add(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()>;

	/// The SwiGLU activation function:
	///
	///     dst[j, i] = lin[j, i] * sigmoid(gate[j, i])
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	///   - have contiguous dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	#[allow(clippy::doc_markdown)]
	fn swiglu(&self, dst: &SliceBatch, lin: &SliceBatch, gate: &SliceBatch) -> Result<()>;

	/// Backward pass for the SwiGLU activation function:
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	///   - have contiguous dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	#[allow(clippy::doc_markdown)]
	fn swiglu_backward(
		&self, d_lin: &SliceBatch, d_gate: &SliceBatch, lin: &SliceBatch, gate: &SliceBatch,
		d_out: &SliceBatch,
	) -> Result<()>;

	/// Sums all elements in the `a` tensor and returns the result.
	///
	/// # Requrements
	/// - The `a` tensor has to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn sum_all(&self, a: &SliceBatch) -> Result<f64>;

	/// Checks if two tensors are approximately equal element-wise:
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn approx_eq(&self, a: &SliceBatch, b: &SliceBatch, eps: f64) -> Result<bool>;

	fn softmax(&self, dst: &SliceBatch, a: &SliceBatch) -> Result<()>;

	fn rms_norm(&self, dst: &SliceBatch, a: &SliceBatch, eps: f64) -> Result<()>;

	/*
	fn dot(&self, dst: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64) -> Result<()>;

		fn dot_acc(
			&self, dst: &SliceBatch, dst_weight: f64, a: &SliceBatch, b: &SliceBatch, ab_weight: f64,
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
