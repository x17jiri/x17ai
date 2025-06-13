//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};
use crate::tensor::generic::map::ND;
use crate::tensor::generic::{self};
use crate::util::array;
use crate::{Error, Result};

use super::buffer::DeviceBuffer;

/// A batch of slices.
pub type SliceBatch<'a> = generic::Tensor<ND<2>, &'a DeviceBuffer>;
/// An immutable borrow of `SliceBatch`.
pub type SliceBatchRef<'a> = generic::Tensor<ND<2>, DeviceBufferRef<'a>>;
/// A mutable borrow of `SliceBatch`.
pub type SliceBatchRefMut<'a> = generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>;

/// A batch of matrices.
pub type MatrixBatch<'a> = generic::Tensor<ND<3>, &'a DeviceBuffer>;
/// An immutable borrow of `MatrixBatch`.
pub type MatrixBatchRef<'a> = generic::Tensor<ND<3>, DeviceBufferRef<'a>>;
/// A mutable borrow of `MatrixBatch`.
pub type MatrixBatchRefMut<'a> = generic::Tensor<ND<3>, DeviceBufferRefMut<'a>>;

/// # Errors
/// - If the shapes of the tensors are not the same.
pub fn ensure_same_shape<const N: usize>(t: [&SliceBatch; N]) -> Result<[usize; 2]> {
	let shapes = array::try_map_into(t, |_, t| t.nd_shape())?;
	let shape = shapes.first().unwrap_or(&[0, 0]);
	if shapes.iter().any(|s| s != shape) {
		#[cold]
		fn err_shape_mismatch<const N: usize>(shapes: &[[usize; 2]]) -> Error {
			let shapes_str =
				shapes.iter().map(|[a, b]| format!("[{a}, {b}]")).collect::<Vec<_>>().join(", ");
			format!("Expected all tensors to have the same shape, but got: {shapes_str}").into()
		}
		return Err(err_shape_mismatch::<N>(&shapes));
	}
	Ok(*shape)
}

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
	// These functions are designed to load/save data from files.
	// And in files, we always use little-endian format.
	// So it expects bytes to be in little-endian format.
	fn read_bin(&self, dst: &SliceBatch, src: &mut dyn std::io::Read) -> Result<()>;
	fn write_bin(&self, src: &SliceBatch, dst: &mut dyn std::io::Write) -> Result<()>;

	/// Fills the `o` tensor with zeros.
	///
	///     o[i] = 0.0;
	///
	/// # Requrements
	/// - The `o` tensor has to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	///   - have contiguous dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn zeros(&self, o: &SliceBatch) -> Result<()>;

	/// Fills the `o` tensor with random values from a normal distribution
	/// with mean 0 and variance 1.
	///
	/// The values are clamped to the range (-10.0, +10.0)
	///
	/// # Requrements
	/// - The `o` tensor has to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	///   - have contiguous dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn randn_clamped(&self, o: &SliceBatch) -> Result<()>;

	/// Copies data from `a` to `o`:
	///
	///     o[i] = a[i];
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `o` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn copy(&self, o: &SliceBatch, a: &SliceBatch) -> Result<()>;

	/// Element-wise unary operation:
	///
	///    o[i] = 1.0 / sqrt(a[i] * scale + eps);
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `o` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn rsqrt(&self, o: &SliceBatch, a: &SliceBatch, scale: f64, eps: f64) -> Result<()>;

	/// Element-wise unary operation:
	///
	///     o[i] = max(ln(a[i]), -1000, DType.MAX_NEGATIVE);
	///
	/// So the output is defined even for a <= 0.
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `o` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn ln_clamped(&self, o: &SliceBatch, a: &SliceBatch) -> Result<()>;

	/// Element-wise weighted addition:
	///
	///     o[i] = (a[i] * a_weight) + (b[i] * b_weight)
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `o` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn add_weighted(
		&self, o: &SliceBatch, a: &SliceBatch, a_weight: f64, b: &SliceBatch, b_weight: f64,
	) -> Result<()>;

	/// Element-wise multiplication:
	///
	///     o[i] = a[j, i] * b[j, i]
	///
	/// # Requrements
	/// - All input tensors have to:
	///   - have a safe map
	///   - have dtype corresponding to this Executor
	///   - be on the device corresponding to this Executor
	/// - The `i` tensor has to:
	///   - have contiguous dimension 1
	/// - Other tensors have to:
	///   - have either contiguous or broadcasted dimension 1
	///
	/// # Errors
	/// - If any of the requirements is not met.
	/// - If there is any problem executing the operation on the device.
	fn mul(&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch) -> Result<()>;

	fn mul_add(
		&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64, c: &SliceBatch,
		c_weight: f64,
	) -> Result<()>;

	/// The SwiGLU activation function:
	///
	///     out[j, i] = lin[j, i] * sigmoid(gate[j, i])
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
	fn swiglu(&self, out: &SliceBatch, lin: &SliceBatch, gate: &SliceBatch) -> Result<()>;

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

	fn softmax(&self, out: &SliceBatch, inp: &SliceBatch) -> Result<()>;

	fn dot(&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, scale: f64) -> Result<()>;

	fn dot_add(
		&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, ab_weight: f64, c: &SliceBatch,
		c_weight: f64,
	) -> Result<()>;

	fn rsqrt_dot(
		&self, o: &SliceBatch, a: &SliceBatch, b: &SliceBatch, scale: f64, eps: f64,
	) -> Result<()>;

	fn mm(&self, o: &MatrixBatch, a: &MatrixBatch, b: &MatrixBatch, scale: f64) -> Result<()>;

	/*
	fn attention(
		&self, dst: &SliceBatch, q: &SliceBatch, k: &SliceBatch, v: &SliceBatch,
		params: &AttentionParams,
	);
	*/
}
