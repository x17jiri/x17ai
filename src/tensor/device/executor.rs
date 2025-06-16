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

/// # Errors
/// - If the shapes of the tensors are not the same.
pub fn ensure_same_shape<const M: usize, const C: usize>(
	m: [&generic::Tensor<ND<2>, DeviceBufferRefMut>; M],
	c: [&generic::Tensor<ND<2>, DeviceBufferRef>; C],
) -> Result<[usize; 2]> {
	let m_shapes = array::try_map_into(m, |_, m| m.nd_shape())?;
	let c_shapes = array::try_map_into(c, |_, c| c.nd_shape())?;
	let shape = if let Some(m) = m_shapes.first() {
		m
	} else if let Some(c) = c_shapes.first() {
		c
	} else {
		return Ok([0, 0]);
	};
	if m_shapes.iter().any(|s| s != shape) || c_shapes.iter().any(|s| s != shape) {
		#[cold]
		fn err_shape_mismatch(m_shapes: &[[usize; 2]], c_shapes: &[[usize; 2]]) -> Error {
			let shapes_str = m_shapes
				.iter()
				.chain(c_shapes.iter())
				.map(|[a, b]| format!("[{a}, {b}]"))
				.collect::<Vec<_>>()
				.join(", ");
			format!("Expected all tensors to have the same shape, but got: {shapes_str}").into()
		}
		return Err(err_shape_mismatch(&m_shapes, &c_shapes));
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
	fn read_bin<'buf>(
		&self,
		dst: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		src: &mut dyn std::io::Read,
	) -> Result<()>;

	fn write_bin<'buf>(
		&self,
		src: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		dst: &mut dyn std::io::Write,
	) -> Result<()>;

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
	fn zeros<'buf>(&self, o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>) -> Result<()>;

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
	fn randn_clamped<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<()>;

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
	fn copy<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<()>;

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
	fn rsqrt<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
		eps: f64,
	) -> Result<()>;

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
	fn ln_clamped<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<()>;

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
	fn add_weighted<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		a_weight: f64,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b_weight: f64,
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
	fn mul<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<()>;

	fn mul_add<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
		c: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		c_weight: f64,
	) -> Result<()>;

	fn mul_acc<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
		o_weight: f64,
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
	fn swiglu<'buf>(
		&self,
		out: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		lin: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		gate: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<()>;

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
	fn swiglu_backward<'buf>(
		&self,
		d_lin: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		d_gate: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		lin: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		gate: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		d_out: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
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
	fn sum_all<'buf>(&self, a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>) -> Result<f64>;

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
	fn approx_eq<'buf>(
		&self,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		eps: f64,
	) -> Result<bool>;

	fn softmax_<'buf>(
		&self,
		t: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<()>;

	fn softmax<'buf>(
		&self,
		out: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		inp: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<()>;

	fn dot<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<()>;

	fn dot_add<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
		c: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		c_weight: f64,
	) -> Result<()>;

	fn rsqrt_dot<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
		eps: f64,
	) -> Result<()>;

	fn mm<'buf>(
		&self,
		o: &mut generic::Tensor<ND<3>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<3>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<3>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<()>;

	/*
	fn attention(
		&self, dst: &generic::Tensor<ND<2>, DeviceBufferRefMut>, q: &generic::Tensor<ND<2>, DeviceBufferRef>, k: &generic::Tensor<ND<2>, DeviceBufferRef>, v: &generic::Tensor<ND<2>, DeviceBufferRef>,
		params: &AttentionParams,
	);
	*/
}
