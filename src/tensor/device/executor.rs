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
	fn read_bin<'a>(
		&'a self,
		dst: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		src: &'a mut dyn std::io::Read,
	) -> Result<()>;

	fn write_bin(
		&self,
		src: &generic::Tensor<ND<2>, DeviceBufferRef>,
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
	fn zeros<'a, 'b, 'c>(
		&'a self,
		o: &'b mut generic::Tensor<ND<2>, DeviceBufferRefMut<'c>>,
	) -> Result<()>;

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
	fn randn_clamped<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
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
	fn copy<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
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
	fn rsqrt<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
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
	fn ln_clamped<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
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
	fn add_weighted<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		a_weight: f64,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
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
	fn mul<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
	) -> Result<()>;

	fn mul_add<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		ab_weight: f64,
		c: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		c_weight: f64,
	) -> Result<()>;

	fn mul_acc<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
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
	fn swiglu<'a>(
		&'a self,
		out: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		lin: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		gate: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
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
	fn swiglu_backward<'a>(
		&'a self,
		d_lin: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		d_gate: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		lin: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		gate: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		d_out: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
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
	fn sum_all<'a>(&'a self, a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>) -> Result<f64>;

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
	fn approx_eq<'a>(
		&'a self,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		eps: f64,
	) -> Result<bool>;

	fn softmax<'a>(
		&'a self,
		out: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		inp: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
	) -> Result<()>;

	fn dot<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		scale: f64,
	) -> Result<()>;

	fn dot_add<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		ab_weight: f64,
		c: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		c_weight: f64,
	) -> Result<()>;

	fn rsqrt_dot<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<2>, DeviceBufferRef<'a>>,
		scale: f64,
		eps: f64,
	) -> Result<()>;

	fn mm<'a>(
		&'a self,
		o: &'a mut generic::Tensor<ND<3>, DeviceBufferRefMut<'a>>,
		a: &'a generic::Tensor<ND<3>, DeviceBufferRef<'a>>,
		b: &'a generic::Tensor<ND<3>, DeviceBufferRef<'a>>,
		scale: f64,
	) -> Result<()>;

	/*
	fn attention(
		&self, dst: &generic::Tensor<ND<2>, DeviceBufferRefMut>, q: &generic::Tensor<ND<2>, DeviceBufferRef>, k: &generic::Tensor<ND<2>, DeviceBufferRef>, v: &generic::Tensor<ND<2>, DeviceBufferRef>,
		params: &AttentionParams,
	);
	*/
}
