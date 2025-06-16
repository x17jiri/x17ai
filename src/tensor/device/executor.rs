//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};
use crate::tensor::generic::map::ND;
use crate::tensor::generic::{self, TensorUnsafeError};
use crate::util::array;
use crate::{ErrExtra, ErrPack};

pub enum ExecutorError {
	ShapeMismatch,
	UnsafeTensor,
	NotContiguous,
	NotContiguousOrBroadcasted,
	InvalidShape,
	IOError,
}

impl ExecutorError {
	#[cold]
	#[inline(never)]
	pub fn shape_mismatch(m_shapes: &[[usize; 2]], c_shapes: &[[usize; 2]]) -> ErrPack<Self> {
		let shapes = m_shapes
			.iter()
			.chain(c_shapes.iter())
			.map(|[a, b]| format!("[{a}, {b}]"))
			.collect::<Vec<_>>()
			.join(", ");
		let message = format!("Expected all tensors to have the same shape, but got: {shapes}");
		let result = ErrPack {
			code: Self::ShapeMismatch,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		};
		result
	}

	#[cold]
	#[inline(never)]
	pub fn not_contiguous() -> ErrPack<Self> {
		let message = "Expected the tensor to have contiguous dimension -1, but it does not".into();
		let result = ErrPack {
			code: Self::NotContiguous,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		};
		result
	}

	#[cold]
	#[inline(never)]
	pub fn not_contiguous_or_broadcasted() -> ErrPack<Self> {
		let message =
			"Expected the tensor to have contiguous or broadcasted dimension -1, but it does not"
				.into();
		let result = ErrPack {
			code: Self::NotContiguousOrBroadcasted,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		};
		result
	}

	#[cold]
	#[inline(never)]
	pub fn invalid_shape(shape: [usize; 2], expected: [usize; 2]) -> ErrPack<Self> {
		let message =
			format!("Tensor shape {:?} does not match expected shape {:?}", shape, expected);
		let result = ErrPack {
			code: Self::InvalidShape,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		};
		result
	}

	#[cold]
	#[inline(never)]
	pub fn io_error(err: std::io::Error) -> ErrPack<Self> {
		let result = ErrPack {
			code: Self::IOError,
			extra: Some(Box::new(ErrExtra {
				message: String::new(),
				nested: Some(Box::new(err)),
			})),
		};
		result
	}
}

impl From<TensorUnsafeError> for ExecutorError {
	fn from(_: TensorUnsafeError) -> Self {
		Self::UnsafeTensor
	}
}

impl From<ErrPack<TensorUnsafeError>> for ErrPack<ExecutorError> {
	fn from(err: ErrPack<TensorUnsafeError>) -> Self {
		ErrPack { code: err.code.into(), extra: err.extra }
	}
}

impl From<std::io::Error> for ErrPack<ExecutorError> {
	fn from(err: std::io::Error) -> Self {
		ExecutorError::io_error(err)
	}
}

/// # Errors
/// - If the shapes of the tensors are not the same.
pub fn ensure_same_shape<const M: usize, const C: usize>(
	m: [&generic::Tensor<ND<2>, DeviceBufferRefMut>; M],
	c: [&generic::Tensor<ND<2>, DeviceBufferRef>; C],
) -> Result<[usize; 2], ErrPack<ExecutorError>> {
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
		return Err(ExecutorError::shape_mismatch(&m_shapes, &c_shapes));
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
	) -> Result<(), ErrPack<ExecutorError>>;

	fn write_bin<'buf>(
		&self,
		src: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		dst: &mut dyn std::io::Write,
	) -> Result<(), ErrPack<ExecutorError>>;

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
	fn zeros<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

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
	) -> Result<(), ErrPack<ExecutorError>>;

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
	) -> Result<(), ErrPack<ExecutorError>>;

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
	) -> Result<(), ErrPack<ExecutorError>>;

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
	) -> Result<(), ErrPack<ExecutorError>>;

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
	) -> Result<(), ErrPack<ExecutorError>>;

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
	) -> Result<(), ErrPack<ExecutorError>>;

	fn mul_add<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
		c: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		c_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn mul_acc<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
		o_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

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
	) -> Result<(), ErrPack<ExecutorError>>;

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
	) -> Result<(), ErrPack<ExecutorError>>;

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
	fn sum_all<'buf>(
		&self,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<f64, ErrPack<ExecutorError>>;

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
	) -> Result<bool, ErrPack<ExecutorError>>;

	fn softmax_<'buf>(
		&self,
		t: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn softmax<'buf>(
		&self,
		out: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		inp: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn dot<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn dot_add<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
		c: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		c_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn rsqrt_dot<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
		eps: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn mm<'buf>(
		&self,
		o: &mut generic::Tensor<ND<3>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<3>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<3>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	/*
	fn attention(
		&self, dst: &generic::Tensor<ND<2>, DeviceBufferRefMut>, q: &generic::Tensor<ND<2>, DeviceBufferRef>, k: &generic::Tensor<ND<2>, DeviceBufferRef>, v: &generic::Tensor<ND<2>, DeviceBufferRef>,
		params: &AttentionParams,
	);
	*/
}
