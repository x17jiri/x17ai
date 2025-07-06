//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};
use crate::tensor::device::cpu::ViewError;
use crate::tensor::generic::map::ND;
use crate::tensor::generic::{self, TensorUnsafeError};
use crate::{ErrExtra, ErrPack};

//--------------------------------------------------------------------------------------------------

/// # Errors
/// - If the shapes of the tensors are not the same.
pub fn ensure_same_shape<const M: usize, const C: usize>(
	m: [&generic::Tensor<ND<2>, DeviceBufferRefMut>; M],
	c: [&generic::Tensor<ND<2>, DeviceBufferRef>; C],
) -> Result<[usize; 2], ErrPack<ExecutorError>> {
	let m_shapes = m.try_map(generic::Tensor::nd_shape)?;
	let c_shapes = c.try_map(generic::Tensor::nd_shape)?;
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

//--------------------------------------------------------------------------------------------------

pub trait Executor {
	/// `read_bin()` and `write_bin()` are designed to load/save data from files.
	/// And in files, we always use little-endian format.
	/// So they expect bytes to be in little-endian format.
	fn read_bin<'buf>(
		&self,
		dst: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		src: &mut dyn std::io::Read,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// `read_bin()` and `write_bin()` are designed to load/save data from files.
	/// And in files, we always use little-endian format.
	/// So they expect bytes to be in little-endian format.
	fn write_bin<'buf>(
		&self,
		src: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		dst: &mut dyn std::io::Write,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Fills the `o` tensor with zeros.
	///
	///     o[i] = 0.0;
	fn zeros<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Fills the `o` tensor with random values from a normal distribution
	/// with mean 0 and variance 1.
	///
	/// The values are clamped to the range (-10.0, +10.0)
	fn randn_clamped<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Copies data from `a` to `o`:
	///
	///     o[i] = a[i];
	fn copy<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Element-wise unary operation:
	///
	///    o[i] = 1.0 / (sqrt(a[i] * scale) + eps);
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
	fn ln_clamped<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Element-wise weighted addition:
	///
	///     o[i] = (a[i] * a_weight) + (b[i] * b_weight)
	fn add_weighted<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		a_weight: f64,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Element-wise weighted addition:
	///
	///     a[i] = (a[i] * a_weight) + (b[i] * b_weight)
	fn acc_weighted<'buf>(
		&self,
		a: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a_weight: f64,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Element-wise multiplication:
	///
	///     o[i] = a[i] * b[i]
	fn mul<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Element-wise multiplication:
	///
	///     a[i] = a[i] * b[i]
	fn mul_<'buf>(
		&self,
		a: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Element-wise mul-add:
	///
	///     o[i] = (a[i] * b[i] * ab_weight) + (c[i] * c_weight)
	fn mul_add<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
		c: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		c_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// Element-wise mul-add:
	///
	///     o[i] = (a[i] * b[i] * ab_weight) + (o[i] * o_weight)
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
	///     out[i] = lin[i] * sigmoid(gate[i])
	#[allow(clippy::doc_markdown)]
	fn swiglu<'buf>(
		&self,
		out: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		lin: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		gate: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn swiglu_backward<'buf>(
		&self,
		d_lin_gate: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		swapped: bool,
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

	fn dot_acc<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		o_weight: f64,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		ab_weight: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn rsqrt_dot<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
		eps: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn sqrt_dot<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	fn mm<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<(), ErrPack<ExecutorError>>;

	/// If `k_heads != qo_heads`, there has to be `k_shift` such that:
	///     `qo_heads = k_heads << k_shift`
	/// The same applies to `v_head`.
	fn attention(
		&self,
		o: &mut generic::Tensor<ND<3>, DeviceBufferRefMut>, // [inputs, qo_heads, vo_features]
		q: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, qo_heads, qk_features]
		k: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, k_heads, qk_features]
		v: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, v_heads, vo_features]
	);
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutorError {
	ShapeMismatch,
	UnsafeTensor,
	NotContiguous,
	NotContiguousOrBroadcasted,
	InvalidShape,
	IOError,
	InvalidDType,
	InvalidDevice,
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
		ErrPack {
			code: Self::ShapeMismatch,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}

	#[cold]
	#[inline(never)]
	pub fn not_contiguous() -> ErrPack<Self> {
		let message = "Expected the tensor to have contiguous dimension -1, but it does not".into();
		ErrPack {
			code: Self::NotContiguous,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}

	#[cold]
	#[inline(never)]
	pub fn not_contiguous_or_broadcasted() -> ErrPack<Self> {
		let message =
			"Expected the tensor to have contiguous or broadcasted dimension -1, but it does not"
				.into();
		ErrPack {
			code: Self::NotContiguousOrBroadcasted,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}

	#[cold]
	#[inline(never)]
	pub fn invalid_shape(shape: [usize; 2], expected: [usize; 2]) -> ErrPack<Self> {
		let message = format!("Tensor shape {shape:?} does not match expected shape {expected:?}");
		ErrPack {
			code: Self::InvalidShape,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}

	#[cold]
	#[inline(never)]
	pub fn io_error(err: std::io::Error) -> ErrPack<Self> {
		ErrPack {
			code: Self::IOError,
			extra: Some(Box::new(ErrExtra {
				message: String::new(),
				nested: Some(Box::new(err)),
			})),
		}
	}
}

impl From<TensorUnsafeError> for ExecutorError {
	fn from(_: TensorUnsafeError) -> Self {
		Self::UnsafeTensor
	}
}

impl From<ErrPack<TensorUnsafeError>> for ErrPack<ExecutorError> {
	fn from(err: ErrPack<TensorUnsafeError>) -> Self {
		Self { code: err.code.into(), extra: err.extra }
	}
}

impl From<std::io::Error> for ErrPack<ExecutorError> {
	fn from(err: std::io::Error) -> Self {
		ExecutorError::io_error(err)
	}
}

impl From<ViewError> for ExecutorError {
	#[cold]
	#[inline(never)]
	fn from(err: ViewError) -> Self {
		match err {
			ViewError::InvalidDType => Self::InvalidDType,
			ViewError::NotOnCPUDevice => Self::InvalidDevice,
		}
	}
}

impl From<ViewError> for ErrPack<ExecutorError> {
	#[cold]
	#[inline(never)]
	fn from(err: ViewError) -> Self {
		Self {
			code: ExecutorError::from(err),
			extra: None,
		}
	}
}

//--------------------------------------------------------------------------------------------------
