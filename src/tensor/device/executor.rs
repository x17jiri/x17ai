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
