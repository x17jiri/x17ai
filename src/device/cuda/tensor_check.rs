//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::device::Device;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::{ErrPack, TensorOpError};

use super::CudaDevice;

//--------------------------------------------------------------------------------------------------

pub(super) fn matrix_shape(
	op_name: &'static str,
	tensor_name: &'static str,
	tensor: &Tensor,
	expected_shape: [Option<usize>; 2],
	expected_device: &CudaDevice,
	expected_dtype: DType,
) -> Result<[usize; 2], ErrPack<TensorOpError>> {
	let shape = tensor.shape();
	let &[rows, cols] = shape else {
		cold_path();
		return Err(ErrPack::new(TensorOpError::Other, format!(
			"{op_name} input {tensor_name} must be 2D, got shape {shape:?}"
		)));
	};
	if rows == 0 || cols == 0 {
		cold_path();
		return Err(ErrPack::new(TensorOpError::Other, format!(
			"{op_name} input {tensor_name} has zero size {shape:?}"
		)));
	}
	if let Some(expected_rows) = expected_shape[0] && rows != expected_rows {
		cold_path();
		return Err(ErrPack::new(TensorOpError::Other, format!(
			"{op_name} input {tensor_name} has {rows} rows, but should have {expected_rows}"
		)));
	}
	if let Some(expected_cols) = expected_shape[1] && cols != expected_cols {
		cold_path();
		return Err(ErrPack::new(TensorOpError::Other, format!(
			"{op_name} input {tensor_name} has {cols} columns, but should have {expected_cols}"
		)));
	}
	tensor_dtype_and_device(op_name, tensor_name, tensor, expected_device, expected_dtype)?;
	Ok([rows, cols])
}

pub(super) fn vector_shape(
	op_name: &'static str,
	tensor_name: &'static str,
	tensor: &Tensor,
	expected_len: usize,
	expected_device: &CudaDevice,
	expected_dtype: DType,
) -> Result<(), ErrPack<TensorOpError>> {
	let shape = tensor.shape();
	match shape {
		&[len] if len == expected_len => {},
		&[len, 1] if len == expected_len => {},
		_ => {
			cold_path();
			return Err(ErrPack::new(TensorOpError::Other, format!(
				"{op_name} input {tensor_name} must have shape [{expected_len}] or [{expected_len}, 1], got {shape:?}"
			)));
		},
	}
	tensor_dtype_and_device(op_name, tensor_name, tensor, expected_device, expected_dtype)
}

fn tensor_dtype_and_device(
	op_name: &'static str,
	tensor_name: &'static str,
	tensor: &Tensor,
	expected_device: &CudaDevice,
	expected_dtype: DType,
) -> Result<(), ErrPack<TensorOpError>> {
	let dtype = tensor.dtype();
	if dtype != expected_dtype {
		cold_path();
		return Err(ErrPack::new(TensorOpError::Other, format!(
			"{op_name} input {tensor_name} has dtype {dtype}, but should have {expected_dtype}"
		)));
	}
	if !tensor.is_on_device(expected_device) {
		cold_path();
		let dev: &dyn Device = expected_device;
		let dev_name = dev.name();
		return Err(ErrPack::new(TensorOpError::Other, format!(
			"{op_name} input {tensor_name} is not on device {dev_name}"
		)));
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------
