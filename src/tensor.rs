//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::{cold_path};
use std::ptr::NonNull;
use std::rc::Rc;

use crate::device::{Device, DevicePtr};
use crate::dtype::{DType, UnsupportedDTypeError};
use crate::literal::TensorLiteral;
use crate::shape::ShapeHelper;
use crate::{DeviceAllocError, ErrExtra, ErrPack, ShapeOverflowError, TensorOpError};

pub struct Tensor {
	dtype: DType,
	device_is_cpu: bool,

	device_ptr: DevicePtr,
	elems: usize,
	bytes: usize,

	device: Rc<dyn Device>,

	shape: Box<[usize]>
}

impl Tensor {
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn ndim(&self) -> usize {
		self.shape.len()
	}

	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	pub fn elems(&self) -> usize {
		self.elems
	}

	pub fn bytes(&self) -> usize {
		self.bytes
	}

	pub fn is_on_cpu(&self) -> bool {
		self.device_is_cpu
	}

	#[inline(never)]
	pub fn new(
		literal: &dyn TensorLiteral,
		device: Rc<dyn Device>,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let result = Self::new_empty(literal.shape(), literal.dtype(), device)?;
		unsafe {
			result.device.upload_data(
				NonNull::from_ref(literal.data()).cast(),
				result.device_ptr,
				literal.data().len()
			)?;
		}
		Ok(result)
	}

	#[inline(never)]
	pub fn from_safetensors_view(
		view: &safetensors::tensor::TensorView<'_>,
		device: Rc<dyn Device>,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let Ok(dtype) = DType::try_from(safetensors::tensor::View::dtype(&view)) else {
			cold_path();
			return Err(UnsupportedDTypeError.into());
		};

		let data = safetensors::tensor::View::data(&view);
		let result = Self::new_empty(safetensors::tensor::View::shape(&view), dtype, device)?;
		debug_assert_eq!(data.len(), result.bytes());

		unsafe {
			result.device.upload_data(
				NonNull::from_ref(data.as_ref()).cast(),
				result.device_ptr,
				result.bytes(),
			)?;
		}
		Ok(result)
	}

	#[inline(never)]
	pub fn from_safetensors_file(
		filename: impl AsRef<std::path::Path>,
		device: Rc<dyn Device>,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let file = std::fs::File::open(filename)?;
		// SAFETY: The mapping is read-only and kept alive until after the tensor bytes are uploaded.
		let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
		let safetensors = safetensors::SafeTensors::deserialize(&mmap)?;
		let mut tensors = safetensors.tensors();
		if tensors.len() != 1 {
			cold_path();
			return Err(ErrPack {
				code: TensorOpError::InvalidSafeTensors,
				extra: Some(Box::new(ErrExtra {
					message: format!("expected exactly one tensor, found {}", tensors.len()).into(),
					nested: None,
				})),
			});
		}
		let Some((_name, view)) = tensors.pop() else {
			cold_path();
			return Err(ErrPack {
				code: TensorOpError::InvalidSafeTensors,
				extra: Some(Box::new(ErrExtra {
					message: "expected exactly one tensor, found 0".into(),
					nested: None,
				})),
			});
		};
		Self::from_safetensors_view(&view, device)
	}

	#[inline(never)]
	pub fn new_empty(
		shape: &[usize],
		dtype: DType,
		device: Rc<dyn Device>,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let Ok(shape_helper) = ShapeHelper::new(dtype, shape) else {
			cold_path();
			return Err(ShapeOverflowError.into());
		};

		let elems = shape_helper.elems();
		let bytes = shape_helper.bytes();
		let device_is_cpu = device.is_cpu();

		let Ok(device_ptr) = (unsafe { device.new_buffer(bytes) }) else {
			cold_path();
			return Err(DeviceAllocError.into());
		};

		Ok(Self {
			dtype,
			device_is_cpu,
			device_ptr,
			elems,
			bytes,
			device,
			shape: shape.into(),
		})
	}
}

impl Drop for Tensor {
	fn drop(&mut self) {
		unsafe { self.device.drop_buffer(self.device_ptr, self.bytes) };
	}
}
