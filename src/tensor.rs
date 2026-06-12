//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::hint::cold_path;
use std::path::Path;
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

	shape: Box<[usize]>,
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

	pub fn device_ptr(&self) -> DevicePtr {
		self.device_ptr
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
				literal.data().len(),
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

	#[inline(never)]
	pub fn save_safetensors_file(
		&self,
		filename: impl AsRef<Path>,
	) -> Result<(), ErrPack<TensorOpError>> {
		let path = filename.as_ref();
		if let Some(parent) = path.parent() {
			if !parent.as_os_str().is_empty() {
				std::fs::create_dir_all(parent).map_err(|err| {
					tensor_io_error(
						format!("failed to create tensor output directory {}", parent.display()),
						err,
					)
				})?;
			}
		}

		if self.device_is_cpu {
			let data = if self.bytes() == 0 {
				&[]
			} else {
				unsafe { std::slice::from_raw_parts(self.device_ptr.as_ptr::<u8>(), self.bytes()) }
			};
			self.save_safetensors_data(path, data)
		} else {
			let mut data = vec![0_u8; self.bytes()];
			unsafe {
				self.device.download_data(
					self.device_ptr,
					NonNull::new_unchecked(data.as_mut_ptr()),
					self.bytes(),
				)?;
			}
			self.save_safetensors_data(path, &data)
		}
	}

	fn save_safetensors_data(
		&self,
		path: &Path,
		data: &[u8],
	) -> Result<(), ErrPack<TensorOpError>> {
		let view = safetensors::tensor::TensorView::new(
			safetensors_dtype(self.dtype),
			self.shape.to_vec(),
			data,
		)?;
		safetensors::serialize_to_file([("tensor", view)], &None, path)?;
		Ok(())
	}
}

impl Drop for Tensor {
	fn drop(&mut self) {
		unsafe { self.device.drop_buffer(self.device_ptr, self.bytes) };
	}
}

fn safetensors_dtype(dtype: DType) -> safetensors::tensor::Dtype {
	match dtype {
		DType::E4m3 => safetensors::tensor::Dtype::F8_E4M3,
		DType::Int8 => safetensors::tensor::Dtype::I8,
		DType::Bf16 => safetensors::tensor::Dtype::BF16,
		DType::F32 => safetensors::tensor::Dtype::F32,
	}
}

fn tensor_io_error(
	message: impl Into<Cow<'static, str>>,
	err: std::io::Error,
) -> ErrPack<TensorOpError> {
	ErrPack {
		code: TensorOpError::IOError,
		extra: Some(Box::new(ErrExtra {
			message: message.into(),
			nested: Some(Box::new(err)),
		})),
	}
}
