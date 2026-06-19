//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;
use std::rc::Rc;

use cuda_shim::{CudaEventTimer, CudaStream};
use crate::{DeviceAllocError, ErrPack, TensorOpError};

use super::{Device, DevicePtr};

pub mod cuda_shim;
pub mod attn;
pub mod gemm;
pub mod gemm_templates;
mod kernel_build;

//--------------------------------------------------------------------------------------------------

pub struct CudaDevice {
	name: String,
	stream: CudaStream,
}

impl CudaDevice {
	pub fn new(device_id: usize) -> Result<Rc<Self>, ErrPack<TensorOpError>> {
		Self::new_named(device_id, format!("CUDA:{device_id}"))
	}

	pub fn new_named(device_id: usize, name: String) -> Result<Rc<Self>, ErrPack<TensorOpError>> {
		let stream = CudaStream::new(device_id)?;
		Ok(Rc::new(Self { name, stream }))
	}

	pub fn synchronize(&self) -> Result<(), ErrPack<TensorOpError>> {
		self.stream.synchronize()
	}
}

impl Device for CudaDevice {
	fn name(&self) -> &str {
		&self.name
	}

	unsafe fn new_buffer(&self, bytes: usize) -> Result<DevicePtr, DeviceAllocError> {
		unsafe { self.stream.alloc(bytes).map_err(|_| DeviceAllocError) }
	}

	unsafe fn drop_buffer(&self, device_ptr: DevicePtr, _bytes: usize) {
		unsafe { self.stream.free(device_ptr) };
	}

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: DevicePtr,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { self.stream.upload_data(src, dst, 0, bytes) }
	}

	unsafe fn download_data(
		&self,
		src: DevicePtr,
		dst: NonNull<u8>,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { self.stream.download_data(src, dst, 0, bytes) }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct CudaTimer<'a> {
	_device: &'a CudaDevice,
	timer: CudaEventTimer,
}

impl<'a> CudaTimer<'a> {
	pub fn new(device: &'a CudaDevice) -> Result<Self, ErrPack<TensorOpError>> {
		let timer = CudaEventTimer::new(&device.stream)?;
		Ok(Self { _device: device, timer })
	}

	pub fn start(&self) -> Result<(), ErrPack<TensorOpError>> {
		self.timer.start()
	}

	pub fn stop(&self) -> Result<(), ErrPack<TensorOpError>> {
		self.timer.stop()
	}

	pub fn elapsed_seconds(&self) -> Result<f64, ErrPack<TensorOpError>> {
		self.timer.elapsed_seconds()
	}
}
