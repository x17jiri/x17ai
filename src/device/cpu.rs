//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::{DeviceAllocError, ErrPack, TensorOpError};

use super::{Device, DevicePtr};

//--------------------------------------------------------------------------------------------------

type UnderlyingElement = u64;

pub struct CPUDevice {
	name: String,
}

impl CPUDevice {
	pub fn new() -> Rc<Self> {
		Self::new_named("CPU".to_string())
	}

	pub fn new_named(name: String) -> Rc<Self> {
		Rc::new(Self { name })
	}
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn is_cpu(&self) -> bool {
		true
	}

	unsafe fn new_buffer(&self, bytes: usize) -> Result<DevicePtr, DeviceAllocError> {
		let underlying_elems = bytes.div_ceil(size_of::<UnderlyingElement>());
		let Ok(layout) = std::alloc::Layout::array::<UnderlyingElement>(underlying_elems) else {
			cold_path();
			return Err(DeviceAllocError);
		};
		unsafe {
			let Some(memory) = NonNull::new(std::alloc::alloc(layout)) else {
				cold_path();
				return Err(DeviceAllocError);
			};
			Ok(DevicePtr::new(memory.as_ptr().cast()))
		}
	}

	unsafe fn drop_buffer(&self, device_ptr: DevicePtr, bytes: usize) {
		let underlying_elems = bytes.div_ceil(size_of::<UnderlyingElement>());
		unsafe {
			let layout =
				std::alloc::Layout::array::<UnderlyingElement>(underlying_elems).unwrap_unchecked();
			let memory = device_ptr.as_ptr::<u8>();
			std::alloc::dealloc(memory, layout);
		}
	}

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: DevicePtr,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			let src: *const u8 = src.as_ptr();
			let dst: *mut u8 = dst.as_ptr::<u8>();
			std::ptr::copy_nonoverlapping(src, dst, bytes);
			Ok(())
		}
	}

	unsafe fn download_data(
		&self,
		src: DevicePtr,
		dst: NonNull<u8>,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			let src: *const u8 = src.as_ptr::<u8>();
			let dst: *mut u8 = dst.as_ptr();
			std::ptr::copy_nonoverlapping(src, dst, bytes);
			Ok(())
		}
	}
}

//--------------------------------------------------------------------------------------------------
