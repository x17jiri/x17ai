//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;
use std::hint::cold_path;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::device::buffer::DeviceBufferVMT;
use crate::tensor::device::{DeviceBuffer, NewDeviceBufferError};
use crate::tensor::{DType, Device, HasDType};
use crate::util::mycell;

pub mod cuda_float_vmt;
pub mod cuda_shim;

use self::cuda_float_vmt::CUDAFloatVMT;

//--------------------------------------------------------------------------------------------------

pub struct CUDADevice {
	pub name: String,
	pub f32_vmt: CUDAFloatVMT<f32>,
}

impl CUDADevice {
	pub fn new() -> Result<Rc<Self>, cuda_shim::CudaInitError> {
		Self::new_named("CUDA".to_string())
	}

	pub fn new_named(name: String) -> Result<Rc<Self>, cuda_shim::CudaInitError> {
		cuda_shim::init()?;
		Ok(Rc::new(Self { name, f32_vmt: CUDAFloatVMT::new() }))
	}

	unsafe fn drop_buffer(this: NonNull<DeviceBufferVMT>, elems: usize, device_data: *mut u8) {
		unsafe {
			let this = this.as_ref();
			let device_ptr = this.device_ptr();

			cuda_shim::free(device_data);

			// Recreate the `Rc` that we forgot in `new_buffer()`
			let rc_device: Rc<dyn Device> = Rc::from_raw(device_ptr.as_ptr());
			std::mem::drop(rc_device);
		}
	}
}

impl Device for CUDADevice {
	fn name(&self) -> &str {
		&self.name
	}

	#[inline(never)]
	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<mycell::RefCell<DeviceBuffer>>, NewDeviceBufferError> {
		#[allow(clippy::single_match_else)]
		let vmt = match dtype {
			f32::dtype => NonNull::from_ref(&self.f32_vmt),
			_ => {
				cold_path();
				return Err(NewDeviceBufferError::UnsupportedDType);
			},
		};

		let memory = unsafe { cuda_shim::alloc(dtype.array_bytes(elems).unwrap()) };
		let Ok(memory) = memory else {
			cold_path();
			return Err(NewDeviceBufferError::AllocationFailed);
		};

		// We will recreate the `Rc` and drop it in `CPUDevice::drop_buffer()`
		std::mem::forget(self);

		let device_data = memory;
		Ok(Rc::new(mycell::RefCell::new(unsafe {
			DeviceBuffer::new(device_data, elems, vmt.cast())
		})))
	}
}
