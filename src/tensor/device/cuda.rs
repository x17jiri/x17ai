//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::device::buffer::DeviceBufferVMT;
use crate::tensor::device::cuda::cuda_shim::{CudaError, CudaStream};
use crate::tensor::device::kernel::runner::KernelRunner;
use crate::tensor::device::{DeviceBuffer, NewDeviceBufferError};
use crate::tensor::{DType, Device, HasDType};
use crate::util::mycell;

pub mod cuda_float_vmt;
pub mod cuda_shim;

use self::cuda_float_vmt::CudaFloatVMT;

//--------------------------------------------------------------------------------------------------

pub struct CudaDevice {
	name: String,
	f32_vmt: CudaFloatVMT<f32, f32>,
	cuda_stream: CudaStream,
}

impl CudaDevice {
	pub fn new() -> Result<Rc<Self>, CudaError> {
		Self::new_named("CUDA".to_string())
	}

	pub fn new_named(name: String) -> Result<Rc<Self>, CudaError> {
		let cuda_stream = CudaStream::new()?;
		let kernel_runner = Rc::new(KernelRunner::new());

		let mut rc_uninit = Rc::new_uninit();
		let instance = Self {
			name,
			f32_vmt: CudaFloatVMT::new(&rc_uninit, kernel_runner),
			cuda_stream,
		};
		unsafe {
			Rc::get_mut_unchecked(&mut rc_uninit).write(instance);
			Ok(rc_uninit.assume_init())
		}
	}

	unsafe fn drop_buffer(this: &DeviceBufferVMT, _elems: usize, device_data: NonNull<u8>) {
		unsafe {
			this.cast_device::<Self>().cuda_stream.free(device_data);

			// Recreate the `Rc` that we forgot in `new_buffer()`
			let rc_device: Rc<dyn Device> = Rc::from_raw(this.device.as_ptr());
			std::mem::drop(rc_device);
		}
	}
}

impl Device for CudaDevice {
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

		if let Some(size) = dtype.array_bytes(elems)
			&& let Ok(memory) = unsafe { self.cuda_stream.alloc(size) }
		{
			// We will recreate the `Rc` and drop it in `drop_buffer()`
			std::mem::forget(self);

			let device_data = memory;
			Ok(Rc::new(mycell::RefCell::new(unsafe {
				DeviceBuffer::new(device_data, elems, vmt.cast())
			})))
		} else {
			cold_path();
			Err(NewDeviceBufferError::AllocationFailed)
		}
	}
}
