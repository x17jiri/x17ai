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

use crate::tensor::device::kernel::library::KernelLibrary;
use crate::tensor::device::{DeviceBuffer, NewDeviceBufferError};
use crate::tensor::{DType, Device, HasDType};

pub mod cuda_float_executor;
pub mod cuda_shim;

use self::cuda_float_executor::CUDAFloatExecutor;

//--------------------------------------------------------------------------------------------------

pub struct CUDADevice {
	pub name: String,
	pub f32_executor: CUDAFloatExecutor<f32>,
}

impl CUDADevice {
	pub fn new() -> Result<Rc<Self>, cuda_shim::CudaInitError> {
		cuda_shim::cuda_init()?;
		Ok(Self::new_named("CUDA".to_string()))
	}

	pub fn new_named(name: String) -> Rc<Self> {
		Rc::new(Self {
			name,
			f32_executor: CUDAFloatExecutor::new(),
		})
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
	) -> Result<Rc<DeviceBuffer>, NewDeviceBufferError> {
		#[allow(clippy::single_match_else)]
		let executor = match dtype {
			f32::dtype => &self.f32_executor,
			_ => {
				cold_path();
				return Err(NewDeviceBufferError::UnsupportedDType);
			},
		};
		let memory = unsafe { cuda_shim::cuda_alloc(dtype.array_bytes(elems).unwrap()) };
		let Some(memory) = NonNull::new(memory) else {
			cold_path();
			return Err(NewDeviceBufferError::AllocationFailed);
		};
		Ok(Rc::new(DeviceBuffer {
			executor: NonNull::from(executor),
			dtype,
			elems,
			device_data: memory.as_ptr(),
			device: ManuallyDrop::new(self.clone()),
			device_is_cpu: false,
			builtin_kernels: KernelLibrary::instance(),
			read_count: Cell::new(0),
			write_count: Cell::new(0),
		}))
	}

	unsafe fn drop_buffer(self: Rc<Self>, _dtype: DType, _elems: usize, device_data: *mut u8) {
		unsafe { x17ai_cuda_free(device_data) };
	}
}
