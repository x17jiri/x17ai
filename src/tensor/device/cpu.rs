//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::HasDType;
use crate::tensor::device::buffer::DeviceBufferVMT;
use crate::tensor::device::kernel::runner::KernelRunner;
use crate::util::mycell;

pub mod attention;
pub mod cpu_float_vmt;
pub mod math;

use crate::tensor::device::cpu::cpu_float_vmt::CPUFloatVMT;
use crate::tensor::device::{DeviceBuffer, NewDeviceBufferError};
use crate::tensor::{DType, Device};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ViewError {
	InvalidDType,
	NotOnCPUDevice,
}

//--------------------------------------------------------------------------------------------------

pub struct CPUDevice {
	name: String,
	f32_vmt: CPUFloatVMT<f32, f64>,
}

impl CPUDevice {
	pub fn new() -> Rc<Self> {
		Self::new_named("CPU".to_string())
	}

	pub fn new_named(name: String) -> Rc<Self> {
		let kernel_runner = Rc::new(KernelRunner::new());

		let mut rc_uninit = Rc::new_uninit();
		let instance = Self {
			name,
			f32_vmt: CPUFloatVMT::new(&rc_uninit, kernel_runner),
		};
		unsafe {
			Rc::get_mut_unchecked(&mut rc_uninit).write(instance);
			rc_uninit.assume_init()
		}
	}

	pub fn ensure_can_view<T: HasDType>(buf: &DeviceBuffer) -> Result<(), ViewError> {
		if buf.dtype() != T::dtype {
			cold_path();
			return Err(ViewError::InvalidDType);
		}
		debug_assert!(T::dtype.bytes() == std::mem::size_of::<T>());
		if !buf.vmt().device_is_cpu() {
			cold_path();
			return Err(ViewError::NotOnCPUDevice);
		}
		Ok(())
	}

	unsafe fn drop_buffer(this: NonNull<DeviceBufferVMT>, elems: usize, device_data: NonNull<u8>) {
		unsafe {
			let this = this.as_ref();
			let dtype = this.dtype();
			let device_ptr = this.device_ptr();

			let align = dtype.bytes().min(1);
			let size = dtype.array_bytes(elems).unwrap();
			let layout = std::alloc::Layout::from_size_align(size, align).unwrap_unchecked();
			std::alloc::dealloc(device_data.as_ptr(), layout);

			// Recreate the `Rc` that we forgot in `new_buffer()`
			let rc_device: Rc<dyn Device> = Rc::from_raw(device_ptr.as_ptr());
			std::mem::drop(rc_device);
		}
	}
}

impl Device for CPUDevice {
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

		let align = dtype.bytes().min(1);
		if let Some(size) = dtype.array_bytes(elems)
			&& let Ok(layout) = std::alloc::Layout::from_size_align(size, align)
			&& let Some(memory) = NonNull::new(unsafe { std::alloc::alloc(layout) })
		{
			// We will recreate the `Rc` and drop it in `CPUDevice::drop_buffer()`
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

//--------------------------------------------------------------------------------------------------
