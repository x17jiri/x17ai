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
use crate::util::mycell::{self, BorrowGuard};

pub mod attention;
pub mod cpu_float_vmt;
pub mod math;

use crate::tensor::device::cpu::cpu_float_vmt::CPUFloatVMT;
use crate::tensor::device::{DeviceBuffer, NewDeviceBufferError};
use crate::tensor::{DType, Device};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BufAsSliceError {
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

	pub fn buf_as_slice<'guard, 'buf, T: HasDType>(
		buf: &BorrowGuard<'buf, DeviceBuffer>,
	) -> Result<&'guard [T], BufAsSliceError> {
		if buf.dtype() != T::dtype {
			cold_path();
			return Err(BufAsSliceError::InvalidDType);
		}
		debug_assert!(T::dtype.bytes() == std::mem::size_of::<T>());
		if !buf.vmt().device_is_cpu {
			cold_path();
			return Err(BufAsSliceError::NotOnCPUDevice);
		}
		let data = buf.device_data();
		let elems = buf.elems();
		let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), elems) };
		Ok(slice)
	}

	unsafe fn drop_buffer(this: &DeviceBufferVMT, elems: usize, device_data: NonNull<u8>) {
		unsafe {
			let dtype = this.dtype;
			let layout = std::alloc::Layout::from_size_align(
				dtype.array_bytes_unchecked(elems),
				dtype.align(),
			)
			.unwrap_unchecked();
			std::alloc::dealloc(device_data.as_ptr(), layout);

			// Recreate the `Rc` that we forgot in `new_buffer()`
			let rc_device: Rc<dyn Device> = Rc::from_raw(this.device.as_ptr());
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

		if let Some(size) = dtype.array_bytes(elems)
			&& let Ok(layout) = std::alloc::Layout::from_size_align(size, dtype.align())
			&& let Some(memory) = NonNull::new(unsafe { std::alloc::alloc(layout) })
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

//--------------------------------------------------------------------------------------------------
