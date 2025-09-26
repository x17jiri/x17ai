//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::device::DeviceBase;
use crate::tensor::generic::buffer::Buffer;
use crate::util::mycell;

use super::Device;
use super::dtype::DType;

//--------------------------------------------------------------------------------------------------

pub struct DeviceBuffer {
	memory: NonNull<u8>,
	dtype: DType,
	elems: usize,
	device: NonNull<DeviceBase>,
}

impl DeviceBuffer {
	#[inline]
	pub unsafe fn new(
		memory: NonNull<u8>,
		dtype: DType,
		elems: usize,
		rc_device: Rc<dyn Device>,
	) -> Self {
		let device = Rc::into_raw(rc_device); // we will recreate the `Rc` in `drop()`
		let device = unsafe { NonNull::new_unchecked(device as *mut DeviceBase) };
		Self { memory, dtype, elems, device }
	}

	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let device = device as *const dyn Device as *const DeviceBase;
		let my_device = self.device_base() as *const DeviceBase;

		my_device == device
	}

	#[inline]
	pub fn memory(&self) -> NonNull<u8> {
		self.memory
	}

	#[inline]
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	#[inline]
	pub fn elems(&self) -> usize {
		self.elems
	}

	#[inline]
	pub fn device_base(&self) -> &DeviceBase {
		unsafe { self.device.as_ref() }
	}

	#[inline]
	pub fn device(&self) -> &dyn Device {
		unsafe { self.device_base().device() }
	}
}

impl Drop for DeviceBuffer {
	fn drop(&mut self) {
		unsafe {
			let device_base = self.device.as_ref();
			let device = device_base.device();
			device.drop_buffer(self.memory, self.dtype, self.elems);
			let rc_device = Rc::from_raw(self.device.as_ptr());
			std::mem::drop(rc_device);
		}
	}
}

//--------------------------------------------------------------------------------------------------

impl Buffer for Rc<mycell::RefCell<DeviceBuffer>> {
	fn len(&self) -> usize {
		self.elems
	}
}

impl Buffer for &mycell::RefCell<DeviceBuffer> {
	fn len(&self) -> usize {
		self.elems
	}
}

impl<'a> Buffer for mycell::BorrowGuard<'a, DeviceBuffer> {
	fn len(&self) -> usize {
		self.elems
	}
}

impl<'a> Buffer for mycell::BorrowMutGuard<'a, DeviceBuffer> {
	fn len(&self) -> usize {
		self.elems
	}
}

//--------------------------------------------------------------------------------------------------
