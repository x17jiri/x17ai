//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::generic::buffer::Buffer;

use super::Device;
use super::dtype::DType;
use super::executor::Executor;

//--------------------------------------------------------------------------------------------------

pub struct DeviceBuffer {
	pub executor: NonNull<dyn Executor>,
	pub dtype: DType,
	pub elems: usize,
	pub device_data: *mut u8,
	pub device: ManuallyDrop<Rc<dyn Device>>,
}

impl DeviceBuffer {
	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let my_dev = self.device.as_ref();
		let my_dev = my_dev as *const dyn Device;
		let my_dev = my_dev as *const u8;

		let dev = device as *const dyn Device;
		let dev = dev as *const u8;

		my_dev == dev
	}

	#[inline]
	pub fn executor(&self) -> &dyn Executor {
		// SAFETY: Executor will live as long as the device
		// and device will live as long as `self` keeps the `Rc` alive.
		unsafe { self.executor.as_ref() }
	}
}

impl Drop for DeviceBuffer {
	fn drop(&mut self) {
		unsafe { ManuallyDrop::take(&mut self.device) }.drop_buffer(
			self.dtype,
			self.elems,
			self.device_data,
		);
	}
}

impl Buffer for Rc<DeviceBuffer> {}
impl<'a> Buffer for &'a DeviceBuffer {}

//--------------------------------------------------------------------------------------------------
