//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::generic::buffer::Buffer;
use crate::util::mycell;

use super::Device;
use super::dtype::DType;

//--------------------------------------------------------------------------------------------------

pub struct DeviceBuffer {
	memory: NonNull<u8>,
	dtype: DType,
	elems: usize,
	device: Rc<dyn Device>, // TODO - use thin_rc
}

impl DeviceBuffer {
	#[inline]
	pub unsafe fn new(
		memory: NonNull<u8>,
		dtype: DType,
		elems: usize,
		device: Rc<dyn Device>,
	) -> Self {
		Self { memory, dtype, elems, device }
	}

	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let device = device as *const dyn Device as *const ();
		let my_device = self.device() as *const dyn Device as *const ();
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
	pub fn device(&self) -> &dyn Device {
		Rc::as_ref(&self.device)
	}

	#[inline]
	pub fn rc_device(&self) -> Rc<dyn Device> {
		self.device.clone()
	}
}

impl Drop for DeviceBuffer {
	fn drop(&mut self) {
		unsafe {
			self.device.drop_buffer(self.memory, self.dtype, self.elems);
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
