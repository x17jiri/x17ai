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
	pub device_is_cpu: bool,
	pub device: ManuallyDrop<Rc<dyn Device>>,

	pub read_count: Cell<usize>,
	pub write_count: Cell<usize>,
}

impl DeviceBuffer {
	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let my_dev = self.device.as_ref();
		let my_dev = std::ptr::from_ref(my_dev);
		let my_dev = my_dev.cast::<u8>();

		let dev = std::ptr::from_ref(device);
		let dev = dev.cast::<u8>();

		my_dev == dev
	}

	#[inline]
	pub fn executor(&self) -> &dyn Executor {
		// SAFETY: Executor will live as long as the device
		// and device will live as long as `self` keeps the `Rc` alive.
		unsafe { self.executor.as_ref() }
	}

	pub fn try_borrow(&self) -> Result<DeviceBufferRef<'_>, BorrowError> {
		DeviceBufferRef::new(self)
	}

	pub fn try_borrow_mut(&self) -> Result<DeviceBufferRefMut<'_>, BorrowMutError> {
		DeviceBufferRefMut::new(self)
	}
}

impl Drop for DeviceBuffer {
	fn drop(&mut self) {
		unsafe {
			let dev = ManuallyDrop::take(&mut self.device);
			dev.drop_buffer(self.dtype, self.elems, self.device_data);
		}
	}
}

//--------------------------------------------------------------------------------------------------

impl Buffer for Rc<DeviceBuffer> {
	fn len(&self) -> usize {
		self.elems
	}
}

impl Buffer for &DeviceBuffer {
	fn len(&self) -> usize {
		self.elems
	}
}

impl<'a> Buffer for DeviceBufferRef<'a> {
	fn len(&self) -> usize {
		self.device_buffer.elems
	}
}

impl<'a> Buffer for DeviceBufferRefMut<'a> {
	fn len(&self) -> usize {
		self.device_buffer.elems
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BorrowError;

impl std::error::Error for BorrowError {}

impl std::fmt::Display for BorrowError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Cannot borrow the device buffer")
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BorrowMutError;

impl std::error::Error for BorrowMutError {}

impl std::fmt::Display for BorrowMutError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Cannot borrow the device buffer mutably")
	}
}

//--------------------------------------------------------------------------------------------------

pub struct DeviceBufferRef<'a> {
	device_buffer: &'a DeviceBuffer,
}

impl<'a> DeviceBufferRef<'a> {
	pub fn new(device_buffer: &'a DeviceBuffer) -> Result<Self, BorrowError> {
		let read_count = device_buffer.read_count.get();
		let write_count = device_buffer.write_count.get();

		if write_count != 0 {
			cold_path();
			return Err(BorrowError);
		}

		device_buffer.read_count.set(read_count + 1);
		Ok(Self { device_buffer })
	}

	pub unsafe fn new_unsafe(device_buffer: &'a DeviceBuffer, fail: &mut usize) -> Self {
		let read_count = device_buffer.read_count.get();
		let write_count = device_buffer.write_count.get();

		*fail |= write_count;

		device_buffer.read_count.set(read_count + 1);
		Self { device_buffer }
	}

	pub fn device_buffer(&self) -> &'a DeviceBuffer {
		self.device_buffer
	}
}

impl<'a> Clone for DeviceBufferRef<'a> {
	fn clone(&self) -> Self {
		let read_count = self.device_buffer.read_count.get();

		debug_assert!(read_count > 0, "DeviceBufferRef: invalid counter state");

		self.device_buffer.read_count.set(read_count + 1);
		Self { device_buffer: self.device_buffer }
	}
}

impl<'a> Drop for DeviceBufferRef<'a> {
	fn drop(&mut self) {
		let read_count = self.device_buffer.read_count.get();

		debug_assert!(read_count > 0, "DeviceBufferRef: invalid counter state");

		self.device_buffer.read_count.set(read_count - 1);
	}
}

impl<'a> std::ops::Deref for DeviceBufferRef<'a> {
	type Target = DeviceBuffer;

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.device_buffer
	}
}

impl<'a> From<DeviceBufferRefMut<'a>> for DeviceBufferRef<'a> {
	fn from(value: DeviceBufferRefMut<'a>) -> Self {
		let read_count = value.device_buffer.read_count.get();
		let write_count = value.device_buffer.write_count.get();

		debug_assert!(write_count > 0, "DeviceBufferRefMut: invalid counter state");

		value.device_buffer.read_count.set(read_count + 1);
		value.device_buffer.write_count.set(write_count - 1);
		let result = Self { device_buffer: value.device_buffer };

		std::mem::forget(value);
		result
	}
}

//--------------------------------------------------------------------------------------------------

pub struct DeviceBufferRefMut<'a> {
	device_buffer: &'a DeviceBuffer,
}

impl<'a> DeviceBufferRefMut<'a> {
	pub fn new(device_buffer: &'a DeviceBuffer) -> Result<Self, BorrowMutError> {
		let read_count = device_buffer.read_count.get();
		let write_count = device_buffer.write_count.get();

		if (read_count | write_count) != 0 {
			cold_path();
			return Err(BorrowMutError);
		}

		device_buffer.write_count.set(1);
		Ok(Self { device_buffer })
	}

	pub unsafe fn new_unsafe(device_buffer: &'a DeviceBuffer, fail: &mut usize) -> Self {
		let read_count = device_buffer.read_count.get();
		let write_count = device_buffer.write_count.get();

		*fail |= read_count | write_count;

		device_buffer.write_count.set(write_count + 1);
		Self { device_buffer }
	}

	pub fn device_buffer(&self) -> &'a DeviceBuffer {
		self.device_buffer
	}
}

impl<'a> Drop for DeviceBufferRefMut<'a> {
	fn drop(&mut self) {
		let write_count = self.device_buffer.write_count.get();

		debug_assert!(write_count > 0, "DeviceBufferRefMut: invalid counter state");

		self.device_buffer.write_count.set(write_count - 1);
	}
}

impl<'a> std::ops::Deref for DeviceBufferRefMut<'a> {
	type Target = DeviceBuffer;

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.device_buffer
	}
}

//--------------------------------------------------------------------------------------------------
