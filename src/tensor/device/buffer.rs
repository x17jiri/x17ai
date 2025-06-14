//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;
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

	/// 0 => not borrowed,
	/// >0 => number of immutable borrows,
	/// <0 => one mutable borrow (exclusive).
	pub borrow_count: Cell<isize>,
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

	pub fn try_borrow(&self) -> Result<DeviceBufferRef, BorrowError> {
		DeviceBufferRef::new(self)
	}

	pub fn try_borrow_mut(&self) -> Result<DeviceBufferRefMut, BorrowError> {
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

impl Buffer for Rc<DeviceBuffer> {}
impl<'a> Buffer for DeviceBufferRef<'a> {}
impl<'a> Buffer for DeviceBufferRefMut<'a> {}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowError {
	CannotBorrow,
	CannotBarrowMut,
}

impl std::error::Error for BorrowError {}

impl std::fmt::Display for BorrowError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			BorrowError::CannotBorrow => write!(f, "Cannot borrow the device buffer"),
			BorrowError::CannotBarrowMut => write!(f, "Cannot borrow the device buffer mutably"),
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct DeviceBufferRef<'a> {
	device_buffer: &'a DeviceBuffer,
}

impl<'a> DeviceBufferRef<'a> {
	pub fn new(device_buffer: &'a DeviceBuffer) -> Result<Self, BorrowError> {
		let count = device_buffer.borrow_count.get();
		debug_assert!(count < isize::MAX, "DeviceBufferRef borrow count overflow");

		let new_count = count.wrapping_add(1);
		if new_count > 0 {
			device_buffer.borrow_count.set(new_count);
			Ok(Self { device_buffer })
		} else {
			Err(BorrowError::CannotBorrow)
		}
	}
}

impl<'a> Clone for DeviceBufferRef<'a> {
	fn clone(&self) -> Self {
		let count = self.device_buffer.borrow_count.get();
		debug_assert!(count > 0, "DeviceBufferRef: invalid counter state");
		debug_assert!(count < isize::MAX, "DeviceBufferRef borrow count overflow");

		let new_count = count + 1;
		self.device_buffer.borrow_count.set(new_count);
		Self { device_buffer: self.device_buffer }
	}
}

impl<'a> Drop for DeviceBufferRef<'a> {
	fn drop(&mut self) {
		let count = self.device_buffer.borrow_count.get();
		debug_assert!(count > 0, "DeviceBufferRef: invalid counter state");

		let new_count = count - 1;
		self.device_buffer.borrow_count.set(new_count);
	}
}

impl<'a> std::ops::Deref for DeviceBufferRef<'a> {
	type Target = DeviceBuffer;

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.device_buffer
	}
}

//--------------------------------------------------------------------------------------------------

pub struct DeviceBufferRefMut<'a> {
	device_buffer: &'a DeviceBuffer,
}

impl<'a> DeviceBufferRefMut<'a> {
	pub fn new(device_buffer: &'a DeviceBuffer) -> Result<Self, BorrowError> {
		let count = device_buffer.borrow_count.get();
		if count == 0 {
			device_buffer.borrow_count.set(-1);
			Ok(Self { device_buffer })
		} else {
			Err(BorrowError::CannotBarrowMut)
		}
	}
}

impl<'a> Drop for DeviceBufferRefMut<'a> {
	fn drop(&mut self) {
		let count = self.device_buffer.borrow_count.get();
		debug_assert!(count == -1, "DeviceBufferRefMut: invalid counter state");

		self.device_buffer.borrow_count.set(0);
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
