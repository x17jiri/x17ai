//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

use crate::tensor::device::DevicePtr;
use crate::util::intrusive_rc::{self, IntrusiveRcTrait};
use crate::util::intrusive_ref_cell::{BorrowCounter, IntrusiveRefCellTrait};

use super::Device;

//--------------------------------------------------------------------------------------------------

pub struct DeviceBuffer {
	refcount: intrusive_rc::RefCount,
	borrow_counter: BorrowCounter,
	device_ptr: DevicePtr,
	bytes: usize,

	// TODO
	// - could this be replaced with some sort of thin rc that stores metadata in the pointee
	//   not in the pointer itself? This would save 8 bytes per buffer.
	device: Rc<dyn Device>,
}

impl IntrusiveRcTrait for DeviceBuffer {
	unsafe fn refcount(&self) -> &intrusive_rc::RefCount {
		&self.refcount
	}

	#[allow(clippy::drop_non_drop)]
	unsafe fn destroy(this: std::ptr::NonNull<Self>) {
		unsafe {
			let Self {
				refcount,
				borrow_counter,
				device_ptr,
				bytes,
				device,
			} = this.read();
			std::mem::drop(refcount);
			std::mem::drop(borrow_counter);
			device.drop_buffer(device_ptr, bytes, this);
		}
	}
}

impl IntrusiveRefCellTrait for DeviceBuffer {
	fn borrow_counter(&self) -> &BorrowCounter {
		&self.borrow_counter
	}
}

impl DeviceBuffer {
	#[inline]
	pub unsafe fn new(device_ptr: DevicePtr, bytes: usize, device: Rc<dyn Device>) -> Self {
		Self {
			refcount: intrusive_rc::RefCount::new(),
			borrow_counter: BorrowCounter::new(),
			device_ptr,
			bytes,
			device,
		}
	}

	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let device = device as *const dyn Device as *const ();
		let my_device = self.device() as *const dyn Device as *const ();
		my_device == device
	}

	#[inline]
	pub fn device_ptr(&self) -> DevicePtr {
		self.device_ptr
	}

	#[inline]
	pub fn byte_len(&self) -> usize {
		self.bytes
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

//--------------------------------------------------------------------------------------------------
