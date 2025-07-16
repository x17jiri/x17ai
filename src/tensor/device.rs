//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

pub mod buffer;
pub mod cpu;
pub mod dtype;
pub mod executor;
pub mod kernel_registry;
pub mod kernel_builder;

pub use buffer::DeviceBuffer;
pub use dtype::{DType, HasDType};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NewDeviceBufferError {
	UnsupportedDType,
	AllocationFailed,
}

pub trait Device {
	fn name(&self) -> &str;

	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<DeviceBuffer>, NewDeviceBufferError>;

	/// # Safety
	/// This function should only be called from `DeviceBuffer::drop()`.
	///
	/// The parameters must come from a valid `DeviceBuffer` instance.
	unsafe fn drop_buffer(self: Rc<Self>, dtype: DType, elems: usize, device_data: *mut u8);
}
