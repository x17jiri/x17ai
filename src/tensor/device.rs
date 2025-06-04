//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

pub mod buffer;
// pub mod cpu; TODO
pub mod dtype;
pub mod executor;

pub use buffer::DeviceBuffer;
pub use dtype::DType;

pub trait Device {
	fn name(&self) -> &str;

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: usize) -> Rc<DeviceBuffer>;

	fn drop_buffer(self: Rc<Self>, dtype: DType, elems: usize, device_data: *mut u8);
}
