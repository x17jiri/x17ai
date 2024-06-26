// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::device::Device;
use std::cell::RefCell;

pub struct Buffer {
	rc_minus_one: isize,
	dev: std::rc::Rc<Device>
}

#[derive(Debug)]
pub struct BufferPtr {
	ptr: *mut Buffer,
}

impl Clone for BufferPtr {
	fn clone(&self) -> Self {
		unsafe { (*self.ptr).rc_minus_one += 1 };
		BufferPtr { ptr: self.ptr }
	}
}

impl Drop for BufferPtr {
	fn drop(&mut self) {
		unsafe {
			let rc = (*self.ptr).rc_minus_one;
			(*self.ptr).rc_minus_one = rc - 1;
			if rc == 0 {
				let drop_buffer = (*self.ptr).dev.drop_buffer;
				drop_buffer(self.ptr);
			}
		}
	}
}
