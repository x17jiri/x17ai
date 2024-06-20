// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::device::Device;

use std::alloc::Layout;

pub struct Buffer {
	rc_minus_one: usize,
	layout: Layout,
}

pub struct BufferPtr {
	ptr: *mut Buffer,
	dev: *const Device,
}

impl Clone for BufferPtr {
	fn clone(&self) -> Self {
		unsafe { (*self.ptr).rc_minus_one += 1 };
		BufferPtr {
			ptr: self.ptr,
			dev: self.dev,
		}
	}
}

impl Drop for BufferPtr {
	fn drop(&mut self) {
		unsafe {
			if (*self.ptr).rc_minus_one == 0 {
				(*self.dev).free(self.ptr);
			} else {
				(*self.ptr).rc_minus_one -= 1;
			}
		}
	}
}
