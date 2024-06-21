// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::buffer::Buffer;
use crate::device::{BufferPtr, DType, Device};
use crate::shape::{Dim, Shape, MAX_LOCAL_DIMS};
use smallvec::SmallVec;
use std::intrinsics::{likely, unlikely};

#[repr(C)]
struct CPUBuffer {
	base: Buffer,
	data: *mut u8,
}

struct CPUDevice_f32;

// Note: Functions that end with underscore work in-place, modifying the input buffer.

impl CPUDevice_f32 {
	type NullaryKernel = fn() -> f32;
	type UnaryKernel = fn(in_: f32) -> f32;

	//--------------

	fn new_buffer(elems: usize) -> BufferPtr {
		// TODO
	}

	//--------------

	fn nullary_impl_(buf: *mut f32, off: usize, dim: &[Dim], kernel: Self::NullaryKernel) {
		if dim.len() == 1 {
			let mut ptr = unsafe { buf.add(off) };
			for i in 0..dim[0].len {
				unsafe {
					ptr.write(kernel());
					ptr = ptr.wrapping_add(dim[0].stride as usize);
				}
			}
		}
	}

	fn nullary_(buf: *mut Buffer, shape: &Shape, kernel: Self::NullaryKernel) {
		let buf = unsafe { &mut *(buf as *mut CPUBuffer) };
		let data = buf.data as *mut f32;

		let dims = shape.merge_dims();
		if unlikely(dims.is_empty()) {
			return;
		}
		Self::nullary_impl_(data, shape.off(), dims.as_slice(), kernel);
	}

	fn zero_(_self: &Device, buf: *mut Buffer, shape: &Shape) {
		Self::nullary_(buf, shape, || 0);
	}

	//--------------

	fn unary(buf: *mut Buffer, shape: &Shape, kernel: Self::UnaryKernel) -> BufferPtr {
		let (out_shape, elems, input_contiguous) = shape.to_contiguous();
		let output = Self::new_buffer(elems);
		if input_contiguous {
			// TODO
		} else {
			let dims = Shape::merge_dims(&[&shape, &out_shape]);
		}
	}

	fn exp(_self: &Device, buf: *mut Buffer, shape: &Shape) -> BufferPtr {
		Self::unary(buf, shape, |x| x.exp())
	}

	//--------------
}
