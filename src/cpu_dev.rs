// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::buffer::Buffer;
use crate::device::{BufferPtr, DType, Device, VMT};
use crate::shape::{Dim, Shape, MAX_LOCAL_DIMS};
use smallvec::SmallVec;
use std::intrinsics::{likely, unlikely};
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::cell::UnsafeCell;

type BufferItem = UnsafeCell<MaybeUninit<u64>>;

#[repr(C)]
struct CPUBuffer {
	base: Buffer,
	elems: usize,
}

fn drop_buffer(buf: *mut Buffer) {
	unsafe {
		let buf = &mut *(buf as *mut CPUBuffer);
		let layout = std::alloc::Layout::new::<CPUBuffer>();
		let data = std::alloc::Layout::array::<BufferItem>(buf.elems).unwrap_unchecked();
		let (layout, _) = layout.extend(data).unwrap_unchecked();
		std::alloc::dealloc(buf as *mut CPUBuffer as *mut u8, layout);
	}
}

pub fn new_cpu_device(name: String) -> Rc<Device> {
	Rc::new(Device { drop_buffer, name })
}

#[allow(non_camel_case_types)]
struct CPU_VMT_f32 {
	base: VMT,

	// How much do I need to L-shift an element offset to get a byte offset
	shift: usize,
}

// Note: Functions that end with underscore work in-place, modifying the input buffer.

impl CPU_VMT_f32 {
	type NullaryKernel = fn() -> f32;
	type UnaryKernel = fn(in_: f32) -> f32;

	//--------------

	pub fn new_vmt(dev: *const Device) -> TODO {
		let vmt = CPU_VMT_f32 {
			base: VMT {
				dtype: DType::Float,
				type_bits: 32,
				dev,
				zero_: Self::zero_,
			},
			shift: 2,
		};
		&vmt.base as *const VMT
	}

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
