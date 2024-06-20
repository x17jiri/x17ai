// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::buffer::Buffer;
use crate::device::{DType, Device};
use crate::shape::{Shape, MAX_LOCAL_DIMS};
use smallvec::SmallVec;

#[repr(C)]
struct CPUBuffer {
	base: Buffer,
	data: *mut u8,
}

struct CPUDevice_f32;

impl CPUDevice_f32 {
	type UnaryKernel = fn(in_: f32) -> f32;

	fn inplace_unary_impl(
		i_ptr: *const f32,
		i_dim: &[Dim],
		o_prt: *mut f32,
		dim: &[Dim],
		kernel: UnaryKernel,
	) {
		if dim.is_empty() {
			return;
		}

		if dim.len() == 1 {
			let ptr = self.data.as_mut_ptr() as *mut f32;
			let ptr = unsafe { ptr.add(off) };
			for i in 0..dim[0].__len {
				unsafe {
					ptr.add(i * dim[0].__stride as usize)
						.write(kernel(ptr.add(i * dim[0].__stride as usize).read()));
				}
			}
		}
	}

	pub fn unary_(buf: *mut Buffer, shape: &Shape, kernel: Self::UnaryKernel) {
		let buf = unsafe { &mut *(buf as *mut CPUBuffer) };
		let data = buf.data as *mut f32;

		let shape = shape.merge_dims();
		let dims = shape.dims();
	}

	fn zero_(self_: &Device, buf: *mut Buffer, shape: &Shape) {
		Self::unary_(buf, shape, |_| 0.0);
	}
}

fn unary_impl(&mut self, mut off: usize, dim: &[Dim], shift: usize, kernel: UnaryCPUKernel) {
	if extra.is_empty() {
		let ptr = self.data.as_mut_ptr() as *mut u8;
		let ptr = unsafe { begin.add(off << shift) };

		let ptr = ptr as *mut Int;
		for x in 0..dim[0].__len {
			for y in 0..dim[1].__len {
				for z in 0..dim[2].__len {
					let off = x * dim[0].__stride as usize
						+ y * dim[1].__stride as usize
						+ z * dim[2].__stride as usize;
					unsafe {
						let ptr = ptr.add(off);
						ptr.write(Int::zero());
					}
				}
			}
		}
	} else {
		let extra0 = extra[0];
		let extra = &extra[1..];
		for j in 0..extra0.__len {
			self.unary_impl(off, extra, dim3, shift, f);
			off += extra0.__stride as usize;
		}
	}
}

pub fn unary_(&mut self, mut shape: Shape, kernel: UnaryCPUKernel) {
	shape.merge_dims();
	self.unary_impl(shape.__off, shape.dims(), shape.__dtype.shift(), kernel);
}

fn zero_(&mut self, shape: Shape) {
	self.unary_(shape, |ptr, dim3| unsafe {
		(*self.ops).zero_(ptr, dim3);
	});
}
