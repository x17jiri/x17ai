// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::shape::{Dim, Shape, MAX_LOCAL_DIMS};
use crate::tensor::{Buffer, DType, Device};
use smallvec::SmallVec;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::intrinsics::{likely, unlikely};
use std::mem::MaybeUninit;
use std::rc::Rc;

mod impl_f32 {
	use super::*;

	//----------------------------------------------------------------------------------------------
	// CPUDeviceF32

	struct CPUDeviceF32 {
		name: String,
	}

	impl CPUDeviceF32 {
		pub fn new(name: &str) -> Rc<dyn Device> {
			Rc::<CPUDeviceF32>::new(CPUDeviceF32 {
				name: name.to_string(),
			})
		}
	}

	impl Device for CPUDeviceF32 {
		fn name(&self) -> &str {
			&self.name
		}

		fn dtype(&self) -> (DType, usize) {
			(DType::Float, 32)
		}

		fn new_buffer(self: Rc<dyn Device>, elems: usize) -> Rc<dyn Buffer> {
			Rc::<CPUBufferF32>::new(CPUBufferF32 {
				dev: self.clone(),
				data: vec![0.0; elems].into_boxed_slice(),
			})
		}
	}

	//----------------------------------------------------------------------------------------------
	// CPUBufferF32

	struct CPUBufferF32 {
		dev: Rc<CPUDeviceF32>,
		data: Box<[f32]>,
	}

	impl Buffer for CPUBufferF32 {
		fn device(&self) -> Rc<dyn Device> {
			self.dev.clone()
		}
		fn dtype(&self) -> (DType, usize) {
			(DType::Float, 32)
		}
		fn zero_(&self, shape: &Shape) {
			let mut buf = self.data.as_mut_ptr();
			nullary_(buf, shape, || 0);
		}
	}

	//----------------------------------------------------------------------------------------------
	// nullary operations

	type NullaryKernel = fn() -> f32;

	fn nullary_impl_(buf: *mut f32, off: usize, dim: &[Dim], kernel: NullaryKernel) {
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

	fn nullary_(buf: *mut f32, shape: &Shape, kernel: NullaryKernel) {
		let dims = shape.merge_dims();
		if unlikely(dims.is_empty()) {
			return;
		}
		nullary_impl_(buf, shape.off(), dims.as_slice(), kernel);
	}
	/*
		//----------------------------------------------------------------------------------------------
		// unary operations

		type UnaryKernel = fn(in_: f32) -> f32;

		fn unary(buf: *mut Buffer, shape: &Shape, kernel: UnaryKernel) -> BufferPtr {
			let (out_shape, elems, input_contiguous) = shape.to_contiguous();
			let output = new_buffer(elems);
			if input_contiguous {
				// TODO
			} else {
				let dims = Shape::merge_dims(&[&shape, &out_shape]);
			}
		}

		fn exp(_self: &Device, buf: *mut Buffer, shape: &Shape) -> BufferPtr {
			unary(buf, shape, |x| x.exp())
		}
	*/
}

//--------------------------------------------------------------------------------------------------
