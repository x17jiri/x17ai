// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::shape::{prep_op, Shape, TraversalDim};
use crate::tensor::{Buffer, DType, Device};
use std::cell::{Cell, RefCell, UnsafeCell};
use std::fmt;
use std::intrinsics::{likely, unlikely};
use std::mem::MaybeUninit;
use std::rc::Rc;

pub struct CPUDevice {
	name: String,
}

impl CPUDevice {
	pub fn new(name: String) -> Rc<CPUDevice> {
		Rc::<CPUDevice>::new(CPUDevice { name })
	}
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: usize) -> Option<Rc<dyn Buffer>> {
		match dtype {
			DType::Float(32) => Some(impl_f32::CPUBufferF32::new(self, elems)?),
			_ => None,
		}
	}
}

mod impl_f32 {
	use super::*;

	pub struct CPUBufferF32 {
		dev: Rc<CPUDevice>,
		data: Box<[Cell<f32>]>,
	}

	impl CPUBufferF32 {
		pub fn new(dev: Rc<CPUDevice>, elems: usize) -> Option<Rc<CPUBufferF32>> {
			let layout = std::alloc::Layout::array::<Cell<f32>>(elems).ok()?;
			let result = unsafe {
				let data = std::alloc::alloc(layout) as *mut Cell<f32>;
				if data.is_null() {
					return None;
				}
				let data = std::slice::from_raw_parts_mut(data, elems);
				let data = Box::from_raw(data);
				Rc::<CPUBufferF32>::new(CPUBufferF32 { dev, data })
			};
			//			/***************************************************************************
			for i in 0..elems {
				result.data[i].set(i as f32);
			}
			//			***************************************************************************/
			Some(result)
		}

		pub fn data(&self) -> *const Cell<f32> {
			self.data.as_ptr()
		}

		pub fn zero_kernel(buf: *const Cell<f32>, len: usize, stride: isize) {
			let mut ptr = buf;
			for _ in 0..len {
				unsafe {
					(*ptr).set(f32::default());
					ptr = ptr.offset(stride);
				}
			}
		}
	}

	impl Buffer for CPUBufferF32 {
		fn device(&self) -> Rc<dyn Device> {
			self.dev.clone()
		}

		fn dtype(&self) -> DType {
			DType::Float(32)
		}

		fn zero_(&self, shape: &Shape) {
			nullary_(self.data(), shape, CPUBufferF32::zero_kernel);
		}

		fn __format(
			&self,
			f: &mut fmt::Formatter,
			off: isize,
			len: usize,
			stride: isize,
		) -> std::fmt::Result {
			let mut ptr = unsafe { self.data().offset(off) };
			for i in 0..len {
				if i > 0 {
					write!(f, ", ")?;
				}
				let val = unsafe { (*ptr).get() };
				write!(f, "{}", val)?;

				ptr = unsafe { ptr.offset(stride) };
			}
			Ok(())
		}
	}

	//-- nullary operations

	type NullaryKernel = fn(buf: *const Cell<f32>, len: usize, stride: isize);

	fn nullary_impl_(
		buf: *const Cell<f32>,
		off: isize,
		dims: &[TraversalDim<1>],
		kernel: NullaryKernel,
	) {
		if dims.len() == 1 {
			let dim = dims[0];
			let buf = unsafe { buf.offset(off) };
			kernel(buf, dim.len, dim.in_strides[0]);
		} else if likely(dims.len() > 1) {
			let mut off = off;
			let dim = dims.last().unwrap();
			let dims = &dims[..dims.len() - 1];
			for _ in 0..dim.len {
				nullary_impl_(buf, off, &dims, kernel);
				off += dim.in_strides[0];
			}
		}
	}

	fn nullary_(buf: *const Cell<f32>, shape: &Shape, kernel: NullaryKernel) {
		let (t, _) = prep_op([shape]).unwrap();
		nullary_impl_(buf, t.in_off[0], t.dims.as_slice(), kernel);
	}

	/*
	//-- unary operations

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
