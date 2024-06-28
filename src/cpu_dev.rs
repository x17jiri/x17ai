// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::shape::{prep_op, LenAndStrides, Shape};
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
			DType::Float(32) => Some(impl_f32::CPUBufferF32::new(self, elems)),
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
		pub fn new(dev: Rc<CPUDevice>, elems: usize) -> Rc<CPUBufferF32> {
			Rc::<CPUBufferF32>::new(CPUBufferF32 {
				dev,
				data: vec![Cell::new(f32::default() + (17 as f32)); elems].into_boxed_slice(),
			})
		}

		pub fn data(&self) -> *const Cell<f32> {
			self.data.as_ptr()
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
			nullary_(self.data(), shape, || f32::default());
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

	type NullaryKernel = fn() -> f32;

	fn nullary_impl_(
		buf: *const Cell<f32>,
		off: isize,
		dim: &[LenAndStrides<1>],
		kernel: NullaryKernel,
	) {
		if dim.len() == 1 {
			unsafe {
				let mut ptr = buf.offset(off);
				for _ in 0..dim[0].len {
					let cell = &*ptr;
					cell.set(kernel());

					ptr = ptr.offset(dim[0].strides[0]);
				}
			}
		} else if likely(dim.len() > 1) {
			let mut off = off;
			for _ in 0..dim[0].len {
				nullary_impl_(buf, off, &dim[1..], kernel);
				off += dim[0].strides[0];
			}
		}
	}

	fn nullary_(buf: *const Cell<f32>, shape: &Shape, kernel: NullaryKernel) {
		let t = prep_op([shape]).unwrap();
		nullary_impl_(buf, t.off[0], t.dims.as_slice(), kernel);
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
