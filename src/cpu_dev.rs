// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::rand::Rng;
use crate::shape::{prep_op, Shape, TraversalDim};
use crate::tensor::{Buffer, DType, Device, Tensor};
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

	fn new_uninit(self: Rc<Self>, dtype: DType, elems: usize) -> Option<Rc<dyn Buffer>> {
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

	impl Buffer for CPUBufferF32 {
		fn device(&self) -> Rc<dyn Device> {
			self.dev.clone()
		}

		fn dtype(&self) -> DType {
			DType::Float(32)
		}

		//-- nullary operations working on the entire buffer

		fn zero_all_(&self) {
			let mut kernel = ZeroKernel;
			kernel.run(self.data.as_ptr(), self.data.len(), 1);
		}

		fn randn_all_(&self, rng: &mut Rng) {
			let mut kernel = RandKernel { rng };
			kernel.run(self.data.as_ptr(), self.data.len(), 1);
		}

		//-- unary operations

		fn zero_(&self, shape: &Shape) {
			let mut kernel = ZeroKernel;
			self.nullary_(shape, &mut kernel);
		}

		fn randn_(&self, shape: &Shape, rng: &mut Rng) {
			let mut kernel = RandKernel { rng };
			self.nullary_(shape, &mut kernel);
		}

		//-- unary operations

		fn exp(&self, shape: &Shape) -> Tensor {
			let mut kernel = ExpKernel;
			self.unary(shape, &mut kernel)
		}

		//-- reduction operations

		fn sum(&self, shape: &Shape) -> Tensor {
			let mut kernel = SumKernel;
			self.reduction(shape, &mut kernel)
		}

		fn max(&self, shape: &Shape) -> Tensor {
			let mut kernel = MaxKernel;
			self.reduction(shape, &mut kernel)
		}

		//-- formatting

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
			Some(result)
		}

		pub fn data(&self) -> *const Cell<f32> {
			self.data.as_ptr()
		}

		//-- nullary operations

		fn nullary_(&self, shape: &Shape, kernel: &mut dyn NullaryKernel) {
			let (t, _) = prep_op([shape]).unwrap();
			self.nullary_impl_(self.data(), t.in_off[0], t.dims.as_slice(), kernel);
		}

		fn nullary_impl_(
			&self,
			buf: *const Cell<f32>,
			off: isize,
			dims: &[TraversalDim<1>],
			kernel: &mut dyn NullaryKernel,
		) {
			if dims.len() == 1 {
				let dim = dims[0];
				let buf = unsafe { buf.offset(off) };
				kernel.run(buf, dim.len, dim.in_strides[0]);
			} else if likely(dims.len() > 1) {
				let mut off = off;
				let dim = dims.last().unwrap();
				let dims = &dims[..dims.len() - 1];
				for _ in 0..dim.len {
					self.nullary_impl_(buf, off, &dims, kernel);
					off += dim.in_strides[0];
				}
			}
		}

		//-- unary operations

		fn unary(&self, shape: &Shape, kernel: &mut dyn UnaryKernel) -> Tensor {
			let (t, out_shape) = prep_op([shape]).unwrap();
			let out_buf = Self::new(self.dev.clone(), t.elems).unwrap();
			self.unary_impl(
				self.data(),
				t.in_off[0],
				out_buf.data(),
				t.out_off,
				t.dims.as_slice(),
				kernel,
			);
			Tensor {
				buf: out_buf,
				shape: out_shape,
			}
		}

		fn unary_impl(
			&self,
			in_buf: *const Cell<f32>,
			in_off: isize,
			out_buf: *const Cell<f32>,
			out_off: isize,
			dims: &[TraversalDim<1>],
			kernel: &mut dyn UnaryKernel,
		) {
			if dims.len() == 1 {
				let dim = dims[0];
				let in_buf = unsafe { in_buf.offset(in_off) };
				let out_buf = unsafe { out_buf.offset(out_off) };
				kernel.run(in_buf, out_buf, dim.len, dim.in_strides[0], dim.out_stride);
			} else {
				let mut in_off = in_off;
				let mut out_off = out_off;
				let dim = dims.last().unwrap();
				let dims = &dims[..dims.len() - 1];
				for _ in 0..dim.len {
					self.unary_impl(in_buf, in_off, out_buf, out_off, dims, kernel);
					in_off += dim.in_strides[0];
					out_off += dim.out_stride;
				}
			}
		}

		//-- reduction operations

		fn reduction(&self, shape: &Shape, kernel: &mut dyn ReductionKernel) -> Tensor {
			let (t, _) = prep_op([shape]).unwrap();
			let (out_shape, _) = Shape::new(&[]).unwrap();
			let out_buf = Self::new(self.dev.clone(), 1).unwrap();
			let sum = self.reduction_impl(self.data(), t.in_off[0], t.dims.as_slice(), kernel);
			unsafe { (*out_buf.data()).set(sum) };
			Tensor {
				buf: out_buf,
				shape: out_shape,
			}
		}

		fn reduction_impl(
			&self,
			buf: *const Cell<f32>,
			off: isize,
			dims: &[TraversalDim<1>],
			kernel: &mut dyn ReductionKernel,
		) -> f32 {
			if dims.len() == 1 {
				let dim = dims[0];
				let buf = unsafe { buf.offset(off) };
				kernel.run(buf, dim.len, dim.in_strides[0])
			} else {
				let mut off = off;
				let dim = dims.last().unwrap();
				let dims = &dims[..dims.len() - 1];
				let mut sum = 0.0;
				for _ in 0..dim.len {
					sum += self.reduction_impl(buf, off, dims, kernel);
					off += dim.in_strides[0];
				}
				sum
			}
		}
	}

	trait NullaryKernel {
		fn run(&mut self, buf: *const Cell<f32>, len: usize, stride: isize);
	}

	struct ZeroKernel;

	impl NullaryKernel for ZeroKernel {
		fn run(&mut self, buf: *const Cell<f32>, len: usize, stride: isize) {
			let mut ptr = buf;
			for _ in 0..len {
				unsafe {
					(*ptr).set(f32::default());
					ptr = ptr.offset(stride);
				}
			}
		}
	}

	struct RandKernel<'a> {
		rng: &'a mut Rng,
	}

	impl<'a> NullaryKernel for RandKernel<'a> {
		fn run(&mut self, buf: *const Cell<f32>, len: usize, stride: isize) {
			let mut ptr = buf;
			for _ in 0..len {
				unsafe {
					(*ptr).set(self.rng.get_normal() as f32);
					ptr = ptr.offset(stride);
				}
			}
		}
	}

	trait UnaryKernel {
		fn run(
			&mut self,
			in_: *const Cell<f32>,
			out: *const Cell<f32>,
			len: usize,
			in_stride: isize,
			out_stride: isize,
		);
	}

	struct ExpKernel;

	impl UnaryKernel for ExpKernel {
		fn run(
			&mut self,
			in_: *const Cell<f32>,
			out: *const Cell<f32>,
			len: usize,
			in_stride: isize,
			out_stride: isize,
		) {
			let mut in_ptr = in_;
			let mut out_ptr = out;
			for _ in 0..len {
				unsafe {
					(*out_ptr).set((*in_ptr).get().exp());
					in_ptr = in_ptr.offset(in_stride);
					out_ptr = out_ptr.offset(out_stride);
				}
			}
		}
	}

	trait ReductionKernel {
		fn run(&mut self, buf: *const Cell<f32>, len: usize, stride: isize) -> f32;
	}

	struct SumKernel;

	impl ReductionKernel for SumKernel {
		fn run(&mut self, buf: *const Cell<f32>, len: usize, stride: isize) -> f32 {
			let mut ptr = buf;
			let mut sum = 0.0;
			for _ in 0..len {
				unsafe {
					sum += (*ptr).get();
					ptr = ptr.offset(stride);
				}
			}
			sum
		}
	}

	struct MaxKernel;

	impl ReductionKernel for MaxKernel {
		fn run(&mut self, buf: *const Cell<f32>, len: usize, stride: isize) -> f32 {
			let mut ptr = buf;
			let mut max = f32::NEG_INFINITY;
			for _ in 0..len {
				unsafe {
					let val = (*ptr).get();
					if val > max {
						max = val;
					}
					ptr = ptr.offset(stride);
				}
			}
			max
		}
	}
}

//--------------------------------------------------------------------------------------------------
