// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::rand::Rng;
use crate::shape::{MatMul, Traversal, TraversalDim};
use crate::tensor::{Buffer, DType, Device};
use std::any::*;
use std::cell::Cell;
use std::fmt;
use std::intrinsics::{likely, unlikely};
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

	unsafe fn new_uninit(self: Rc<Self>, dtype: DType, elems: usize) -> Option<Rc<dyn Buffer>> {
		match dtype {
			DType::Float(32) => Some(impl_f32::CPUBufferF32::new_uninit(self, elems)?),
			_ => None,
		}
	}
}

mod impl_f32 {
	use super::*;

	struct KernelInput {
		ptr: *const Cell<f32>,
		stride: isize,
	}

	impl KernelInput {
		fn get(&mut self) -> f32 {
			unsafe {
				let val = (*self.ptr).get();
				self.ptr = self.ptr.offset(self.stride);
				val
			}
		}
	}

	struct KernelOutput {
		ptr: *const Cell<f32>,
		stride: isize,
	}

	impl KernelOutput {
		fn set(&mut self, val: f32) {
			unsafe {
				(*self.ptr).set(val);
				self.ptr = self.ptr.offset(self.stride);
			}
		}
	}

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
			self.nullary_all_(&mut ZeroKernel);
		}

		fn randn_all_(&self, rng: &mut Rng) {
			self.nullary_all_(&mut RandKernel { rng });
		}

		//-- nullary operations

		fn zero_(&self, traversal: Traversal<1>) {
			self.nullary_(traversal, &mut ZeroKernel);
		}

		fn randn_(&self, traversal: Traversal<1>, rng: &mut Rng) {
			self.nullary_(traversal, &mut RandKernel { rng });
		}

		//-- unary operations

		fn exp(&self, traversal: Traversal<1>) -> Rc<dyn Buffer> {
			self.unary(traversal, &mut ExpKernel)
		}

		//-- reduction operations

		fn sum(&self, traversal: Traversal<1>) -> Rc<dyn Buffer> {
			self.reduction(traversal, &mut SumKernel)
		}

		fn max(&self, traversal: Traversal<1>) -> Rc<dyn Buffer> {
			self.reduction(traversal, &mut MaxKernel)
		}

		//-- binary operations

		fn add(&self, other: &dyn Buffer, traversal: Traversal<2>) -> Rc<dyn Buffer> {
			self.binary(other, traversal, &mut AddKernel)
		}

		//-- special

		fn matmul(&self, _other: &dyn Buffer, data: MatMul) -> Rc<dyn Buffer> {
			unimplemented!();
		}

		//-- formatting

		fn format(
			&self,
			f: &mut fmt::Formatter,
			off: isize,
			len: usize,
			stride: isize,
		) -> std::fmt::Result {
			let mut input = KernelInput { ptr: self.ptr(off), stride };
			for i in 0..len {
				if i > 0 {
					write!(f, ", ")?;
				}
				write!(f, "{}", input.get())?;
			}
			Ok(())
		}
	}

	impl CPUBufferF32 {
		pub fn new_uninit(dev: Rc<CPUDevice>, elems: usize) -> Option<Rc<CPUBufferF32>> {
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

		fn ptr(&self, off: isize) -> *const Cell<f32> {
			unsafe { self.data.as_ptr().offset(off) }
		}

		//-- nullary operations working on the entire buffer

		fn nullary_all_(&self, kernel: &mut dyn NullaryKernel) {
			kernel.run(
				self.data.len(),
				KernelOutput { ptr: self.ptr(0), stride: 1 },
			);
		}

		//-- nullary operations

		fn nullary_(&self, traversal: Traversal<1>, kernel: &mut dyn NullaryKernel) {
			self.nullary_impl_(self.ptr(traversal.in_off[0]), traversal.dims(), kernel);
		}

		fn nullary_impl_(
			&self,
			mut ptr: *const Cell<f32>,
			dims: &[TraversalDim<1>],
			kernel: &mut dyn NullaryKernel,
		) {
			if dims.len() == 1 {
				let dim = dims[0];
				kernel.run(dim.len, KernelOutput { ptr, stride: dim.in_strides[0] });
			} else if likely(dims.len() > 1) {
				let dim = dims.last().unwrap();
				let dims = &dims[..dims.len() - 1];
				for _ in 0..dim.len {
					self.nullary_impl_(ptr, &dims, kernel);
					unsafe {
						ptr = ptr.offset(dim.in_strides[0]);
					}
				}
			}
		}

		//-- unary operations

		fn unary(&self, traversal: Traversal<1>, kernel: &mut dyn UnaryKernel) -> Rc<dyn Buffer> {
			let out_buf = Self::new_uninit(self.dev.clone(), traversal.elems).unwrap();
			self.unary_impl(
				self.ptr(traversal.in_off[0]),
				out_buf.ptr(traversal.out_off),
				traversal.dims(),
				kernel,
			);
			out_buf
		}

		fn unary_impl(
			&self,
			mut a: *const Cell<f32>,
			mut o: *const Cell<f32>,
			dims: &[TraversalDim<1>],
			kernel: &mut dyn UnaryKernel,
		) {
			if dims.len() == 1 {
				let dim = dims[0];
				kernel.run(
					dim.len,
					KernelInput { ptr: a, stride: dim.in_strides[0] },
					KernelOutput { ptr: o, stride: dim.out_stride },
				);
			} else {
				let dim = dims.last().unwrap();
				let dims = &dims[..dims.len() - 1];
				for _ in 0..dim.len {
					self.unary_impl(a, o, dims, kernel);
					unsafe {
						a = a.offset(dim.in_strides[0]);
						o = o.offset(dim.out_stride);
					}
				}
			}
		}

		//-- reduction operations

		fn reduction(
			&self,
			traversal: Traversal<1>,
			kernel: &mut dyn ReductionKernel,
		) -> Rc<dyn Buffer> {
			let out_buf = Self::new_uninit(self.dev.clone(), 1).unwrap();
			let val = self.reduction_impl(self.ptr(traversal.in_off[0]), traversal.dims(), kernel);
			unsafe { (*out_buf.ptr(0)).set(val) };
			out_buf
		}

		fn reduction_impl(
			&self,
			mut a: *const Cell<f32>,
			dims: &[TraversalDim<1>],
			kernel: &mut dyn ReductionKernel,
		) -> f32 {
			let mut sum = 0.0;
			if dims.len() == 1 {
				let dim = dims[0];
				sum = kernel.run(dim.len, KernelInput { ptr: a, stride: dim.in_strides[0] });
			} else {
				let dim = dims.last().unwrap();
				let dims = &dims[..dims.len() - 1];
				for _ in 0..dim.len {
					sum += self.reduction_impl(a, dims, kernel);
					unsafe {
						a = a.offset(dim.in_strides[0]);
					}
				}
			}
			sum
		}

		//-- binary operations

		fn binary(
			&self,
			other: &dyn Buffer,
			traversal: Traversal<2>,
			kernel: &mut dyn BinaryKernel,
		) -> Rc<dyn Buffer> {
			debug_assert!(self.type_id() == other.type_id());
			let other = other as *const dyn Buffer;
			let other = other as *const CPUBufferF32;
			let other = unsafe { &*other };

			let out_buf = Self::new_uninit(self.dev.clone(), traversal.elems).unwrap();
			self.binary_impl(
				self.ptr(traversal.in_off[0]),
				other.ptr(traversal.in_off[1]),
				out_buf.ptr(traversal.out_off),
				traversal.dims(),
				kernel,
			);
			out_buf
		}

		fn binary_impl(
			&self,
			mut a: *const Cell<f32>,
			mut b: *const Cell<f32>,
			mut o: *const Cell<f32>,
			dims: &[TraversalDim<2>],
			kernel: &mut dyn BinaryKernel,
		) {
			if dims.len() == 1 {
				let dim = dims[0];
				kernel.run(
					dim.len,
					KernelInput { ptr: a, stride: dim.in_strides[0] },
					KernelInput { ptr: b, stride: dim.in_strides[1] },
					KernelOutput { ptr: o, stride: dim.out_stride },
				);
			} else {
				let dim = dims.last().unwrap();
				let dims = &dims[..dims.len() - 1];
				for _ in 0..dim.len {
					self.binary_impl(a, b, o, dims, kernel);
					unsafe {
						a = a.offset(dim.in_strides[0]);
						b = b.offset(dim.in_strides[1]);
						o = o.offset(dim.out_stride);
					}
				}
			}
		}
	}

	trait NullaryKernel {
		fn run(&mut self, len: usize, o: KernelOutput);
	}

	struct ZeroKernel;

	impl NullaryKernel for ZeroKernel {
		fn run(&mut self, len: usize, mut o: KernelOutput) {
			for _ in 0..len {
				o.set(Default::default());
			}
		}
	}

	struct RandKernel<'a> {
		rng: &'a mut Rng,
	}

	impl<'a> NullaryKernel for RandKernel<'a> {
		fn run(&mut self, len: usize, mut o: KernelOutput) {
			for _ in 0..len {
				o.set(self.rng.get_normal() as f32);
			}
		}
	}

	trait UnaryKernel {
		fn run(&mut self, len: usize, a: KernelInput, o: KernelOutput);
	}

	struct ExpKernel;

	impl UnaryKernel for ExpKernel {
		fn run(&mut self, len: usize, mut a: KernelInput, mut o: KernelOutput) {
			for _ in 0..len {
				o.set(a.get().exp());
			}
		}
	}

	trait ReductionKernel {
		fn run(&mut self, len: usize, a: KernelInput) -> f32;
	}

	struct SumKernel;

	impl ReductionKernel for SumKernel {
		fn run(&mut self, len: usize, mut a: KernelInput) -> f32 {
			let mut sum = Default::default();
			for _ in 0..len {
				sum += a.get();
			}
			sum
		}
	}

	struct MaxKernel;

	impl ReductionKernel for MaxKernel {
		fn run(&mut self, len: usize, mut a: KernelInput) -> f32 {
			let mut max = f32::NEG_INFINITY; // TODO - this will not work for integers
			for _ in 0..len {
				max = f32::max(max, a.get());
			}
			max
		}
	}

	trait BinaryKernel {
		fn run(&mut self, len: usize, a: KernelInput, b: KernelInput, o: KernelOutput);
	}

	struct AddKernel;

	impl BinaryKernel for AddKernel {
		fn run(&mut self, len: usize, mut a: KernelInput, mut b: KernelInput, mut o: KernelOutput) {
			for _ in 0..len {
				o.set(a.get() + b.get());
			}
		}
	}
}

//--------------------------------------------------------------------------------------------------
