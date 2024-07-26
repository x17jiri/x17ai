// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::boxed::Box;
use std::cell::{Cell, RefCell};
use std::fmt;

pub struct CPUDevice {
	name: String,
	rng: RefCell<Rng>,
}

impl CPUDevice {
	pub fn new(name: String) -> Rc<Self> {
		Rc::new(Self {
			name,
			rng: RefCell::new(Rng::new_default()),
		})
	}
}

type CPUBufferElement = u64;

#[repr(C)] // make sure the addr of base == the addr of the struct
pub struct CPUBuffer {
	base: BufferBase,
	memory: Box<[Cell<CPUBufferElement>]>,
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn new_buffer(self: Rc<Self>, size_bytes: usize) -> Rc<dyn Buffer> {
		let elem_size = std::mem::size_of::<CPUBufferElement>();
		let elems = (size_bytes + elem_size - 1) / elem_size;
		Rc::new(CPUBuffer {
			base: BufferBase {
				device: self.clone(),
				capacity: size_bytes,
			},
			memory: vec![Cell::new(0); elems].into_boxed_slice(),
		})
	}
}

impl CPUBuffer {
	fn device(&self) -> &CPUDevice {
		let dev = self.base.device.as_ref();
		let dev = dev as *const dyn Device;
		let dev = dev as *const CPUDevice;
		unsafe { &*dev }
	}

	fn is_on_my_device(&self, tensor: &Tensor) -> bool {
		is_buf_owned_by_device(tensor.buffer.as_ref(), self.base.device.as_ref())
	}

	fn get_buffer_of(&self, t: &Tensor) -> &CPUBuffer {
		debug_assert!(self.is_on_my_device(t));
		let buf = t.buffer.as_ref();
		let buf = buf as *const dyn Buffer;
		let buf = buf as *const CPUBuffer;
		unsafe { &*buf }
	}

	fn cast<T>(&self, offset: usize, elems: usize) -> &[Cell<T>] {
		let ptr = self.memory.as_ptr();
		let ptr = ptr as *const Cell<T>;
		let ptr = ptr.wrapping_add(offset);
		unsafe { std::slice::from_raw_parts(ptr, elems) }
	}

	fn zeros_f32_(&self, offset: usize, elems: usize) {
		let data = self.cast::<f32>(offset, elems);
		for val in data {
			val.set(0.0);
		}
	}

	fn randn_f32_(&self, offset: usize, elems: usize) {
		let data = self.cast::<f32>(offset, elems);
		let mut rng = self.device().rng.borrow_mut();
		for val in data {
			val.set(rng.get_normal() as f32);
		}
	}

	fn rms_norm_dtype(
		&self,
		dtype: DType,
		offset: usize,
		a: &Self,
		a_offset: usize,
		dim_size: usize,
		eps: f64,
		batch: &[BatchDim<1>],
	) {
		if !batch.is_empty() {
			let batch_dim = batch[0];
			let batch = &batch[1..];
			for i in 0..batch_dim.size {
				self.rms_norm_dtype(
					dtype,
					offset + i * batch_dim.out_stride,
					a,
					a_offset + i * batch_dim.in_strides[0],
					dim_size,
					eps,
					batch,
				);
			}
			return;
		}

		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.rms_norm_f32(offset, a, a_offset, dim_size, eps)
			},
			_ => todo!(),
		}
	}

	fn rms_norm_f32(&self, offset: usize, a: &Self, a_offset: usize, dim_size: usize, eps: f64) {
		let out_vec = self.cast::<f32>(offset, dim_size);
		let in_vec = a.cast::<f32>(a_offset, dim_size);

		let mut sqr_sum = eps;
		for i in in_vec {
			let i = i.get() as f64;
			sqr_sum += i * i;
		}

		let scale = ((dim_size as f64) / sqr_sum).sqrt();

		for (o, i) in out_vec.iter().zip(in_vec) {
			let i = i.get() as f64;
			o.set((i * scale) as f32);
		}
	}
	/*
		fn mm_f32(&self, a: &Tensor, b: &Tensor, c: &Tensor, prep: PrepMM) {
			if prep.a_transpose || prep.b_transpose {
				todo!("TODO");
			}

			let a = self.traversal::<f32>(a.byte_offset, prep.batch_size, prep.a_rows * prep.a_cols);
			let b = self.traversal::<f32>(b.byte_offset, prep.batch_size, prep.a_cols * prep.b_cols);
			let c = self.traversal::<f32>(c.byte_offset, prep.batch_size, prep.a_rows * prep.b_cols);

			for ((a, b), c) in a.zip(b).zip(c) {
				for i in 0..prep.a_rows {
					for j in 0..prep.b_cols {
						let mut sum = 0.0;
						for k in 0..prep.a_cols {
							sum += a[i * prep.a_cols + k].get() * b[k * prep.b_cols + j].get();
						}
						c[i * prep.b_cols + j].set(sum);
					}
				}
			}
		}
	*/
	fn format_f32(
		&self,
		offset: usize,
		f: &mut fmt::Formatter,
		count: usize,
		stride: usize,
	) -> fmt::Result {
		for i in 0..count {
			if i != 0 {
				write!(f, ", ")?;
			}
			let val = self.cast::<f32>(offset + i * stride, 1)[0].get();
			write!(f, "{:.6}", val)?;
		}
		Ok(())
	}
}

impl Buffer for CPUBuffer {
	fn zeros_(&self, tensor: &Tensor) {
		debug_assert!(self.is_on_my_device(tensor));

		match tensor.dtype() {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.zeros_f32_(tensor.offset, tensor.elems())
			},
			_ => todo!(),
		}
	}

	fn randn_(&self, tensor: &Tensor) {
		debug_assert!(self.is_on_my_device(tensor));

		match tensor.dtype() {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.randn_f32_(tensor.offset, tensor.elems())
			},
			_ => todo!(),
		}
	}

	unsafe fn rms_norm(
		&self,
		out: &Tensor,
		a: &Tensor,
		dim_size: usize,
		eps: f64,
		batch: &[BatchDim<1>],
	) {
		self.get_buffer_of(out).rms_norm_dtype(
			out.dtype(),
			out.offset,
			self.get_buffer_of(a),
			a.offset,
			dim_size,
			eps,
			batch,
		)
	}
	/*
		fn mm(&self, a: &Tensor, b: &Tensor, c: &Tensor) {
			// TODO - could relax this to just check that the tensors are on the same device
			debug_assert!(self.owns(a) && self.owns(b) && self.owns(c));

			let prep = prep_mm(a, b, c);

			match prep.dtype {
				DType { kind: DTypeKind::Float, bits: 32 } => self.mm_f32(a, b, c, prep),
				_ => todo!(),
			}
		}
	*/
	unsafe fn format(
		&self,
		f: &mut fmt::Formatter,
		dtype: DType,
		offset: usize,
		count: usize,
		stride: usize,
	) -> fmt::Result {
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => self.format_f32(offset, f, count, stride),
			_ => todo!(),
		}
	}
}
