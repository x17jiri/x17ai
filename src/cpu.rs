// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use matrixmultiply::sgemm;
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
			let batch_dim = batch[batch.len() - 1];
			let batch = &batch[..batch.len() - 1];
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

	// internally uses f64 and then stores the result as f32
	// on modern CPUs, this should have no performance impact and we get better precision
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

	#[rustfmt::skip]
	fn gemm_dtype(
		&self, dtype: DType, c_offset: usize, ldc: usize, // self == c
		m: usize, n: usize, k: usize,
		a: &Self, a_offset: usize, lda: usize, transa: bool,
		b: &Self, b_offset: usize, ldb: usize, transb: bool,
		alpha: f64, beta: f64,
		batch: &[BatchDim<2>],
	) {
		if !batch.is_empty() {
			let batch_dim = batch[batch.len() - 1];
			let batch = &batch[..batch.len() - 1];
			for i in 0..batch_dim.size {
				self.gemm_dtype(
					dtype, c_offset + i * batch_dim.out_stride, ldc,
					m, n, k,
					a, a_offset + i * batch_dim.in_strides[0], lda, transa,
					b, b_offset + i * batch_dim.in_strides[1], ldb, transb,
					alpha, beta,
					batch,
				);
			}
			return;
		}

		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.gemm_f32(
					c_offset, ldc,
					m, n, k,
					a, a_offset, lda, transa,
					b, b_offset, ldb, transb,
					alpha, beta,
				);
			},
			_ => todo!(),
		}
	}

	#[rustfmt::skip]
	fn gemm_f32(
		&self, c_offset: usize, ldc: usize, // self == c
		m: usize, n: usize, k: usize,
		a: &Self, a_offset: usize, lda: usize, transa: bool,
		b: &Self, b_offset: usize, ldb: usize, transb: bool,
		alpha: f64, beta: f64,
	) {
		let (rsa, csa) = if transa { (1, lda) } else { (lda, 1) };
		let (rsb, csb) = if transb { (1, ldb) } else { (ldb, 1) };
		let (rsc, csc) = (ldc, 1_usize);

		unsafe {
			sgemm(
				m, k, n,
				alpha as f32,
				a.cast::<f32>(a_offset, 0).as_ptr() as *const f32, rsa as isize, csa as isize,
				b.cast::<f32>(b_offset, 0).as_ptr() as *const f32, rsb as isize, csb as isize,
				beta as f32,
				self.cast::<f32>(c_offset, 0).as_ptr() as *mut f32, rsc as isize, csc as isize,
			);
		}
	}

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

	#[rustfmt::skip]
	unsafe fn gemm(
		&self, c: &Tensor, ldc: usize,
		m: usize, n: usize, k: usize,
		a: &Tensor, lda: usize, transa: bool,
		b: &Tensor, ldb: usize, transb: bool,
		alpha: f64, beta: f64,
		batch: &[BatchDim<2>],
	) {
		self.get_buffer_of(c).gemm_dtype(
			c.dtype(), c.offset, ldc,
			m, n, k,
			self.get_buffer_of(a), a.offset, lda, transa,
			self.get_buffer_of(b), b.offset, ldb, transb,
			alpha, beta,
			batch,
		)
	}

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
