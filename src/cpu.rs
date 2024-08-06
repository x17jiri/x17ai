// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use matrixmultiply::sgemm;
use std::boxed::Box;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::intrinsics::unlikely;

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

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: usize) -> Rc<dyn Buffer> {
		// we want to allocate `elems` elements of type `dtype`, but internally we use elements
		// of type `CPUBufferElement`. So convert `elems` to `buf_elems`.
		let size_bytes = dtype.array_bytes(elems).unwrap();
		let step_size = std::mem::size_of::<CPUBufferElement>();
		let buf_elems = (size_bytes + step_size - 1) / step_size;
		Rc::new(CPUBuffer {
			base: BufferBase {
				device: self.clone(),
				capacity: size_bytes,
			},
			// TODO - we could leave the memory uninitialized
			memory: vec![Cell::new(0); buf_elems].into_boxed_slice(),
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

	fn cast_buffer(&self, buf: &BufferBase) -> &CPUBuffer {
		debug_assert!(buf.is_on_device(self.base.device.as_ref()));
		let buf = buf as *const BufferBase;
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

	// internally uses f64 and then stores the result as f32
	// on modern CPUs, this should have no performance impact and we get better precision
	fn rms_norm_f32(&self, offset: usize, a: &Self, a_offset: usize, count: usize, eps: f64) {
		let out_vec = self.cast::<f32>(offset, count);
		let in_vec = a.cast::<f32>(a_offset, count);

		let mut sqr_sum = eps;
		for i in in_vec {
			let i = f64::from(i.get());
			sqr_sum += i * i;
		}

		let scale = ((count as f64) / sqr_sum).sqrt();

		for (o, i) in out_vec.iter().zip(in_vec) {
			let i = f64::from(i.get());
			o.set((i * scale) as f32);
		}
	}

	fn softmax_f32(&self, offset: usize, a: &Self, a_offset: usize, count: usize) {
		let out_vec = self.cast::<f32>(offset, count);

		let in_vec = a.cast::<f32>(a_offset, count);
		// convert to f64
		let in_iter = in_vec.iter().map(|x| f64::from(x.get()));
		// find max
		let max: f64 = in_iter.clone().fold(f64::NEG_INFINITY, f64::max);

		if unlikely(max == f64::NEG_INFINITY) {
			// all values are -inf, set all to 0
			let val = 1.0 / (count as f64);
			for o in out_vec.iter() {
				o.set(val as f32);
			}
			return;
		}

		let mut sum = 0.0;
		for (o, val) in out_vec.iter().zip(in_iter) {
			let val = (val - max).exp();
			o.set(val as f32);
			sum += val;
		}

		for o in out_vec.iter() {
			let val = o.get() / sum as f32;
			o.set(val);
		}
	}

	fn acc_f32(
		&self,
		offset: usize,
		a: &Self,
		a_offset: usize,
		count: usize,
		alpha: f64,
		beta: f64,
	) {
		let out_vec = self.cast::<f32>(offset, count);
		let in_vec = a.cast::<f32>(a_offset, count);

		for (o, i) in out_vec.iter().zip(in_vec) {
			let val = alpha * f64::from(o.get()) + beta * f64::from(i.get());
			o.set(val as f32);
		}
	}

	fn acc_sum_f32(
		&self,
		offset: usize,
		a: &Self,
		a_offset: usize,
		count: usize,
		alpha: f64,
		beta: f64,
	) {
		let in_vec = a.cast::<f32>(a_offset, count);

		let mut sum = 0.0;
		for i in in_vec {
			sum += f64::from(i.get());
		}

		let out_vec = self.cast::<f32>(offset, 1);
		let o = &out_vec[0];
		let val = alpha * f64::from(o.get()) + beta * sum;
		o.set(val as f32);
	}

	#[rustfmt::skip]
	#[allow(clippy::too_many_arguments)]
	fn gemm_f32(
		// self == c
		&self, c_offset: usize, ldc: usize,
		m: usize, n: usize, k: usize,
		a: &Self, a_offset: usize, lda: usize, transa: bool,
		b: &Self, b_offset: usize, ldb: usize, transb: bool,
		alpha: f64, beta: f64,
	) {
		let a = a.cast::<f32>(a_offset, 0).as_ptr();
		let b = b.cast::<f32>(b_offset, 0).as_ptr();
		let c = self.cast::<f32>(c_offset, 0).as_ptr();

		let (rsa, csa) = if transa { (1, lda) } else { (lda, 1) };
		let (rsb, csb) = if transb { (1, ldb) } else { (ldb, 1) };
		let (rsc, csc) = (ldc, 1_usize);

		unsafe {
			sgemm(
				m, k, n,
				alpha as f32,
				a as *const f32, rsa as isize, csa as isize,
				b as *const f32, rsb as isize, csb as isize,
				beta as f32,
				c as *mut f32, rsc as isize, csc as isize,
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
		o: BufOff<()>,
		a: BufOff<&BufferBase>,
		common: CommonArgs1D,
		eps: f64,
	) {
		let out = self;
		for i in 0..common.batch_size {
			let out_offset = o.offset + i * o.batch_stride;
			let a_offset = a.offset + i * a.batch_stride;
			let a = self.cast_buffer(a.buffer);
			match common.dtype {
				DType { kind: DTypeKind::Float, bits: 32 } => {
					out.rms_norm_f32(out_offset, a, a_offset, common.len, eps)
				},
				_ => todo!(),
			}
		}
	}

	unsafe fn softmax(
		&self,
		o: BufOff<()>,
		a: BufOff<&BufferBase>,
		common: CommonArgs1D, // rustfmt::newline
	) {
		let out = self;
		for i in 0..common.batch_size {
			let out_offset = o.offset + i * o.batch_stride;
			let a_offset = a.offset + i * a.batch_stride;
			let a = self.cast_buffer(a.buffer);
			match common.dtype {
				DType { kind: DTypeKind::Float, bits: 32 } => {
					out.softmax_f32(out_offset, a, a_offset, common.len);
				},
				_ => todo!(),
			}
		}
	}

	unsafe fn acc(
		&self,
		o: BufOff<()>,
		a: BufOff<&BufferBase>,
		common: CommonArgs1D,
		alpha: f64,
		beta: f64,
	) {
		let out = self;
		for i in 0..common.batch_size {
			let out_offset = o.offset + i * o.batch_stride;
			let a_offset = a.offset + i * a.batch_stride;
			let a = self.cast_buffer(a.buffer);
			match common.dtype {
				DType { kind: DTypeKind::Float, bits: 32 } => {
					out.acc_f32(out_offset, a, a_offset, common.len, alpha, beta)
				},
				_ => todo!(),
			}
		}
	}

	unsafe fn acc_sum(
		&self,
		o: BufOff<()>,
		a: BufOff<&BufferBase>,
		common: CommonArgs1D,
		alpha: f64,
		beta: f64,
	) {
		let out = self;
		for i in 0..common.batch_size {
			let out_offset = o.offset + i * o.batch_stride;
			let a_offset = a.offset + i * a.batch_stride;
			let a = self.cast_buffer(a.buffer);
			match common.dtype {
				DType { kind: DTypeKind::Float, bits: 32 } => {
					out.acc_sum_f32(out_offset, a, a_offset, common.len, alpha, beta)
				},
				_ => todo!(),
			}
		}
	}


	#[rustfmt::skip]
	unsafe fn gemm(
		&self, dtype: DType, c_offset: usize, ldc: usize, c_batch_stride: usize,
		m: usize, n: usize, k: usize,
		a: &BufferBase, a_offset: usize, lda: usize, transa: bool, a_batch_stride: usize,
		b: &BufferBase, b_offset: usize, ldb: usize, transb: bool, b_batch_stride: usize,
		alpha: f64, beta: f64,
		batch_size: usize,
	) {
		let c = self;
		let a = self.cast_buffer(a);
		let b = self.cast_buffer(b);
		for i in 0..batch_size {
			let c_offset = c_offset + i * c_batch_stride;
			let a_offset = a_offset + i * a_batch_stride;
			let b_offset = b_offset + i * b_batch_stride;
			match dtype {
				DType { kind: DTypeKind::Float, bits: 32 } => {
					c.gemm_f32(
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
