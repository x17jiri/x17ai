// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::slice;
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
			base: BufferBase { device: self.clone(), size_bytes },
			// TODO - we could leave the memory uninitialized
			memory: vec![Cell::new(0); buf_elems].into_boxed_slice(),
		})
	}
}

trait FromToF64 {
	fn from_f64(val: f64) -> Self;
	fn to_f64(&self) -> f64;
}

impl FromToF64 for f32 {
	fn from_f64(val: f64) -> Self {
		val as f32
	}

	fn to_f64(&self) -> f64 {
		*self as f64
	}
}

impl FromToF64 for f64 {
	fn from_f64(val: f64) -> Self {
		val
	}

	fn to_f64(&self) -> f64 {
		*self
	}
}

impl CPUDevice {
	// .
}

struct BatchIter<'a, T, const N: usize> {
	buffers: [(&'a CPUBuffer, &'a SliceSet); N],
	i: usize,
	batch_size: usize,
	phantom: std::marker::PhantomData<T>,
}

impl<'a, T: 'a, const N: usize> BatchIter<'a, T, N> {
	fn new(buffers: [(&'a CPUBuffer, &'a SliceSet); N]) -> Self {
		let first_slices = buffers[0].1;
		for (buf, slices) in buffers {
			assert!(buf.base.are_slices_in_bounds_T::<T>(slices));
			assert!(first_slices.batch_size == slices.batch_size);
		}
		BatchIter {
			buffers,
			i: 0,
			batch_size: first_slices.batch_size,
			phantom: std::marker::PhantomData,
		}
	}
}

impl<'a, T: 'a, const N: usize> Iterator for BatchIter<'a, T, N> {
	type Item = [&'a [Cell<T>]; N];

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.i >= self.batch_size {
			return None;
		}
		let out = self.buffers.map(|(buf, slices)| {
			let offset = slices.offset + self.i * slices.batch_stride;
			let elems = slices.len;
			// SAFETY: In `new()`, we assert that all slices are in bounds
			unsafe { buf.cast::<T>(offset, elems) }
		});
		self.i += 1;
		Some(out)
	}
}

impl CPUBuffer {
	#[inline]
	fn device(&self) -> &CPUDevice {
		let dev = self.base.device.as_ref();
		let dev = dev as *const dyn Device;
		let dev = dev as *const CPUDevice;
		unsafe { &*dev }
	}

	#[inline]
	fn cast_buffer(&self, buf: &BufferBase) -> &CPUBuffer {
		assert!(buf.is_on_device(self.base.device.as_ref()));
		let buf = buf as *const BufferBase;
		let buf = buf as *const CPUBuffer;
		unsafe { &*buf }
	}

	#[inline]
	unsafe fn cast<T>(&self, offset: usize, elems: usize) -> &[Cell<T>] {
		debug_assert!(self.base.is_in_bounds_T::<T>(offset, elems));
		let ptr = self.memory.as_ptr();
		let ptr = ptr as *const Cell<T>;
		let ptr = ptr.wrapping_add(offset);
		unsafe { std::slice::from_raw_parts(ptr, elems) }
	}

	fn zeros<T: Default>(&self, dst_slices: &SliceSet) {
		for [dst] in BatchIter::<T, 1>::new([(&self, dst_slices)]) {
			for d in dst {
				d.set(T::default());
			}
		}
	}

	fn randn_f<T: FromToF64>(&self, dst_slices: &SliceSet) {
		let mut rng = self.device().rng.borrow_mut();
		for [dst] in BatchIter::<T, 1>::new([(&self, dst_slices)]) {
			for d in dst {
				d.set(T::from_f64(rng.get_normal()));
			}
		}
	}

	fn copy<T: Copy>(&self, dst_slices: &SliceSet, src: &CPUBuffer, src_slices: &SliceSet) {
		assert!(dst_slices.len == src_slices.len);
		for [dst, src] in BatchIter::<T, 2>::new([(&self, dst_slices), (src, src_slices)]) {
			for (d, s) in dst.iter().zip(src) {
				d.set(s.get());
			}
		}
	}

	fn acc_f<T: Copy + FromToF64>(
		&self, dst_slices: &SliceSet, dst_weight: f64, b: &CPUBuffer, b_slices: &SliceSet,
		b_weight: f64,
	) {
		assert!(dst_slices.len == b_slices.len);
		for [dst, b] in BatchIter::<T, 2>::new([(&self, dst_slices), (b, b_slices)]) {
			for (d, b) in dst.iter().zip(b) {
				let d_val = d.get().to_f64();
				let b_val = b.get().to_f64();
				let d_val = d_val * dst_weight + b_val * b_weight;
				d.set(T::from_f64(d_val));
			}
		}
	}

	fn vec_mul_f<T: Copy + FromToF64>(
		&self, dst_slices: &SliceSet, a: &CPUBuffer, a_slices: &SliceSet, b: &CPUBuffer,
		b_slices: &SliceSet,
	) {
		assert!(a_slices.len == b_slices.len);
		assert!(dst_slices.len > 0);
		for [dst, a, b] in
			BatchIter::<T, 3>::new([(&self, dst_slices), (a, a_slices), (b, b_slices)])
		{
			let prod = a.iter().zip(b).map(|(a, b)| a.get().to_f64() * b.get().to_f64()).sum();
			dst[0].set(T::from_f64(prod));
		}
	}

	fn vec_mul_acc_f<T: Copy + FromToF64>(
		&self, dst_slices: &SliceSet, dst_weight: f64, a: &CPUBuffer, a_slices: &SliceSet,
		b: &CPUBuffer, b_slices: &SliceSet, ab_weight: f64,
	) {
		assert!(a_slices.len == b_slices.len);
		assert!(dst_slices.len > 0);
		for [dst, a, b] in
			BatchIter::<T, 3>::new([(&self, dst_slices), (a, a_slices), (b, b_slices)])
		{
			let prod: f64 = a.iter().zip(b).map(|(a, b)| a.get().to_f64() * b.get().to_f64()).sum();
			let d_val = dst[0].get().to_f64();
			let d_val = d_val * dst_weight + prod * ab_weight;
			dst[0].set(T::from_f64(d_val));
		}
	}

	fn rsqrt_f<T: Copy + FromToF64>(
		&self, dst_slices: &SliceSet, a: &CPUBuffer, a_slices: &SliceSet, eps: f64,
	) {
		assert!(dst_slices.len == a_slices.len);
		for [dst, a] in BatchIter::<T, 2>::new([(&self, dst_slices), (a, a_slices)]) {
			for (d, a) in dst.iter().zip(a) {
				let val = 1.0 / (a.get().to_f64() + eps).sqrt();
				d.set(T::from_f64(val));
			}
		}
	}

	fn softmax_f<T: Copy + FromToF64>(
		&self, dst_slices: &SliceSet, a: &CPUBuffer, a_slices: &SliceSet,
	) {
		assert!(dst_slices.len == a_slices.len);
		assert!(dst_slices.len > 0); // avoid division by zero
		let len = dst_slices.len;

		for [dst, a] in BatchIter::<T, 2>::new([(&self, dst_slices), (a, a_slices)]) {
			// convert to f64
			let a_iter = a.iter().map(|x| x.get().to_f64());
			// find max
			let max: f64 = a_iter.clone().fold(f64::NEG_INFINITY, f64::max);

			if unlikely(max == f64::NEG_INFINITY) {
				// all values are -inf, set all to 0
				let val = T::from_f64(1.0 / (len as f64));
				for o in dst {
					o.set(val);
				}
				return;
			}

			let mut sum = 0.0;
			for (d, a) in dst.iter().zip(a) {
				let e = (a.get().to_f64() - max).exp();
				d.set(T::from_f64(e));

				sum += e;
			}

			for d in dst.iter() {
				// NOTE: Subtracting max in the loop above ensures at least one exp(0) = 1.0,
				// so sum will be >= 1.0 and division by zero is impossible.
				let val = d.get().to_f64() / sum;
				d.set(T::from_f64(val));
			}
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

	#[allow(clippy::too_many_arguments)]
	fn acc_mul_f32(
		&self, offset: usize, b: &Self, b_offset: usize, c: &Self, c_offset: usize, count: usize,
		alpha: f64, beta: f64,
	) {
		let out_vec = self.cast::<f32>(offset, count);
		let b_vec = b.cast::<f32>(b_offset, count);
		let c_vec = c.cast::<f32>(c_offset, count);

		for ((o, b), c) in out_vec.iter().zip(b_vec).zip(c_vec) {
			let val = alpha * f64::from(o.get()) + beta * f64::from(b.get()) * f64::from(c.get());
			o.set(val as f32);
		}
	}

	fn acc_sum_f32(
		&self, offset: usize, b: &Self, b_offset: usize, count: usize, alpha: f64, beta: f64,
	) {
		let in_vec = b.cast::<f32>(b_offset, count);

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
		&self, offset: usize, f: &mut fmt::Formatter, count: usize, stride: usize,
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
	fn zeros(&self, dtype: DType, dst_slices: &SliceSet) {
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => self.zeros::<f32>(dst_slices),
			_ => todo!(),
		}
	}

	fn randn(&self, dtype: DType, dst_slices: &SliceSet) {
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => self.randn_f::<f32>(dst_slices),
			_ => todo!(),
		}
	}

	fn copy(&self, dtype: DType, dst_slices: &SliceSet, src: &BufferBase, src_slices: &SliceSet) {
		let src = self.cast_buffer(src);
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.copy::<f32>(dst_slices, src, src_slices)
			},
			_ => todo!(),
		}
	}

	fn acc(
		&self, dtype: DType, dst_slices: &SliceSet, dst_weight: f64, b: &BufferBase,
		b_slices: &SliceSet, b_weight: f64,
	) {
		let b = self.cast_buffer(b);
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.acc_f::<f32>(dst_slices, dst_weight, b, b_slices, b_weight)
			},
			_ => todo!(),
		}
	}

	fn vec_mul(
		&self, dtype: DType, dst_slices: &SliceSet, a: &BufferBase, a_slices: &SliceSet,
		b: &BufferBase, b_slices: &SliceSet,
	) {
		let a = self.cast_buffer(a);
		let b = self.cast_buffer(b);
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.vec_mul_f::<f32>(dst_slices, a, a_slices, b, b_slices)
			},
			_ => todo!(),
		}
	}

	fn vec_mul_acc(
		&self, dtype: DType, dst_slices: &SliceSet, dst_weight: f64, a: &BufferBase,
		a_slices: &SliceSet, b: &BufferBase, b_slices: &SliceSet, ab_weight: f64,
	) {
		let a = self.cast_buffer(a);
		let b = self.cast_buffer(b);
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => self
				.vec_mul_acc_f::<f32>(dst_slices, dst_weight, a, a_slices, b, b_slices, ab_weight),
			_ => todo!(),
		}
	}

	fn rsqrt(
		&self, dtype: DType, dst_slices: &SliceSet, a: &BufferBase, a_slices: &SliceSet, eps: f64,
	) {
		let a = self.cast_buffer(a);
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.rsqrt_f::<f32>(dst_slices, a, a_slices, eps)
			},
			_ => todo!(),
		}
	}

	fn softmax(&self, dtype: DType, dst_slices: &SliceSet, a: &BufferBase, a_slices: &SliceSet) {
		let a = self.cast_buffer(a);
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.softmax_f::<f32>(dst_slices, a, a_slices)
			},
			_ => todo!(),
		}
	}

	unsafe fn rms_norm(
		&self, o: BatchBufOff<()>, a: BatchBufOff<&BufferBase>, common: CommonArgs1D, eps: f64,
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
		&self, f: &mut fmt::Formatter, dtype: DType, offset: usize, count: usize, stride: usize,
	) -> fmt::Result {
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => self.format_f32(offset, f, count, stride),
			_ => todo!(),
		}
	}
}
