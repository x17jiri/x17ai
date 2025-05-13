// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::slice;
use matrixmultiply::sgemm;
use std::boxed::Box;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::intrinsics::{cold_path, unlikely};
use std::ops::Div;

mod math {
	use super::FromToF64;
	use std::cell::Cell;

	pub fn dot_prod<T: Copy + FromToF64>(a: &[Cell<T>], b: &[Cell<T>]) -> f64 {
		let res = a.iter().zip(b).map(|(a, b)| a.get().to_f64() * b.get().to_f64()).sum();
		//println!("dot_prod: {}", res);
		res
	}

	pub fn rsqrt(a: f64) -> f64 {
		let res = 1.0 / a.sqrt();
		//println!("rsqrt: {}", res);
		res
	}
}

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

impl ToBufferBase for CPUBuffer {
	fn to_buffer_base(&self) -> &BufferBase {
		&self.base
	}
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
	slices: [&'a TypedSliceSet<'a, CPUBuffer>; N],
	i: usize,
	batch_size: usize,
	phantom: std::marker::PhantomData<T>,
}

impl<'a, T: 'a, const N: usize> BatchIter<'a, T, N> {
	fn new(slices: [&'a TypedSliceSet<'a, CPUBuffer>; N]) -> Self {
		let batch_size = slices[0].batch_size;
		for slice in slices {
			debug_assert!(slice.dtype.bytes() == std::mem::size_of::<T>());
			assert!(slice.batch_size == batch_size);
		}
		BatchIter {
			slices,
			i: 0,
			batch_size,
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
		let out = self.slices.map(|slices| {
			let offset = slices.offset + self.i * slices.batch_stride;
			let elems = slices.len;
			// SAFETY: In `CPUBuffer.cast_slices()`, we assert that all slices are in bounds
			unsafe { slices.buffer.cast::<T>(offset, elems) }
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
	fn cast_slices<'a>(&'a self, slices: &'a SliceSet<'a>) -> TypedSliceSet<'a, CPUBuffer> {
		assert!(BufferBase::are_slices_in_bounds(slices));
		slices.to_typed_slice_set(|buf| self.cast_buffer(BufferBase::from_dyn_buf(buf)))
	}

	#[inline]
	fn cast_matrices<'a>(&'a self, mat: &'a MatrixSet<'a>) -> TypedMatrixSet<'a, CPUBuffer> {
		assert!(BufferBase::are_matrices_in_bounds(mat));
		mat.to_typed_matrix_set(|buf| self.cast_buffer(BufferBase::from_dyn_buf(buf)))
	}

	#[inline]
	fn is_in_bounds<T>(&self, dtype: DType, offset: usize, len: usize) -> bool {
		debug_assert!(dtype.bytes() == std::mem::size_of::<T>());
		let elems = offset + len;
		let bytes = std::mem::size_of::<T>().checked_mul(elems);
		bytes.is_some_and(|b| b <= self.base.size_bytes)
	}

	#[inline]
	unsafe fn cast<T>(&self, offset: usize, elems: usize) -> &[Cell<T>] {
		debug_assert!(self.base.is_in_bounds_T::<T>(offset, elems));
		let ptr = self.memory.as_ptr();
		let ptr = ptr as *const Cell<T>;
		let ptr = ptr.wrapping_add(offset);
		unsafe { std::slice::from_raw_parts(ptr, elems) }
	}

	fn zeros<T: Default>(dst: &TypedSliceSet<'_, CPUBuffer>) {
		for [dst_arr] in BatchIter::<T, 1>::new([dst]) {
			for d in dst_arr {
				d.set(T::default());
			}
		}
	}

	fn randn_f<T: FromToF64>(dst: &TypedSliceSet<'_, CPUBuffer>) {
		let mut rng = dst.buffer.device().rng.borrow_mut();
		for [dst_arr] in BatchIter::<T, 1>::new([dst]) {
			for d in dst_arr {
				d.set(T::from_f64(rng.get_normal()));
			}
		}
	}

	fn copy<T: Copy>(dst: &TypedSliceSet<'_, CPUBuffer>, src: &TypedSliceSet<'_, CPUBuffer>) {
		assert!(dst.len == src.len);
		for [dst_arr, src_arr] in BatchIter::<T, 2>::new([dst, src]) {
			for (d, s) in dst_arr.iter().zip(src_arr) {
				d.set(s.get());
			}
		}
	}

	fn acc_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, dst_weight: f64, new: &TypedSliceSet<'_, CPUBuffer>,
		new_weight: f64,
	) {
		assert!(dst.len == new.len);
		for [dst_arr, new_arr] in BatchIter::<T, 2>::new([dst, new]) {
			for (d, n) in dst_arr.iter().zip(new_arr) {
				let d_val = d.get().to_f64();
				let n_val = n.get().to_f64();
				let d_val = d_val * dst_weight + n_val * new_weight;
				d.set(T::from_f64(d_val));
			}
		}
	}

	fn mul_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, a: &TypedSliceSet<'_, CPUBuffer>,
		b: &TypedSliceSet<'_, CPUBuffer>,
	) {
		assert!(dst.len == a.len);
		assert!(dst.len == b.len);
		for [dst_arr, a_arr, b_arr] in BatchIter::<T, 3>::new([dst, a, b]) {
			for ((d, a), b) in dst_arr.iter().zip(a_arr).zip(b_arr) {
				let val = a.get().to_f64() * b.get().to_f64();
				d.set(T::from_f64(val));
			}
		}
	}

	fn mul_acc_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, dst_weight: f64, a: &TypedSliceSet<'_, CPUBuffer>,
		b: &TypedSliceSet<'_, CPUBuffer>, ab_weight: f64,
	) {
		assert!(dst.len == a.len);
		assert!(dst.len == b.len);
		for [dst_arr, a_arr, b_arr] in BatchIter::<T, 3>::new([dst, a, b]) {
			for ((d, a), b) in dst_arr.iter().zip(a_arr).zip(b_arr) {
				let d_val = d.get().to_f64();
				let a_val = a.get().to_f64();
				let b_val = b.get().to_f64();
				let val = d_val * dst_weight + a_val * b_val * ab_weight;
				d.set(T::from_f64(val));
			}
		}
	}

	fn vec_mul_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, a: &TypedSliceSet<'_, CPUBuffer>,
		b: &TypedSliceSet<'_, CPUBuffer>,
	) {
		assert!(a.len == b.len);
		assert!(dst.len == 1);
		for [dst_arr, a_arr, b_arr] in BatchIter::<T, 3>::new([dst, a, b]) {
			let val = math::dot_prod(a_arr, b_arr);
			dst_arr[0].set(T::from_f64(val));
		}
	}

	fn vec_mul_acc_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, dst_weight: f64, a: &TypedSliceSet<'_, CPUBuffer>,
		b: &TypedSliceSet<'_, CPUBuffer>, ab_weight: f64,
	) {
		assert!(a.len == b.len);
		assert!(dst.len == 1);
		for [dst_arr, a_arr, b_arr] in BatchIter::<T, 3>::new([dst, a, b]) {
			let old_val = dst_arr[0].get().to_f64();
			let new_val = dst_weight * old_val + ab_weight * math::dot_prod(a_arr, b_arr);
			dst_arr[0].set(T::from_f64(new_val));
		}
	}

	fn rsqrt_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, inp: &TypedSliceSet<'_, CPUBuffer>, eps: f64,
	) {
		assert!(dst.len == inp.len);
		for [dst_arr, inp_arr] in BatchIter::<T, 2>::new([dst, inp]) {
			for (d, i) in dst_arr.iter().zip(inp_arr) {
				let val = math::rsqrt(i.get().to_f64() + eps);
				d.set(T::from_f64(val));
			}
		}
	}

	fn softmax_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, inp: &TypedSliceSet<'_, CPUBuffer>,
	) {
		assert!(dst.len == inp.len);

		for [dst_arr, inp_arr] in BatchIter::<T, 2>::new([dst, inp]) {
			let max: f64 = inp_arr.iter().map(|x| x.get().to_f64()).fold(f64::MIN, f64::max);

			let mut sum = 0.0;
			for (d, i) in dst_arr.iter().zip(inp_arr) {
				let val = i.get().to_f64();
				let val = val - max; // avoid overflow
				let e = val.exp();
				d.set(T::from_f64(e));

				sum += e;
			}

			// NOTE: Subtracting max in the loop above ensures at least one of the exponents
			// is `exp(max - max) == exp(0) == 1.0`. So sum will be >= 1.0. This could only fail
			// if all inputs are `-inf`. Make sure that in that case we don't divide by zero.
			sum = sum.max(1.0);

			for d in dst_arr.iter() {
				let val = d.get().to_f64() / sum; // sum >= 1.0, so no div by zero
				d.set(T::from_f64(val));
			}
		}
	}

	fn rms_norm_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, inp: &TypedSliceSet<'_, CPUBuffer>, eps: f64,
	) {
		assert!(dst.len == inp.len);
		let len = dst.len;
		let rlen = 1.0 / (len as f64);

		for [dst_arr, inp_arr] in BatchIter::<T, 2>::new([dst, inp]) {
			let scale = math::rsqrt(math::dot_prod(inp_arr, inp_arr) * rlen + eps);
			for (d, i) in dst_arr.iter().zip(inp_arr) {
				let val = i.get().to_f64() * scale;
				d.set(T::from_f64(val));
			}
		}
	}

	fn gemm_f32(
		c: &TypedMatrixSet<'_, CPUBuffer>, dst_weight: f64, a: &TypedMatrixSet<'_, CPUBuffer>,
		b: &TypedMatrixSet<'_, CPUBuffer>, ab_weight: f64,
	) {
		let m = c.rows.get();
		let n = c.cols.get();
		let k = a.cols.get();

		assert!(a.rows.get() == m);
		assert!(b.cols.get() == n);
		assert!(b.rows.get() == k);

		for [c_arr, a_arr, b_arr] in
			BatchIter::<f32, 3>::new([&c.slice_set, &a.slice_set, &b.slice_set])
		{
			// SAFETY: TODO
			#[rustfmt::skip] unsafe {
				sgemm(
					m, k, n,
					dst_weight as f32,
					a_arr.as_ptr() as *const f32, a.row_stride as isize, a.col_stride as isize,
					b_arr.as_ptr() as *const f32, b.row_stride as isize, b.col_stride as isize,
					ab_weight as f32,
					c_arr.as_ptr() as *mut f32, c.row_stride as isize, c.col_stride as isize,
				);
			}
		}
	}

	fn format_f<T: Copy + FromToF64>(
		&self, f: &mut fmt::Formatter, offset: usize, len: usize, stride: usize,
	) -> fmt::Result {
		let slices = TypedSliceSet {
			buffer: self,
			dtype: DType::f32(),
			offset,

			len: 1,
			batch_size: len,
			batch_stride: stride,
		};
		let mut first_item = true;
		for [arr] in BatchIter::<T, 1>::new([&slices]) {
			if !first_item {
				write!(f, ", ")?;
			}
			first_item = false;

			write!(f, "{:.6}", arr[0].get().to_f64())?;
		}
		Ok(())
	}
}

impl Buffer for CPUBuffer {
	fn zeros(&self, dst: &SliceSet) {
		let dst = self.cast_slices(dst);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => Self::zeros::<f32>(&dst),
			_ => todo!(),
		}
	}

	fn randn(&self, dst: &SliceSet) {
		let dst = self.cast_slices(dst);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => Self::randn_f::<f32>(&dst),
			_ => todo!(),
		}
	}

	fn copy(&self, dst: &SliceSet, src: &SliceSet) {
		let dst = self.cast_slices(dst);
		let src = self.cast_slices(src);
		assert!(dst.dtype == src.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => Self::copy::<f32>(&dst, &src),
			_ => todo!(),
		}
	}

	fn acc(&self, dst: &SliceSet, dst_weight: f64, new: &SliceSet, new_weight: f64) {
		let dst = self.cast_slices(dst);
		let new = self.cast_slices(new);
		assert!(dst.dtype == new.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				Self::acc_f::<f32>(&dst, dst_weight, &new, new_weight)
			},
			_ => todo!(),
		}
	}

	fn mul(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet) {
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		assert!(dst.dtype == a.dtype);
		assert!(dst.dtype == b.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => Self::mul_f::<f32>(&dst, &a, &b),
			_ => todo!(),
		}
	}

	fn mul_acc(&self, dst: &SliceSet, dst_weight: f64, a: &SliceSet, b: &SliceSet, ab_weight: f64) {
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		assert!(dst.dtype == a.dtype);
		assert!(dst.dtype == b.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				Self::mul_acc_f::<f32>(&dst, dst_weight, &a, &b, ab_weight)
			},
			_ => todo!(),
		}
	}

	fn vec_mul(&self, dst_slices: &SliceSet, a: &SliceSet, b: &SliceSet) {
		let dst = self.cast_slices(dst_slices);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		assert!(dst.dtype == a.dtype);
		assert!(dst.dtype == b.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => Self::vec_mul_f::<f32>(&dst, &a, &b),
			_ => todo!(),
		}
	}

	fn vec_mul_acc(
		&self, dst: &SliceSet, dst_weight: f64, a: &SliceSet, b: &SliceSet, ab_weight: f64,
	) {
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		assert!(dst.dtype == a.dtype);
		assert!(dst.dtype == b.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				Self::vec_mul_acc_f::<f32>(&dst, dst_weight, &a, &b, ab_weight)
			},
			_ => todo!(),
		}
	}

	fn rsqrt(&self, dst: &SliceSet, inp: &SliceSet, eps: f64) {
		let dst = self.cast_slices(dst);
		let inp = self.cast_slices(inp);
		assert!(dst.dtype == inp.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => Self::rsqrt_f::<f32>(&dst, &inp, eps),
			_ => todo!(),
		}
	}

	fn softmax(&self, dst: &SliceSet, inp: &SliceSet) {
		let dst = self.cast_slices(dst);
		let inp = self.cast_slices(inp);
		assert!(dst.dtype == inp.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => Self::softmax_f::<f32>(&dst, &inp),
			_ => todo!(),
		}
	}

	fn rms_norm(&self, dst: &SliceSet, inp: &SliceSet, eps: f64) {
		let dst = self.cast_slices(dst);
		let inp = self.cast_slices(inp);
		assert!(dst.dtype == inp.dtype);
		match dst.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => Self::rms_norm_f::<f32>(&dst, &inp, eps),
			_ => todo!(),
		}
	}

	fn gemm(&self, dst: &MatrixSet, dst_weight: f64, a: &MatrixSet, b: &MatrixSet, ab_weight: f64) {
		let dst = self.cast_matrices(dst);
		let a = self.cast_matrices(a);
		let b = self.cast_matrices(b);
		assert!(dst.slice_set.dtype == a.slice_set.dtype);
		assert!(dst.slice_set.dtype == b.slice_set.dtype);
		match dst.slice_set.dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				Self::gemm_f32(&dst, dst_weight, &a, &b, ab_weight);
			},
			_ => todo!(),
		}
	}

	fn format(
		&self, f: &mut fmt::Formatter, dtype: DType, offset: usize, len: usize, stride: usize,
	) -> fmt::Result {
		match dtype {
			DType { kind: DTypeKind::Float, bits: 32 } => {
				self.format_f::<f32>(f, offset, len, stride)
			},
			_ => todo!(),
		}
	}
}
