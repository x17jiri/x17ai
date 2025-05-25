// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::boxed::Box;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::Rc;

mod rng;

use rng::Rng;

use crate::tensor::buffer::{
	Buffer, BufferBase, MatrixSet, SliceSet, ToBufferBase, TypedMatrixSet, TypedSliceSet,
};
use crate::tensor::dtype::DType;
use crate::tensor::{Tensor, TensorSize, tensor_size_to_usize};

use super::Device;

mod math {
	use super::FromToF64;
	use std::cell::Cell;

	pub fn dot<T: Copy + FromToF64>(a: &[Cell<T>], b: &[Cell<T>]) -> f64 {
		let res = a.iter().zip(b).map(|(a, b)| a.get().to_f64() * b.get().to_f64()).sum();
		//println!("dot: {}", res);
		res
	}

	pub fn rsqrt(a: f64) -> f64 {
		1.0 / a.sqrt()
	}

	pub fn sigmoid(x: f64) -> f64 {
		1.0 / (1.0 + (-x).exp())
	}

	pub fn swish(x: f64) -> f64 {
		let sigmoid = sigmoid(x);
		x * sigmoid
	}

	pub fn swiglu(lin: f64, gate: f64) -> f64 {
		let swish = swish(gate);
		lin * swish
	}

	pub fn swiglu_backward(lin: f64, gate: f64) -> (f64, f64) {
		let sigmoid = sigmoid(gate);
		let swish = gate * sigmoid;

		let d_lin = swish;
		let d_gate = lin * (swish + sigmoid * (1.0 - swish));

		(d_lin, d_gate)
	}
}

#[derive(std::marker::ConstParamTy, PartialEq, Eq)]
enum Commutativity {
	Commutative,
	NonCommutative,
}

const Commutative: Commutativity = Commutativity::Commutative;
const NonCommutative: Commutativity = Commutativity::NonCommutative;

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

	pub fn tensor_as_slice<'a, T>(&self, tensor: &'a Tensor) -> &'a [Cell<T>] {
		assert!(tensor.dtype.bytes() == std::mem::size_of::<T>());

		let buf_base = BufferBase::from_dyn_buf(tensor.buffer.as_ref());
		assert!(buf_base.is_on_device(self));

		let buf_base = buf_base as *const BufferBase;
		let buf = buf_base as *const CPUBuffer;
		let buf = unsafe { &*buf };

		let offset = tensor.offset;
		let mut elems = 1;
		for i in 0..tensor.ndim() {
			let dim = tensor.dim_from_start(i);
			if dim.size == 0 {
				return &[];
			}

			elems += (dim.size - 1) * dim.stride;
		}

		assert!(buf.base.is_in_bounds_T::<T>(offset, elems));
		unsafe { buf.cast::<T>(offset, elems) }
	}

	fn new_cpu_buffer(self: Rc<Self>, dtype: DType, elems: TensorSize) -> Rc<CPUBuffer> {
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

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: TensorSize) -> Rc<dyn Buffer> {
		self.new_cpu_buffer(dtype, elems)
	}
}

trait FromToF64 {
	const MIN: f64; // largest negative value of type

	fn from_f64(val: f64) -> Self;
	fn to_f64(&self) -> f64;
}

impl FromToF64 for f32 {
	const MIN: f64 = f32::MIN as f64;

	fn from_f64(val: f64) -> Self {
		val as f32
	}

	fn to_f64(&self) -> f64 {
		*self as f64
	}
}

impl FromToF64 for f64 {
	const MIN: f64 = f64::MIN;

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
	i: TensorSize,
	batch_size: TensorSize,
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
	fn is_in_bounds<T>(&self, dtype: DType, offset: TensorSize, len: TensorSize) -> bool {
		debug_assert!(dtype.bytes() == std::mem::size_of::<T>());
		let elems = tensor_size_to_usize(offset + len);
		let bytes = std::mem::size_of::<T>().checked_mul(elems);
		bytes.is_some_and(|b| b <= self.base.size_bytes)
	}

	#[inline]
	unsafe fn cast<T>(&self, offset: TensorSize, elems: TensorSize) -> &[Cell<T>] {
		debug_assert!(self.base.is_in_bounds_T::<T>(offset, elems));
		let ptr = self.memory.as_ptr();
		let ptr = ptr as *const Cell<T>;
		let ptr = ptr.wrapping_add(tensor_size_to_usize(offset));
		unsafe { std::slice::from_raw_parts(ptr, tensor_size_to_usize(elems)) }
	}

	fn array_wise<T: Copy, const N: usize>(
		slices: [&TypedSliceSet<'_, CPUBuffer>; N], mut f: impl FnMut([&[Cell<T>]; N]),
	) {
		let dtype = slices[0].dtype;
		assert!(slices.iter().all(|i| i.dtype == dtype));
		for arrays in BatchIter::<T, N>::new(slices) {
			f(arrays);
		}
	}

	fn elem_wise<T: Copy, const N: usize>(
		slices: [&TypedSliceSet<'_, CPUBuffer>; N], mut f: impl FnMut([&Cell<T>; N]),
	) {
		let len = slices[0].len;
		assert!(slices.iter().all(|i| i.len == len));
		Self::array_wise::<T, N>(slices, |arrays| {
			for i in 0..len {
				f(arrays.map(|arr| unsafe { arr.get_unchecked(i as usize) }));
			}
		});
	}

	fn elem_wise_bin<T: Copy, const C: Commutativity>(
		dst: &TypedSliceSet<'_, CPUBuffer>, a_inp: &TypedSliceSet<'_, CPUBuffer>,
		b_inp: &TypedSliceSet<'_, CPUBuffer>, f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		let dtype = dst.dtype;
		assert!(a_inp.dtype == dtype);
		assert!(b_inp.dtype == dtype);
		if a_inp.len == 1 && b_inp.len == 1 {
			return Self::elem_wise_bin_nbb::<T>(dst, a_inp, b_inp, f);
		}
		if b_inp.len == 1 {
			return Self::elem_wise_bin_nnb::<T>(dst, a_inp, b_inp, f);
		}
		if a_inp.len == 1 {
			if C == Commutative {
				return Self::elem_wise_bin_nnb::<T>(dst, b_inp, a_inp, f);
			} else {
				return Self::elem_wise_bin_nbn::<T>(dst, a_inp, b_inp, f);
			}
		}
		Self::elem_wise_bin_nnn::<T>(dst, a_inp, b_inp, f);
	}

	fn elem_wise_bin_nnn<T: Copy>(
		dst: &TypedSliceSet<'_, CPUBuffer>, a_inp: &TypedSliceSet<'_, CPUBuffer>,
		b_inp: &TypedSliceSet<'_, CPUBuffer>, mut f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		let len = dst.len;
		assert!(a_inp.len == len);
		assert!(b_inp.len == len);
		Self::array_wise::<T, 3>([dst, a_inp, b_inp], |[dst_arr, a_arr, b_arr]| {
			for (d, (a, b)) in dst_arr.iter().zip(a_arr.iter().zip(b_arr)) {
				let val = f(d, a.get(), b.get());
				d.set(val);
			}
		});
	}

	fn elem_wise_bin_nnb<T: Copy>(
		dst: &TypedSliceSet<'_, CPUBuffer>, a_inp: &TypedSliceSet<'_, CPUBuffer>,
		b_inp: &TypedSliceSet<'_, CPUBuffer>, mut f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		let len = dst.len;
		assert!(a_inp.len == len);
		assert!(b_inp.len == 1);
		Self::array_wise::<T, 3>([dst, a_inp, b_inp], |[dst_arr, a_arr, b_arr]| {
			let b = b_arr[0].get();
			for (d, a) in dst_arr.iter().zip(a_arr) {
				let val = f(d, a.get(), b);
				d.set(val);
			}
		});
	}

	fn elem_wise_bin_nbn<T: Copy>(
		dst: &TypedSliceSet<'_, CPUBuffer>, a_inp: &TypedSliceSet<'_, CPUBuffer>,
		b_inp: &TypedSliceSet<'_, CPUBuffer>, mut f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		let len = dst.len;
		assert!(a_inp.len == 1);
		assert!(b_inp.len == len);
		Self::array_wise::<T, 3>([dst, a_inp, b_inp], |[dst_arr, a_arr, b_arr]| {
			let a = a_arr[0].get();
			for (d, b) in dst_arr.iter().zip(b_arr) {
				let val = f(d, a, b.get());
				d.set(val);
			}
		});
	}

	fn elem_wise_bin_nbb<T: Copy>(
		dst: &TypedSliceSet<'_, CPUBuffer>, a_inp: &TypedSliceSet<'_, CPUBuffer>,
		b_inp: &TypedSliceSet<'_, CPUBuffer>, mut f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		assert!(a_inp.len == 1);
		assert!(b_inp.len == 1);
		Self::array_wise::<T, 3>([dst, a_inp, b_inp], |[dst_arr, a_arr, b_arr]| {
			let a = a_arr[0].get();
			let b = b_arr[0].get();
			for d in dst_arr {
				let val = f(d, a, b);
				d.set(val);
			}
		});
	}

	#[inline(never)]
	fn softmax_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, inp: &TypedSliceSet<'_, CPUBuffer>,
	) {
		assert!(dst.len == inp.len);

		for [dst_arr, inp_arr] in BatchIter::<T, 2>::new([dst, inp]) {
			let max: f64 = inp_arr.iter().map(|x| x.get().to_f64()).fold(f64::MIN, f64::max);

			let mut sum = 0.0;
			for (d, i) in dst_arr.iter().zip(inp_arr) {
				let val = i.get().to_f64();
				let val = val - max;
				let e = val.exp();
				d.set(T::from_f64(e));

				sum += e;
			}

			// NOTE:
			// Subtracting max in the loop above ensures at least one of the exponents
			// is `exp(max - max) == 1.0`. So sum will be >= 1.0 and division by zero
			// is impossible.
			// This could only fail if all inputs are `-inf` or at least one input is `+inf`.
			// In that case, `sum == nan` and so all outputs will be `nan`.
			let sum_recip = 1.0 / sum;

			for d in dst_arr.iter() {
				let val = d.get().to_f64() * sum_recip;
				d.set(T::from_f64(val));
			}
		}
	}

	#[inline(never)]
	fn rms_norm_f<T: Copy + FromToF64>(
		dst: &TypedSliceSet<'_, CPUBuffer>, inp: &TypedSliceSet<'_, CPUBuffer>, eps: f64,
		scale_storage: Option<&TypedSliceSet<'_, CPUBuffer>>,
	) {
		assert!(dst.len == inp.len);
		let len = dst.len;
		let len_recip = 1.0 / (len as f64);

		if let Some(scale_storage) = scale_storage {
			assert!(scale_storage.len == 1);
			for [dst_arr, inp_arr, sc] in BatchIter::<T, 3>::new([dst, inp, scale_storage]) {
				let scale = math::rsqrt(math::dot(inp_arr, inp_arr) * len_recip + eps);
				sc[0].set(T::from_f64(scale));
				for (d, i) in dst_arr.iter().zip(inp_arr) {
					let val = i.get().to_f64() * scale;
					d.set(T::from_f64(val));
				}
			}
		} else {
			for [dst_arr, inp_arr] in BatchIter::<T, 2>::new([dst, inp]) {
				let scale = math::rsqrt(math::dot(inp_arr, inp_arr) * len_recip + eps);
				for (d, i) in dst_arr.iter().zip(inp_arr) {
					let val = i.get().to_f64() * scale;
					d.set(T::from_f64(val));
				}
			}
		}
	}

	#[inline(never)]
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
			for row in 0..m {
				for col in 0..n {
					let mut sum = 0.0;
					for i in 0..k {
						let a_index = tensor_size_to_usize(row * a.row_stride + i * a.col_stride);
						let a_cell = unsafe { a_arr.get_unchecked(a_index) };
						let a_val = a_cell.get().to_f64();

						let b_index = tensor_size_to_usize(i * b.row_stride + col * b.col_stride);
						let b_cell = unsafe { b_arr.get_unchecked(b_index) };
						let b_val = b_cell.get().to_f64();

						sum += a_val * b_val;
					}
					let c_index = tensor_size_to_usize(row * c.row_stride + col * c.col_stride);
					let c_cell = unsafe { c_arr.get_unchecked(c_index) };
					let c_val = c_cell.get().to_f64();

					let new_val = c_val * dst_weight + sum * ab_weight;
					c_cell.set(new_val as f32);
				}
			}
		}
	}

	#[inline(never)]
	fn format_f<T: Copy + FromToF64>(
		&self, f: &mut fmt::Formatter, offset: TensorSize, len: TensorSize, stride: TensorSize,
	) -> fmt::Result {
		let slices = TypedSliceSet {
			buffer: self,
			dtype: DType::F32,
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

			let val = arr[0].get().to_f64();
			if val >= 0.0 {
				write!(f, " ")?;
			}
			write!(f, "{:.4}", val)?;
		}
		Ok(())
	}
}

impl Buffer for CPUBuffer {
	fn load_data(&self, dtype: DType, offset: TensorSize, len: TensorSize, src: &[u8]) {
		let begin = dtype.array_bytes(offset).and_then(|t| TensorSize::try_from(t).ok()).unwrap();
		let len = dtype.array_bytes(len).and_then(|t| TensorSize::try_from(t).ok()).unwrap();
		let end = tensor_size_to_usize(begin.checked_add(len).unwrap());

		assert!(end <= self.base.size_bytes);
		let dst = unsafe { self.cast::<u8>(begin, len) };

		assert!(dst.len() == src.len());
		for (d, s) in dst.iter().zip(src) {
			d.set(*s);
		}
	}

	fn zeros(&self, dst: &SliceSet) {
		let dst = self.cast_slices(dst);
		match dst.dtype {
			DType::F32 => Self::elem_wise::<f32, 1>([&dst], |[d]| d.set(0.0)),
			_ => todo!(),
		}
	}

	fn randn(&self, dst: &SliceSet) {
		let dst = self.cast_slices(dst);
		let mut rng = self.device().rng.borrow_mut();
		match dst.dtype {
			DType::F32 => Self::elem_wise::<f32, 1>([&dst], |[d]| d.set(rng.get_normal() as f32)),
			_ => todo!(),
		}
	}

	fn copy(&self, dst: &SliceSet, src: &SliceSet) {
		let dst = self.cast_slices(dst);
		let src = self.cast_slices(src);
		match dst.dtype {
			DType::F32 => Self::elem_wise::<f32, 2>([&dst, &src], |[d, s]| d.set(s.get())),
			_ => todo!(),
		}
	}

	fn acc(&self, dst: &SliceSet, dst_weight: f64, new: &SliceSet, new_weight: f64) {
		let dst = self.cast_slices(dst);
		let new = self.cast_slices(new);
		match dst.dtype {
			DType::F32 => Self::elem_wise::<f32, 2>([&dst, &new], |[d, n]| {
				let d_val = f64::from(d.get());
				let n_val = f64::from(n.get());
				let val = d_val * dst_weight + n_val * new_weight;
				d.set(val as f32)
			}),
			_ => todo!(),
		}
	}

	fn mul(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet) {
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		match dst.dtype {
			DType::F32 => Self::elem_wise_bin::<f32, Commutative>(&dst, &a, &b, |_, a, b| a * b),
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
			DType::F32 => Self::elem_wise_bin::<f32, Commutative>(&dst, &a, &b, |d, a, b| {
				let d = f64::from(d.get());
				let a = f64::from(a);
				let b = f64::from(b);
				(d * dst_weight + a * b * ab_weight) as f32
			}),
			_ => todo!(),
		}
	}

	fn sub(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet) {
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		match dst.dtype {
			DType::F32 => Self::elem_wise_bin::<f32, NonCommutative>(&dst, &a, &b, |_, a, b| a - b),
			_ => todo!(),
		}
	}

	fn add(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet) {
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		match dst.dtype {
			DType::F32 => Self::elem_wise_bin::<f32, Commutative>(&dst, &a, &b, |_, a, b| a + b),
			_ => todo!(),
		}
	}

	fn swiglu(&self, dst: &SliceSet, lin: &SliceSet, gate: &SliceSet) {
		let dst = self.cast_slices(dst);
		let lin = self.cast_slices(lin);
		let gate = self.cast_slices(gate);
		match dst.dtype {
			DType::F32 => Self::elem_wise::<f32, 3>([&dst, &lin, &gate], |[dst, lin, gate]| {
				let forward = math::swiglu(f64::from(lin.get()), f64::from(gate.get()));
				dst.set(forward as f32);
			}),
			_ => todo!(),
		}
	}

	fn swiglu_backward(
		&self, d_lin: &SliceSet, d_gate: &SliceSet, lin: &SliceSet, gate: &SliceSet,
		d_out: &SliceSet,
	) {
		let d_lin = self.cast_slices(d_lin);
		let d_gate = self.cast_slices(d_gate);
		let lin = self.cast_slices(lin);
		let gate = self.cast_slices(gate);
		let d_out = self.cast_slices(d_out);
		match d_lin.dtype {
			DType::F32 => Self::elem_wise::<f32, 5>(
				[&d_lin, &d_gate, &lin, &gate, &d_out],
				|[d_lin, d_gate, lin, gate, d_out]| {
					let lin = f64::from(lin.get());
					let gate = f64::from(gate.get());
					let d_out = f64::from(d_out.get());
					let (d_lin_val, d_gate_val) = math::swiglu_backward(lin, gate);
					let d_lin_val = d_lin_val * d_out;
					let d_gate_val = d_gate_val * d_out;
					d_lin.set(d_lin_val as f32);
					d_gate.set(d_gate_val as f32);
				},
			),
			_ => todo!(),
		}
	}

	fn dot(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet, ab_weight: f64) {
		assert!(a.len == b.len);
		assert!(dst.len == 1);
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		match dst.dtype {
			DType::F32 => Self::array_wise::<f32, 3>([&dst, &a, &b], |[dst, a, b]| {
				let val = math::dot(a, b) * ab_weight;
				let val = f32::from_f64(val);
				dst[0].set(val);
			}),
			_ => todo!(),
		}
	}

	fn dot_acc(&self, dst: &SliceSet, dst_weight: f64, a: &SliceSet, b: &SliceSet, ab_weight: f64) {
		assert!(a.len == b.len);
		assert!(dst.len == 1);
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		match dst.dtype {
			DType::F32 => Self::array_wise::<f32, 3>([&dst, &a, &b], |[dst, a, b]| {
				let old_val = f64::from(dst[0].get());
				let dot = math::dot(a, b);
				let val = dst_weight * old_val + ab_weight * dot;
				let val = f32::from_f64(val);
				dst[0].set(val);
			}),
			_ => todo!(),
		}
	}

	fn sum_all(&self, a: &SliceSet) -> f64 {
		let a = self.cast_slices(a);
		let mut sum = 0.0;
		match a.dtype {
			DType::F32 => Self::array_wise::<f32, 1>([&a], |[arr]| {
				sum += arr.iter().map(|x| x.get().to_f64()).sum::<f64>()
			}),
			_ => todo!(),
		}
		sum
	}

	fn approx_eq(&self, a: &SliceSet, b: &SliceSet, eps: f64) -> bool {
		let a = self.cast_slices(a);
		let b = self.cast_slices(b);
		let mut result = true;
		match a.dtype {
			DType::F32 => Self::elem_wise::<f32, 2>([&a, &b], |[a, b]| {
				let a_val = f64::from(a.get());
				let b_val = f64::from(b.get());
				result &= (a_val - b_val).abs() < eps;
			}),
			_ => todo!(),
		}
		result
	}

	fn rsqrt(&self, dst: &SliceSet, inp: &SliceSet, eps: f64) {
		let dst = self.cast_slices(dst);
		let inp = self.cast_slices(inp);
		assert!(dst.dtype == inp.dtype);
		match dst.dtype {
			DType::F32 => Self::elem_wise::<f32, 2>([&dst, &inp], |[d, i]| {
				let i = f64::from(i.get());
				d.set(math::rsqrt(i + eps) as f32);
			}),
			_ => todo!(),
		}
	}

	fn log_clamped(&self, dst: &SliceSet, a: &SliceSet) {
		let dst = self.cast_slices(dst);
		let a = self.cast_slices(a);
		assert!(dst.dtype == a.dtype);
		match dst.dtype {
			DType::F32 => Self::elem_wise::<f32, 2>([&dst, &a], |[d, a]| {
				let a = f64::from(a.get());
				d.set(a.ln().max(-1000.0) as f32);
			}),
			_ => todo!(),
		}
	}

	fn softmax(&self, dst: &SliceSet, inp: &SliceSet) {
		let dst = self.cast_slices(dst);
		let inp = self.cast_slices(inp);
		assert!(dst.dtype == inp.dtype);
		match dst.dtype {
			DType::F32 => Self::softmax_f::<f32>(&dst, &inp),
			_ => todo!(),
		}
	}

	fn rms_norm(&self, dst: &SliceSet, inp: &SliceSet, eps: f64, scale_storage: Option<&SliceSet>) {
		let dst = self.cast_slices(dst);
		let inp = self.cast_slices(inp);
		let scale_storage = scale_storage.map(|s| {
			assert!(dst.dtype == s.dtype);
			self.cast_slices(s)
		});
		assert!(dst.dtype == inp.dtype);
		match dst.dtype {
			DType::F32 => Self::rms_norm_f::<f32>(&dst, &inp, eps, scale_storage.as_ref()),
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
			DType::F32 => {
				Self::gemm_f32(&dst, dst_weight, &a, &b, ab_weight);
			},
			_ => todo!(),
		}
	}

	fn format(
		&self, f: &mut fmt::Formatter, dtype: DType, offset: TensorSize, len: TensorSize,
		stride: TensorSize,
	) -> fmt::Result {
		match dtype {
			DType::F32 => self.format_f::<f32>(f, offset, len, stride),
			_ => todo!(),
		}
	}
}
/*
#[cfg(test)]
mod tests {
	use std::num::NonZero;

	use super::*;

	#[test]
	fn test_math_dot() {
		let a = [Cell::new(-0.5656), Cell::new(0.2939), Cell::new(-0.1837)];
		let b = [Cell::new(0.7546), Cell::new(1.0750), Cell::new(1.0206)];
		let result = math::dot(&a, &b);
		assert_approx_eq!(result, -0.2983, 1e-4);
	}

	#[test]
	fn test_math_rsqrt() {
		let result = math::rsqrt(0.6387);
		assert_approx_eq!(result, 1.2513, 1e-4);
	}

	fn new_test_buffer_f32(size: TensorSize) -> Rc<CPUBuffer> {
		let dev = CPUDevice::new("CPU".to_string());
		let buf = dev.new_cpu_buffer(DType::F32, size);
		let data = unsafe { buf.cast::<f32>(0, size) };
		for i in 0..size {
			data[tensor_size_to_usize(i)].set((i + 1) as f32);
		}
		buf
	}

	fn new_default_slices(buf: &CPUBuffer) -> TypedSliceSet<'_, CPUBuffer> {
		TypedSliceSet {
			buffer: buf,
			dtype: DType::F32,
			offset: 13,
			len: 11,
			batch_size: 3,
			batch_stride: 17,
		}
	}

	fn in_default_slices(i: TensorSize) -> bool {
		(i >= 13 && i < 24) || (i >= 30 && i < 41) || (i >= 47 && i < 58)
	}

	fn default_value(i: TensorSize) -> f32 {
		(i + 1) as f32
	}

	#[test]
	fn test_batch_iter() {
		let buf = new_test_buffer_f32(100);
		let slices = new_default_slices(&buf);
		for [arr] in BatchIter::<f32, 1>::new([&slices]) {
			for i in arr {
				i.set(3.1415);
			}
		}
		let data = unsafe { buf.cast::<f32>(0, 100) };
		for i in 0..100 {
			let val = data[tensor_size_to_usize(i)].get();
			if in_default_slices(i) {
				assert_eq!(val, 3.1415);
			} else {
				assert_eq!(val, default_value(i));
			}
		}
	}

	#[test]
	fn test_zeros() {
		let buf = new_test_buffer_f32(100);
		let slices = new_default_slices(&buf);
		CPUBuffer::zeros::<f32>(&slices);
		let data = unsafe { buf.cast::<f32>(0, 100) };
		for i in 0..100 {
			let val = data[tensor_size_to_usize(i)].get();
			if in_default_slices(i) {
				assert_eq!(val, 0.0);
			} else {
				assert_eq!(val, default_value(i));
			}
		}
	}

	#[test]
	fn test_randn() {
		let buf = new_test_buffer_f32(100);
		let slices = new_default_slices(&buf);
		CPUBuffer::randn_f::<f32>(&slices);
		let data = unsafe { buf.cast::<f32>(0, 100) };
		for i in 0..100 {
			let val = data[tensor_size_to_usize(i)].get();
			if in_default_slices(i) {
				assert_ne!(val, default_value(i));
			} else {
				assert_eq!(val, default_value(i));
			}
		}
	}

	#[test]
	fn test_copy() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = new_default_slices(&a_buf);
		CPUBuffer::randn_f::<f32>(&a_slices);

		let b_buf = new_test_buffer_f32(100);
		let b_slices = new_default_slices(&b_buf);

		CPUBuffer::copy::<f32>(&b_slices, &a_slices);

		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };
		for i in 0..100 {
			assert_eq!(a_data[i].get(), b_data[i].get());
		}
	}

	#[test]
	fn test_acc() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = new_default_slices(&a_buf);
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = new_default_slices(&b_buf);
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			if in_default_slices(i) {
				a_data[tensor_size_to_usize(i)].set(333.3 + i as f32);
				b_data[tensor_size_to_usize(i)].set(222.2 + i as f32);
			}
		}

		CPUBuffer::acc_f::<f32>(&a_slices, 0.7, &b_slices, 1.3);

		for i in 0..100 {
			let a = a_data[tensor_size_to_usize(i)].get();
			let b = b_data[tensor_size_to_usize(i)].get();

			if in_default_slices(i) {
				assert_approx_eq!(a, 0.7 * (333.3 + i as f32) + 1.3 * (222.2 + i as f32), 1e-4);
				assert_eq!(b, 222.2 + i as f32);
			} else {
				assert_eq!(a, default_value(i));
				assert_eq!(b, default_value(i));
			}
		}
	}

	#[test]
	fn test_mul_nnn() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = new_default_slices(&a_buf);
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = new_default_slices(&b_buf);
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		let c_buf = new_test_buffer_f32(100);
		let c_slices = new_default_slices(&c_buf);
		let c_data = unsafe { c_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			if in_default_slices(i) {
				a_data[tensor_size_to_usize(i)].set(333.3 + i as f32);
				b_data[tensor_size_to_usize(i)].set(222.2 + i as f32);
				c_data[tensor_size_to_usize(i)].set(111.1 + i as f32);
			}
		}

		CPUBuffer::mul_nnn_f::<f32>(&c_slices, &a_slices, &b_slices);

		for i in 0..100 {
			let a = a_data[tensor_size_to_usize(i)].get();
			let b = b_data[tensor_size_to_usize(i)].get();
			let c = c_data[tensor_size_to_usize(i)].get();

			if in_default_slices(i) {
				assert_eq!(a, 333.3 + i as f32);
				assert_eq!(b, 222.2 + i as f32);
				assert_approx_eq!(c, (333.3 + i as f32) * (222.2 + i as f32), 1e-4);
			} else {
				assert_eq!(a, default_value(i));
				assert_eq!(b, default_value(i));
				assert_eq!(c, default_value(i));
			}
		}
	}

	// TODO - test_mul_nnb

	#[test]
	fn test_sub() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = new_default_slices(&a_buf);
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = new_default_slices(&b_buf);
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		let c_buf = new_test_buffer_f32(100);
		let c_slices = new_default_slices(&c_buf);
		let c_data = unsafe { c_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			if in_default_slices(i) {
				a_data[tensor_size_to_usize(i)].set(3.3 + i as f32);
				b_data[tensor_size_to_usize(i)].set(2.2 + i as f32);
				c_data[tensor_size_to_usize(i)].set(1.1 + i as f32);
			}
		}

		CPUBuffer::sub_nnn_f::<f32>(&c_slices, &a_slices, &b_slices);

		for i in 0..100 {
			let a = a_data[tensor_size_to_usize(i)].get();
			let b = b_data[tensor_size_to_usize(i)].get();
			let c = c_data[tensor_size_to_usize(i)].get();

			if in_default_slices(i) {
				assert_eq!(a, 3.3 + i as f32);
				assert_eq!(b, 2.2 + i as f32);
				assert_approx_eq!(c, (3.3 + i as f32) - (2.2 + i as f32), 1e-4);
			} else {
				assert_eq!(a, default_value(i));
				assert_eq!(b, default_value(i));
				assert_eq!(c, default_value(i));
			}
		}
	}

	#[test]
	fn test_mul_nnn_acc() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = new_default_slices(&a_buf);
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = new_default_slices(&b_buf);
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		let c_buf = new_test_buffer_f32(100);
		let c_slices = new_default_slices(&c_buf);
		let c_data = unsafe { c_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			if in_default_slices(i) {
				a_data[tensor_size_to_usize(i)].set(3.3 + i as f32);
				b_data[tensor_size_to_usize(i)].set(2.2 + i as f32);
				c_data[tensor_size_to_usize(i)].set(1.1 + i as f32);
			}
		}

		CPUBuffer::mul_nnn_acc_f::<f32>(&c_slices, 0.7, &a_slices, &b_slices, 1.3);

		for i in 0..100 {
			let a = a_data[tensor_size_to_usize(i)].get();
			let b = b_data[tensor_size_to_usize(i)].get();
			let c = c_data[tensor_size_to_usize(i)].get();

			if in_default_slices(i) {
				assert_eq!(a, 3.3 + i as f32);
				assert_eq!(b, 2.2 + i as f32);
				assert_approx_eq!(
					c as f64,
					0.7 * (1.1 + i as f64) + 1.3 * (3.3 + i as f64) * (2.2 + i as f64),
					1e-3
				);
			} else {
				assert_eq!(a, default_value(i));
				assert_eq!(b, default_value(i));
				assert_eq!(c, default_value(i));
			}
		}
	}

	// TODO - test_mul_nnb_acc

	#[test]
	fn test_dot() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = TypedSliceSet {
			buffer: a_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 4,
			batch_size: 3,
			batch_stride: 17,
		};
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = TypedSliceSet {
			buffer: b_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 4,
			batch_size: 3,
			batch_stride: 17,
		};
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		let c_buf = new_test_buffer_f32(100);
		let c_slices = TypedSliceSet {
			buffer: c_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 1,
			batch_size: 3,
			batch_stride: 17,
		};
		let c_data = unsafe { c_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			if (i >= 13 && i < 17) || (i >= 30 && i < 34) || (i >= 47 && i < 51) {
				a_data[i].set(3.3 + i as f32);
				b_data[i].set(2.2 + i as f32);
				c_data[i].set(1.1 + i as f32);
			}
		}

		CPUBuffer::dot_f::<f32>(&c_slices, &a_slices, &b_slices, 1.3);

		for i in 0..100 {
			let a = a_data[tensor_size_to_usize(i)].get();
			let b = b_data[tensor_size_to_usize(i)].get();
			let c = c_data[tensor_size_to_usize(i)].get();

			if (i >= 13 && i < 17) || (i >= 30 && i < 34) || (i >= 47 && i < 51) {
				assert_eq!(a, 3.3 + i as f32);
				assert_eq!(b, 2.2 + i as f32);
				if i == 13 || i == 30 || i == 47 {
					let i = tensor_size_to_usize(i);
					let val = math::dot(&a_data[i..i + 4], &b_data[i..i + 4]) * 1.3;
					assert_approx_eq!(c, val as f32, 1e-4);
				}
			} else {
				assert_eq!(a, default_value(i));
				assert_eq!(b, default_value(i));
				assert_eq!(c, default_value(i));
			}
		}
	}

	#[test]
	fn test_dot_acc() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = TypedSliceSet {
			buffer: a_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 4,
			batch_size: 3,
			batch_stride: 17,
		};
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = TypedSliceSet {
			buffer: b_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 4,
			batch_size: 3,
			batch_stride: 17,
		};
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		let c_buf = new_test_buffer_f32(100);
		let c_slices = TypedSliceSet {
			buffer: c_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 1,
			batch_size: 3,
			batch_stride: 17,
		};
		let c_data = unsafe { c_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			if (i >= 13 && i < 17) || (i >= 30 && i < 34) || (i >= 47 && i < 51) {
				a_data[i].set(3.3 + i as f32);
				b_data[i].set(2.2 + i as f32);
				c_data[i].set(1.1 + i as f32);
			}
		}

		CPUBuffer::dot_acc_f::<f32>(&c_slices, 0.7, &a_slices, &b_slices, 1.3);

		for i in 0..100 {
			let a = a_data[tensor_size_to_usize(i)].get();
			let b = b_data[tensor_size_to_usize(i)].get();
			let c = c_data[tensor_size_to_usize(i)].get();

			if (i >= 13 && i < 17) || (i >= 30 && i < 34) || (i >= 47 && i < 51) {
				assert_eq!(a, 3.3 + i as f32);
				assert_eq!(b, 2.2 + i as f32);
				if i == 13 || i == 30 || i == 47 {
					let i = tensor_size_to_usize(i);
					let val = math::dot(&a_data[i..i + 4], &b_data[i..i + 4]);
					assert_approx_eq!(c, 0.7 * (1.1 + i as f32) + 1.3 * val as f32, 1e-4);
				} else {
					assert_eq!(c, 1.1 + i as f32);
				}
			} else {
				assert_eq!(a, default_value(i));
				assert_eq!(b, default_value(i));
				assert_eq!(c, default_value(i));
			}
		}
	}

	// TODO - test sum_all_f()

	#[test]
	fn test_rsqrt() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = new_default_slices(&a_buf);
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = new_default_slices(&b_buf);
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			if in_default_slices(i) {
				a_data[tensor_size_to_usize(i)].set(3.3 + i as f32);
				b_data[tensor_size_to_usize(i)].set(1.1 + i as f32);
			}
		}

		CPUBuffer::rsqrt_f::<f32>(&b_slices, &a_slices, 0.01);

		for i in 0..100 {
			let a = a_data[tensor_size_to_usize(i)].get();
			let b = b_data[tensor_size_to_usize(i)].get();

			if in_default_slices(i) {
				assert_eq!(a, 3.3 + i as f32);
				assert_approx_eq!(b as f64, math::rsqrt(3.3 + i as f64 + 0.01), 1e-4);
			} else {
				assert_eq!(a, default_value(i));
				assert_eq!(b, default_value(i));
			}
		}
	}

	#[test]
	fn test_log_clamped() {
		let buf = new_test_buffer_f32(100);
		let a_slices = TypedSliceSet {
			buffer: buf.as_ref(),
			dtype: DType::F32,
			offset: 10,
			len: 4,
			batch_size: 1,
			batch_stride: 7,
		};
		let b_slices = TypedSliceSet {
			buffer: buf.as_ref(),
			dtype: DType::F32,
			offset: 20,
			len: 4,
			batch_size: 1,
			batch_stride: 7,
		};

		let data = unsafe { buf.cast::<f32>(0, 100) };

		// [-1.0000,  0.0000, 20.0855, 10.0000] -> [-1000, -1000, 3.0000, 2.3026]
		data[10].set(-1.0);
		data[11].set(0.0);
		data[12].set(20.0855);
		data[13].set(10.0);

		CPUBuffer::log_clamped_f::<f32>(&b_slices, &a_slices);

		assert_eq!(data[20].get(), -1000.0);
		assert_eq!(data[21].get(), -1000.0);
		assert_approx_eq!(data[22].get(), 3.0, 1e-4);
		assert_approx_eq!(data[23].get(), 2.3026, 1e-4);
	}

	#[test]
	fn test_softmax() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = TypedSliceSet {
			buffer: a_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 4,
			batch_size: 5,
			batch_stride: 7,
		};
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = TypedSliceSet {
			buffer: b_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 4,
			batch_size: 5,
			batch_stride: 7,
		};
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			a_data[i].set(3.3 + i as f32);
			b_data[i].set(2.2 + i as f32);
		}

		// [-0.6057,  0.7070, -0.1872,  1.9023] -> [0.0540, 0.2007, 0.0821, 0.6632]
		a_data[13].set(-0.6057);
		a_data[14].set(0.7070);
		a_data[15].set(-0.1872);
		a_data[16].set(1.9023);

		// [ 1.2575, -1.2767,  1.1971, -1.4403] -> [0.4789, 0.0380, 0.4509, 0.0323]
		a_data[20].set(1.2575);
		a_data[21].set(-1.2767);
		a_data[22].set(1.1971);
		a_data[23].set(-1.4403);

		// [ 1.2575,    -inf,  1.1971, -1.4403] -> [0.4978, 0.0000, 0.4687, 0.0335]
		a_data[27].set(1.2575);
		a_data[28].set(f32::NEG_INFINITY);
		a_data[29].set(1.1971);
		a_data[30].set(-1.4403);

		// [   -inf,    -inf,    -inf,    -inf] -> [    nan,   nan,    nan,    nan]
		a_data[34].set(f32::NEG_INFINITY);
		a_data[35].set(f32::NEG_INFINITY);
		a_data[36].set(f32::NEG_INFINITY);
		a_data[37].set(f32::NEG_INFINITY);

		// [ 1.2575,    +inf,  1.1971, -1.4403] -> [    nan,   nan,    nan,    nan]
		a_data[41].set(1.2575);
		a_data[42].set(f32::INFINITY);
		a_data[43].set(1.1971);
		a_data[44].set(-1.4403);

		CPUBuffer::softmax_f::<f32>(&b_slices, &a_slices);

		assert_approx_eq!(b_data[13].get(), 0.0540, 1e-4);
		assert_approx_eq!(b_data[14].get(), 0.2007, 1e-4);
		assert_approx_eq!(b_data[15].get(), 0.0821, 1e-4);
		assert_approx_eq!(b_data[16].get(), 0.6632, 1e-4);

		assert_approx_eq!(b_data[20].get(), 0.4789, 1e-4);
		assert_approx_eq!(b_data[21].get(), 0.0380, 1e-4);
		assert_approx_eq!(b_data[22].get(), 0.4509, 1e-4);
		assert_approx_eq!(b_data[23].get(), 0.0323, 1e-4);

		assert_approx_eq!(b_data[27].get(), 0.4978, 1e-4);
		assert_approx_eq!(b_data[28].get(), 0.0000, 1e-4);
		assert_approx_eq!(b_data[29].get(), 0.4687, 1e-4);
		assert_approx_eq!(b_data[30].get(), 0.0335, 1e-4);

		assert!(b_data[34].get().is_nan());
		assert!(b_data[35].get().is_nan());
		assert!(b_data[36].get().is_nan());
		assert!(b_data[37].get().is_nan());

		assert!(b_data[41].get().is_nan());
		assert!(b_data[42].get().is_nan());
		assert!(b_data[43].get().is_nan());
		assert!(b_data[44].get().is_nan());

		for i in 0..100 {
			let a = a_data[i].get();
			let b = b_data[i].get();

			if (i < 13 || i >= 17)
				&& (i < 20 || i >= 24)
				&& (i < 27 || i >= 34)
				&& (i < 34 || i >= 38)
				&& (i < 41 || i >= 45)
			{
				assert_eq!(a, 3.3 + i as f32);
				assert_eq!(b, 2.2 + i as f32);
			}
		}
	}

	#[test]
	fn test_rms_norm() {
		let a_buf = new_test_buffer_f32(100);
		let a_slices = TypedSliceSet {
			buffer: a_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 4,
			batch_size: 1,
			batch_stride: 7,
		};
		let a_data = unsafe { a_buf.cast::<f32>(0, 100) };

		let b_buf = new_test_buffer_f32(100);
		let b_slices = TypedSliceSet {
			buffer: b_buf.as_ref(),
			dtype: DType::F32,
			offset: 13,
			len: 4,
			batch_size: 1,
			batch_stride: 7,
		};
		let b_data = unsafe { b_buf.cast::<f32>(0, 100) };

		for i in 0..100 {
			a_data[i].set(3.3 + i as f32);
			b_data[i].set(2.2 + i as f32);
		}

		// [-0.9236, -0.4050,  1.8707,  0.6424] -> [-0.8320, -0.3648,  1.6852,  0.5787]
		a_data[13].set(-0.9236);
		a_data[14].set(-0.4050);
		a_data[15].set(1.8707);
		a_data[16].set(0.6424);

		CPUBuffer::rms_norm_f::<f32>(&b_slices, &a_slices, 1e-4, None);

		assert_approx_eq!(b_data[13].get(), -0.8320, 1e-4);
		assert_approx_eq!(b_data[14].get(), -0.3648, 1e-4);
		assert_approx_eq!(b_data[15].get(), 1.6852, 1e-4);
		assert_approx_eq!(b_data[16].get(), 0.5787, 1e-4);

		for i in 0..100 {
			let a = a_data[i].get();
			let b = b_data[i].get();

			if i < 13 || i >= 17 {
				assert_eq!(a, 3.3 + i as f32);
				assert_eq!(b, 2.2 + i as f32);
			}
		}
	}

	#[test]
	fn test_gemm() {
		let buf = new_test_buffer_f32(100);
		let a_slices = TypedSliceSet {
			buffer: buf.as_ref(),
			dtype: DType::F32,
			offset: 10,
			len: 15,
			batch_size: 1,
			batch_stride: 7,
		};
		let a_matrix = TypedMatrixSet {
			slice_set: a_slices,
			rows: NonZeroTensorSize::new(3).unwrap(),
			cols: NonZeroTensorSize::new(4).unwrap(),
			row_stride: 5,
			col_stride: 1,
		};

		let b_slices = TypedSliceSet {
			buffer: buf.as_ref(),
			dtype: DType::F32,
			offset: 40,
			len: 20,
			batch_size: 1,
			batch_stride: 7,
		};
		let b_matrix = TypedMatrixSet {
			slice_set: b_slices,
			rows: NonZeroTensorSize::new(4).unwrap(),
			cols: NonZeroTensorSize::new(2).unwrap(),
			row_stride: 4,
			col_stride: 1,
		};

		let c_slices = TypedSliceSet {
			buffer: buf.as_ref(),
			dtype: DType::F32,
			offset: 70,
			len: 6,
			batch_size: 1,
			batch_stride: 7,
		};
		let c_matrix = TypedMatrixSet {
			slice_set: c_slices,
			rows: NonZeroTensorSize::new(3).unwrap(),
			cols: NonZeroTensorSize::new(2).unwrap(),
			row_stride: 2,
			col_stride: 1,
		};

		let data = unsafe { buf.cast::<f32>(0, 100) };
		for i in 0..100 {
			data[i].set(3.3 + i as f32);
		}

		// a = [[-0.9723,  0.3464, -0.3898, -1.2662],
		//      [ 0.1080,  0.0155,  0.7527, -1.2998],
		//      [ 0.1093,  0.5383, -0.8293,  0.6001]]
		data[10].set(-0.9723);
		data[11].set(0.3464);
		data[12].set(-0.3898);
		data[13].set(-1.2662);

		data[15].set(0.1080);
		data[16].set(0.0155);
		data[17].set(0.7527);
		data[18].set(-1.2998);

		data[20].set(0.1093);
		data[21].set(0.5383);
		data[22].set(-0.8293);
		data[23].set(0.6001);

		// b = [[ 1.9844,  0.6664],
		//      [ 0.2676, -1.9024],
		//      [-1.1005, -1.4824],
		//      [-0.5377,  1.0839]]
		data[40].set(1.9844);
		data[41].set(0.6664);

		data[44].set(0.2676);
		data[45].set(-1.9024);

		data[48].set(-1.1005);
		data[49].set(-1.4824);

		data[52].set(-0.5377);
		data[53].set(1.0839);

		CPUBuffer::gemm_f32(&c_matrix, 0.7, &a_matrix, &b_matrix, 1.3);

		// c = [[-0.7269, -2.1015],
		//      [ 0.0890, -2.4822],
		//      [ 0.9509,  0.9286]]

		assert_approx_eq!(data[70].get(), 0.7 * (3.3 + 70.0) + 1.3 * -0.7269, 1e-4);
		assert_approx_eq!(data[71].get(), 0.7 * (3.3 + 71.0) + 1.3 * -2.1015, 1e-4);

		assert_approx_eq!(data[72].get(), 0.7 * (3.3 + 72.0) + 1.3 * 0.0890, 1e-4);
		assert_approx_eq!(data[73].get(), 0.7 * (3.3 + 73.0) + 1.3 * -2.4822, 1e-4);

		assert_approx_eq!(data[74].get(), 0.7 * (3.3 + 74.0) + 1.3 * 0.9509, 1e-4);
		assert_approx_eq!(data[75].get(), 0.7 * (3.3 + 75.0) + 1.3 * 0.9286, 1e-4);

		for i in 0..100 {
			if (i < 10 || i > 13)
				&& (i < 15 || i > 18)
				&& (i < 20 || i > 23)
				&& (i < 40 || i > 41)
				&& (i < 44 || i > 45)
				&& (i < 48 || i > 49)
				&& (i < 52 || i > 53)
				&& (i < 70 || i > 75)
			{
				assert_eq!(data[i].get(), 3.3 + i as f32);
			}
		}
	}
}
*/
