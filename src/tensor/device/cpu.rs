// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::rc::Rc;

mod rng;

use rng::Rng;

use crate::tensor::buffer::{Buffer, MatrixSet, SliceSet};
use crate::tensor::device::AttentionParams;
use crate::tensor::dtype::{DType, HasDType};

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

//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct CPUSliceSet<'a, T> {
	buffer: &'a [Cell<T>],
	len: usize,
	count: usize,
	stride: usize,
}

impl<'a, T> CPUSliceSet<'a, T> {
	unsafe fn get_unchecked(&self, i: usize) -> &'a [Cell<T>] {
		debug_assert!(i < self.count);
		let begin = i * self.stride;
		let end = begin + self.len;
		debug_assert!(end <= self.buffer.len());
		unsafe { self.buffer.get_unchecked(begin..end) }
	}
}

//--------------------------------------------------------------------------------------------------

type CPUBufferElement = u64;

//--------------------------------------------------------------------------------------------------

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

	fn cast_buffer<T: HasDType>(&self, buffer: &Buffer) -> &[Cell<T>] {
		assert!(buffer.is_on_device(self));
		debug_assert!(T::dtype.bytes() == std::mem::size_of::<T>());
		unsafe {
			let ptr = buffer.device_buffer.as_ptr() as *const Cell<T>;
			let bytes = buffer.size_bytes;
			let elems = bytes / std::mem::size_of::<T>();
			std::slice::from_raw_parts(ptr, elems)
		}
	}

	fn cast_slice_set<'a, T: HasDType>(&'a self, slice_set: &SliceSet<'a>) -> CPUSliceSet<'a, T> {
		let dtype = slice_set.dtype;
		assert_eq!(dtype, T::dtype);

		let buffer = self.cast_buffer::<T>(slice_set.buffer);
		CPUSliceSet {
			buffer: &buffer[slice_set.span()],
			len: slice_set.len,
			count: slice_set.count,
			stride: slice_set.stride,
		}
	}

	fn array_wise<'a, T: Copy + HasDType, const N: usize>(
		&self, slices: [&SliceSet<'a>; N], mut f: impl FnMut([&[Cell<T>]; N]),
	) {
		let slices = slices.map(|s| self.cast_slice_set::<T>(s));

		let count = slices.get(0).map_or(0, |s| s.count);
		assert!(slices.iter().all(|s| s.count == count));

		for i in 0..count {
			// SAFETY: When we create a `CPUSliceSet` in `CPUDevice::cast_slices()`,
			// we assert that the slice is in bounds.
			let arrays = slices.map(|s| unsafe { s.get_unchecked(i) });
			f(arrays);
		}
	}

	fn elem_wise<'a, T: Copy + HasDType, const N: usize>(
		&self, slices: [&SliceSet<'a>; N], mut f: impl FnMut([&Cell<T>; N]),
	) {
		let len = slices.get(0).map_or(0, |s| s.len);
		assert!(slices.iter().all(|i| i.len == len));

		self.array_wise::<T, N>(slices, |arrays| {
			for i in 0..len {
				f(arrays.map(|array| {
					debug_assert!(i < array.len());
					let element = unsafe { array.get_unchecked(i) };
					element
				}));
			}
		});
	}

	fn elem_wise_bin<'a, T: Copy + HasDType, const C: Commutativity>(
		&self, dst: &SliceSet<'a>, a_inp: &SliceSet<'a>, b_inp: &SliceSet<'a>,
		f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		if a_inp.len == 1 && b_inp.len == 1 {
			return self.elem_wise_bin_nbb::<T>(dst, a_inp, b_inp, f);
		}
		if b_inp.len == 1 {
			return self.elem_wise_bin_nnb::<T>(dst, a_inp, b_inp, f);
		}
		if a_inp.len == 1 {
			if C == Commutative {
				return self.elem_wise_bin_nnb::<T>(dst, b_inp, a_inp, f);
			} else {
				return self.elem_wise_bin_nbn::<T>(dst, a_inp, b_inp, f);
			}
		}
		self.elem_wise_bin_nnn::<T>(dst, a_inp, b_inp, f);
	}

	fn elem_wise_bin_nnn<'a, T: Copy + HasDType>(
		&self, dst: &SliceSet<'a>, a_inp: &SliceSet<'a>, b_inp: &SliceSet<'a>,
		mut f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		let len = dst.len;
		assert!(a_inp.len == len);
		assert!(b_inp.len == len);
		self.array_wise::<T, 3>([dst, a_inp, b_inp], |[dst_arr, a_arr, b_arr]| {
			for (d, (a, b)) in dst_arr.iter().zip(a_arr.iter().zip(b_arr)) {
				let val = f(d, a.get(), b.get());
				d.set(val);
			}
		});
	}

	fn elem_wise_bin_nnb<'a, T: Copy + HasDType>(
		&self, dst: &SliceSet<'a>, a_inp: &SliceSet<'a>, b_inp: &SliceSet<'a>,
		mut f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		let len = dst.len;
		assert!(a_inp.len == len);
		assert!(b_inp.len == 1);
		self.array_wise::<T, 3>([dst, a_inp, b_inp], |[dst_arr, a_arr, b_arr]| {
			let b = b_arr[0].get();
			for (d, a) in dst_arr.iter().zip(a_arr) {
				let val = f(d, a.get(), b);
				d.set(val);
			}
		});
	}

	fn elem_wise_bin_nbn<'a, T: Copy + HasDType>(
		&self, dst: &SliceSet<'a>, a_inp: &SliceSet<'a>, b_inp: &SliceSet<'a>,
		mut f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		let len = dst.len;
		assert!(a_inp.len == 1);
		assert!(b_inp.len == len);
		self.array_wise::<T, 3>([dst, a_inp, b_inp], |[dst_arr, a_arr, b_arr]| {
			let a = a_arr[0].get();
			for (d, b) in dst_arr.iter().zip(b_arr) {
				let val = f(d, a, b.get());
				d.set(val);
			}
		});
	}

	fn elem_wise_bin_nbb<'a, T: Copy + HasDType>(
		&self, dst: &SliceSet<'a>, a_inp: &SliceSet<'a>, b_inp: &SliceSet<'a>,
		mut f: impl FnMut(&Cell<T>, T, T) -> T,
	) {
		assert!(a_inp.len == 1);
		assert!(b_inp.len == 1);
		self.array_wise::<T, 3>([dst, a_inp, b_inp], |[dst_arr, a_arr, b_arr]| {
			let a = a_arr[0].get();
			let b = b_arr[0].get();
			for d in dst_arr {
				let val = f(d, a, b);
				d.set(val);
			}
		});
	}

	#[inline(never)]
	fn softmax<'a, T: Copy + HasDType + FromToF64>(&self, dst: &SliceSet<'a>, inp: &SliceSet<'a>) {
		assert!(dst.len == inp.len);
		self.array_wise::<T, 2>([dst, inp], |[dst_arr, inp_arr]| {
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
		});
	}

	#[inline(never)]
	fn rms_norm_f<'a, T: Copy + HasDType + FromToF64>(
		&self, dst: &SliceSet<'a>, inp: &SliceSet<'a>, eps: f64,
	) {
		let len = dst.len;
		let len_recip = 1.0 / (len as f64);
		assert!(inp.len == len);

		self.array_wise::<T, 2>([dst, inp], |[dst_arr, inp_arr]| {
			let scale = math::rsqrt(math::dot(inp_arr, inp_arr) * len_recip + eps);
			for (d, i) in dst_arr.iter().zip(inp_arr) {
				let val = i.get().to_f64() * scale;
				d.set(T::from_f64(val));
			}
		});
	}

	#[inline(never)]
	fn rms_norm_with_scale_storage_f<'a, T: Copy + HasDType + FromToF64>(
		&self, dst: &SliceSet<'a>, inp: &SliceSet<'a>, eps: f64, scale_storage: &SliceSet<'a>,
	) {
		let len = dst.len;
		let len_recip = 1.0 / (len as f64);
		assert!(inp.len == len);
		assert!(scale_storage.len == 1);

		self.array_wise::<T, 3>([dst, inp, scale_storage], |[dst_arr, inp_arr, sc]| {
			let scale = math::rsqrt(math::dot(inp_arr, inp_arr) * len_recip + eps);
			sc[0].set(T::from_f64(scale));
			for (d, i) in dst_arr.iter().zip(inp_arr) {
				let val = i.get().to_f64() * scale;
				d.set(T::from_f64(val));
			}
		});
	}

	#[inline(never)]
	fn gemm_f<'a, T: Copy + HasDType + FromToF64>(
		&self, c: &MatrixSet<'a>, dst_weight: f64, a: &MatrixSet<'a>, b: &MatrixSet<'a>,
		ab_weight: f64,
	) {
		let m = c.rows.get();
		let n = c.cols.get();
		let k = a.cols.get();

		assert!(a.rows.get() == m);
		assert!(b.cols.get() == n);
		assert!(b.rows.get() == k);

		self.array_wise::<T, 3>(
			[&c.slice_set, &a.slice_set, &b.slice_set],
			|[c_arr, a_arr, b_arr]| {
				for row in 0..m {
					for col in 0..n {
						let mut sum = 0.0;
						for i in 0..k {
							let a_index = row * a.row_stride + i * a.col_stride;
							let a_cell = unsafe { a_arr.get_unchecked(a_index) };
							let a_val = a_cell.get().to_f64();

							let b_index = i * b.row_stride + col * b.col_stride;
							let b_cell = unsafe { b_arr.get_unchecked(b_index) };
							let b_val = b_cell.get().to_f64();

							sum += a_val * b_val;
						}
						let c_index = row * c.row_stride + col * c.col_stride;
						let c_cell = unsafe { c_arr.get_unchecked(c_index) };
						let c_val = c_cell.get().to_f64();

						let new_val = c_val * dst_weight + sum * ab_weight;
						c_cell.set(T::from_f64(new_val));
					}
				}
			},
		);
	}

	#[inline(never)]
	fn format_f<T: Copy + HasDType + FromToF64>(
		&self, f: &mut std::fmt::Formatter, buffer: &Buffer, offset: usize, len: usize,
		stride: usize,
	) -> std::fmt::Result {
		let buffer = self.cast_buffer::<T>(buffer);
		let mut first_item = true;
		for i in 0..len {
			if !first_item {
				write!(f, ", ")?;
			}
			first_item = false;

			let val = buffer[offset + i * stride].get().to_f64();
			if val >= 0.0 {
				write!(f, " ")?;
			}
			write!(f, "{:.4}", val)?;
		}
		Ok(())
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

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: usize) -> Rc<Buffer> {
		let step_size = std::mem::size_of::<CPUBufferElement>();
		let size_bytes = dtype.array_bytes(elems).unwrap().next_multiple_of(step_size);
		let layout = std::alloc::Layout::from_size_align(size_bytes, step_size).unwrap();
		let memory = unsafe { std::alloc::alloc(layout) };
		let memory = NonNull::new(memory).expect("Failed to allocate memory for CPUBuffer");
		Rc::new(Buffer {
			device: ManuallyDrop::new(self.clone()),
			size_bytes,
			device_buffer: memory,
		})
	}

	fn drop_buffer(self: Rc<Self>, device_buffer: NonNull<u8>, size_bytes: usize) {
		let step_size = std::mem::size_of::<CPUBufferElement>();
		debug_assert!(size_bytes % step_size == 0);
		let layout = std::alloc::Layout::from_size_align(size_bytes, step_size).unwrap();
		unsafe { std::alloc::dealloc(device_buffer.as_ptr(), layout) }
	}

	fn load_data(&self, buffer: &Buffer, dtype: DType, offset: usize, len: usize, src: &[u8]) {
		let buffer = self.cast_buffer::<u8>(buffer);

		let begin = dtype.array_bytes(offset).unwrap();
		let len = dtype.array_bytes(len).unwrap();
		let end = begin.checked_add(len).unwrap();

		let dst = &buffer[begin..end];

		assert!(dst.len() == src.len());
		for (d, s) in dst.iter().zip(src) {
			d.set(*s);
		}
	}

	fn zeros(&self, dst: &SliceSet) {
		match dst.dtype {
			f32::dtype => self.elem_wise::<f32, 1>([dst], |[d]| d.set(0.0)),
			_ => todo!(),
		}
	}

	fn randn(&self, dst: &SliceSet) {
		let mut rng = self.rng.borrow_mut();
		match dst.dtype {
			f32::dtype => self.elem_wise::<f32, 1>([dst], |[d]| d.set(rng.get_normal() as f32)),
			_ => todo!(),
		}
	}

	fn copy(&self, dst: &SliceSet, src: &SliceSet) {
		match dst.dtype {
			f32::dtype => self.elem_wise::<f32, 2>([dst, src], |[d, s]| d.set(s.get())),
			_ => todo!(),
		}
	}

	fn acc(&self, dst: &SliceSet, dst_weight: f64, new: &SliceSet, new_weight: f64) {
		match dst.dtype {
			f32::dtype => self.elem_wise::<f32, 2>([dst, new], |[d, n]| {
				let d_val = f64::from(d.get());
				let n_val = f64::from(n.get());
				let val = d_val * dst_weight + n_val * new_weight;
				d.set(val as f32)
			}),
			_ => todo!(),
		}
	}

	fn mul(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet) {
		match dst.dtype {
			f32::dtype => self.elem_wise_bin::<f32, Commutative>(dst, a, b, |_, a, b| a * b),
			_ => todo!(),
		}
	}

	fn mul_acc(&self, dst: &SliceSet, dst_weight: f64, a: &SliceSet, b: &SliceSet, ab_weight: f64) {
		match dst.dtype {
			f32::dtype => self.elem_wise_bin::<f32, Commutative>(dst, a, b, |d, a, b| {
				let d = f64::from(d.get());
				let a = f64::from(a);
				let b = f64::from(b);
				(d * dst_weight + a * b * ab_weight) as f32
			}),
			_ => todo!(),
		}
	}

	fn sub(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet) {
		match dst.dtype {
			f32::dtype => self.elem_wise_bin::<f32, NonCommutative>(dst, a, b, |_, a, b| a - b),
			_ => todo!(),
		}
	}

	fn add(&self, dst: &SliceSet, a: &SliceSet, b: &SliceSet) {
		match dst.dtype {
			f32::dtype => self.elem_wise_bin::<f32, Commutative>(dst, a, b, |_, a, b| a + b),
			_ => todo!(),
		}
	}

	fn swiglu(&self, dst: &SliceSet, lin: &SliceSet, gate: &SliceSet) {
		match dst.dtype {
			f32::dtype => self.elem_wise::<f32, 3>([dst, lin, gate], |[dst, lin, gate]| {
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
		match d_lin.dtype {
			f32::dtype => self.elem_wise::<f32, 5>(
				[d_lin, d_gate, lin, gate, d_out],
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
		match dst.dtype {
			f32::dtype => self.array_wise::<f32, 3>([&dst, &a, &b], |[dst, a, b]| {
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
		match dst.dtype {
			f32::dtype => self.array_wise::<f32, 3>([&dst, &a, &b], |[dst, a, b]| {
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
		let mut sum = 0.0;
		match a.dtype {
			f32::dtype => self.array_wise::<f32, 1>([&a], |[arr]| {
				sum += arr.iter().map(|x| x.get().to_f64()).sum::<f64>()
			}),
			_ => todo!(),
		}
		sum
	}

	fn approx_eq(&self, a: &SliceSet, b: &SliceSet, eps: f64) -> bool {
		let mut result = true;
		match a.dtype {
			f32::dtype => self.elem_wise::<f32, 2>([a, b], |[a, b]| {
				let a_val = f64::from(a.get());
				let b_val = f64::from(b.get());
				result &= (a_val - b_val).abs() < eps;
			}),
			_ => todo!(),
		}
		result
	}

	fn rsqrt(&self, dst: &SliceSet, inp: &SliceSet, eps: f64) {
		match dst.dtype {
			f32::dtype => self.elem_wise::<f32, 2>([dst, inp], |[d, i]| {
				let i = f64::from(i.get());
				d.set(math::rsqrt(i + eps) as f32);
			}),
			_ => todo!(),
		}
	}

	fn log_clamped(&self, dst: &SliceSet, a: &SliceSet) {
		match dst.dtype {
			f32::dtype => self.elem_wise::<f32, 2>([dst, a], |[d, a]| {
				let a = f64::from(a.get());
				d.set(a.ln().max(-1000.0) as f32);
			}),
			_ => todo!(),
		}
	}

	fn softmax(&self, dst: &SliceSet, inp: &SliceSet) {
		match dst.dtype {
			f32::dtype => self.softmax::<f32>(dst, inp),
			_ => todo!(),
		}
	}

	fn rms_norm(&self, dst: &SliceSet, inp: &SliceSet, eps: f64, scale_storage: Option<&SliceSet>) {
		match dst.dtype {
			f32::dtype => {
				if let Some(scale_storage) = scale_storage {
					self.rms_norm_with_scale_storage_f::<f32>(&dst, &inp, eps, scale_storage);
				} else {
					self.rms_norm_f::<f32>(&dst, &inp, eps);
				}
			},
			_ => todo!(),
		}
	}

	fn gemm(&self, dst: &MatrixSet, dst_weight: f64, a: &MatrixSet, b: &MatrixSet, ab_weight: f64) {
		match dst.slice_set.dtype {
			f32::dtype => self.gemm_f::<f32>(&dst, dst_weight, &a, &b, ab_weight),
			_ => todo!(),
		}
	}

	fn attention(
		&self, dst: &SliceSet, q: &SliceSet, k: &SliceSet, v: &SliceSet, params: &AttentionParams,
	) {
		/*let dst = self.cast_slices(dst);
		let q = self.cast_slices(q);
		let k = self.cast_slices(k);
		let v = self.cast_slices(v);*/
		let _ = (dst, q, k, v, params);
		todo!()
	}

	fn format(
		&self, f: &mut std::fmt::Formatter, buffer: &Buffer, dtype: DType, offset: usize,
		len: usize, stride: usize,
	) -> std::fmt::Result {
		match dtype {
			f32::dtype => self.format_f::<f32>(f, buffer, offset, len, stride),
			_ => todo!(),
		}
	}
}
