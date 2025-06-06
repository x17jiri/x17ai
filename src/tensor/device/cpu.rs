//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::HasDType;
use crate::{Error, Result};

pub mod float_executor;
pub mod rng;
pub mod zip;

use rng::Rng;

#[cfg(false)]
mod math {
	use std::cell::Cell;
	use std::ops::{Range, RangeFull};

	pub fn dot<T: Copy + FromToF64>(a: &[Cell<T>], b: &[Cell<T>]) -> f64 {
		let res = a
			.iter()
			.zip(b)
			.map(|(a, b)| {
				let val = a.get().to_f64() * b.get().to_f64();
				//println!("dot: {} * {} = {}", a.get().to_f64(), b.get().to_f64(), val);
				val
			})
			.sum();
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

	pub fn softmax_part1<T: Copy + FromToF64, S: Copy + FromToF64>(
		inp: &[Cell<T>], scratch: &[Cell<S>],
	) -> (f64, f64) {
		// TODO
		// - calculating `max` is one loop
		// - calculating `sum` is another loop
		// - there are online algorithms for calculating `max` and `sum` simultaneously
		// - would they be worth it?

		let max: f64 = inp.iter().map(|x| x.get().to_f64()).fold(f64::MIN, f64::max);

		let mut sum = 0.0;
		for (i, s) in inp.iter().zip(scratch) {
			let val = i.get().to_f64();
			let val = val - max;
			let e = val.exp();
			s.set(S::from_f64(e));

			sum += e;
		}

		(max, sum)
	}

	pub fn softmax_part2<S: Copy + FromToF64, T: Copy + FromToF64>(
		scratch: &[Cell<S>], sum: f64, dst: &[Cell<T>],
	) {
		// NOTE:
		// Subtracting max in part1 ensures at least one of the exponents
		// is `exp(max - max) == 1.0`. So sum will be >= 1.0 and division by zero
		// is impossible.
		// This could only fail if all inputs are `-inf` or at least one input is `+inf`.
		// In that case, `sum == nan` and so all outputs will be `nan`.
		let sum_recip = 1.0 / sum;

		for (s, d) in scratch.iter().zip(dst) {
			let val = s.get().to_f64() * sum_recip;
			d.set(T::from_f64(val));
		}
	}

	pub fn softmax<T: Copy + FromToF64>(dst: &[Cell<T>], inp: &[Cell<T>]) {
		// use `dst` as scratch space between part1 and part2
		let scratch = dst;
		let (_, sum) = softmax_part1(inp, scratch);
		softmax_part2(scratch, sum, dst);
	}

	#[derive(Clone, Copy)]
	pub struct View2D<'a, T> {
		pub data: &'a [Cell<T>],
		pub cols: usize,
	}

	impl<'a, T> View2D<'a, T> {
		pub fn item(&self, row: usize, col: usize) -> &'a Cell<T> {
			debug_assert!(col < self.cols);
			let index = row * self.cols + col;
			&self.data[index]
		}

		pub fn slice(&self, head: usize, _: RangeFull) -> &'a [Cell<T>] {
			let begin = head * self.cols;
			let end = begin + self.cols;
			&self.data[begin..end]
		}
	}

	#[derive(Clone, Copy)]
	pub struct View3D<'a, T> {
		pub data: &'a [Cell<T>],
		pub seq_len: usize,
		pub seq_stride: usize,
		pub head_shift: usize,
		pub heads: usize,
		pub features: usize,
	}

	impl<'a, T> View3D<'a, T> {
		pub fn slice(&self, input: usize, head: usize, _: RangeFull) -> &'a [Cell<T>] {
			let head = head >> self.head_shift;
			let begin = input * self.seq_stride + head * self.features;
			let end = begin + self.features;
			&self.data[begin..end]
		}

		pub fn sub_sequence(&self, range: Range<usize>) -> Self {
			let data_begin = range.start * self.seq_stride;
			let data_end = range.end * self.seq_stride;
			let seq_len = range.end.saturating_sub(range.start);
			Self {
				data: &self.data[data_begin..data_end],
				seq_len,
				..*self
			}
		}
	}
}

use crate::tensor::device::DeviceBuffer;
use crate::tensor::device::cpu::float_executor::FloatExecutor;
use crate::tensor::{DType, Device};

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

#[derive(std::marker::ConstParamTy, PartialEq, Eq)]
enum Commutativity {
	Commutative,
	NonCommutative,
}

const Commutative: Commutativity = Commutativity::Commutative;
const NonCommutative: Commutativity = Commutativity::NonCommutative;

pub struct CPUDevice {
	name: String,
	rng: Rc<RefCell<Rng>>,
	f32_executor: FloatExecutor<f32>,
}

impl CPUDevice {
	pub fn new(name: String) -> Rc<Self> {
		let rng = Rc::new(RefCell::new(Rng::new_default()));
		let f32_rng = rng.clone();
		Rc::new(Self {
			name,
			rng,
			f32_executor: FloatExecutor::new(f32_rng),
		})
	}

	/// Returns a slice view of the buffer on CPU device.
	///
	/// # Errors
	/// If the buffer's dtype does not match `T` or if the buffer is not on CPU device.
	pub fn view<T: HasDType>(buf: &DeviceBuffer) -> Result<&[Cell<T>]> {
		if buf.dtype != T::dtype {
			#[cold]
			fn err_view_invalid_dtype(buf_dtype: DType, T_dtype: DType) -> Error {
				format!("Cannot view a tensor with dtype {buf_dtype:?} as {T_dtype:?}").into()
			}
			return Err(err_view_invalid_dtype(buf.dtype, T::dtype));
		}
		debug_assert!(T::dtype.bytes() == std::mem::size_of::<T>());
		if !buf.device_is_cpu {
			#[cold]
			fn err_view_not_cpu_buffer() -> Error {
				"buffer is not on CPU device".into()
			}
			return Err(err_view_not_cpu_buffer());
		}

		let elems = buf.elems;
		Ok(unsafe { std::slice::from_raw_parts(buf.device_data as *const Cell<T>, elems) })
	}
	/*
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
				math::softmax(dst_arr, inp_arr);
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

		fn attention_tile<T: Copy + FromToF64, const FIRST: bool>(
			acc: View3D<f64>, // [output, head, vo_feature]

			q: View3D<T>, // [output, head, qk_feature]
			k: View3D<T>, // [input, head, qk_feature]
			v: View3D<T>, // [input, head, vo_feature]

			// `acc`, `prev_m` and `prev_l` will be initialized when processing the first tile.
			prev_m: View2D<f64>, // [output, head]
			prev_l: View2D<f64>, // [output, head]

			// Scratch space for storing scores. It doesn't need to be initialized.
			// On GPU, its shape will be [output, head, input]. However, we process outputs
			// sequentially, so we don't need separate space for each output.
			scores: View2D<f64>, // [head, input]
		) {
			let O = q.seq_len;
			let I = k.seq_len;
			let H = q.heads;
			let VO = v.features;
			for j in 0..O {
				for i in 0..I {
					for h in 0..H {
						let q = q.slice(j, h, ..);
						let k = k.slice(i, h, ..);
						scores.item(h, i).set(math::dot(q, k));
						// scores[h][i].set(math::dot(q, k))
					}
				}
				if FIRST {
					for h in 0..H {
						let scores = scores.slice(h, ..);
						let scores = &scores[..I]; // TODO
						let (first_m, first_l) = math::softmax_part1(scores, scores);

						//let S: Vec<f64> = scores.iter().map(|s| s.get() / first_l).collect();
						//println!("j = {}, h = {}, scores = {:.4?}", j, h, S.as_slice());

						prev_m.item(j, h).set(first_m);
						prev_l.item(j, h).set(first_l);

						let acc = acc.slice(j, h, ..);
						for i in 0..1 {
							let v = v.slice(i, h, ..);
							let score = scores[i].get().to_f64();
							for f in 0..VO {
								let v = v[f].get().to_f64();
								acc[f].set(score * v);
							}
						}
						for i in 1..I {
							let v = v.slice(i, h, ..);
							let score = scores[i].get().to_f64();
							for f in 0..VO {
								let v = v[f].get().to_f64();
								acc[f].set(acc[f].get() + score * v);
							}
						}
					}
				} else {
					for h in 0..H {
						let scores = scores.slice(h, ..);
						let scores = &scores[..I]; // TODO
						let (new_m, new_l) = math::softmax_part1(scores, scores);

						//let S: Vec<f64> = scores.iter().map(|s| s.get() / new_l).collect();
						//println!("j = {}, h = {}, ..scores = {:.4?}", j, h, S.as_slice());

						let prev_m = prev_m.item(j, h);
						let m = new_m.max(prev_m.get());

						let prev_weight = (prev_m.get() - m).exp();
						let new_weight = (new_m - m).exp();
						prev_m.set(m);

						let prev_l = prev_l.item(j, h);
						prev_l.set(prev_l.get() * prev_weight + new_l * new_weight);

						let acc = acc.slice(j, h, ..);
						for i in 0..1 {
							let v = v.slice(i, h, ..);
							let score = scores[i].get().to_f64() * new_weight;
							for f in 0..VO {
								let v = v[f].get().to_f64();
								acc[f].set(acc[f].get() * prev_weight + score * v);
							}
						}
						for i in 1..I {
							let v = v.slice(i, h, ..);
							let score = scores[i].get().to_f64() * new_weight;
							for f in 0..VO {
								let v = v[f].get().to_f64();
								acc[f].set(acc[f].get() + score * v);
							}
						}
					}
				}
			}
		}

		fn attention_finish<T: Copy + FromToF64>(
			dst: View3D<T>,      // [output, head, vo_feature]
			acc: View3D<f64>,    // [output, head, vo_feature]
			prev_l: View2D<f64>, // [output, head]
		) {
			let O = dst.seq_len;
			let H = dst.heads;
			for j in 0..O {
				for h in 0..H {
					let norm = prev_l.item(j, h).get();
					let o_slice = acc.slice(j, h, ..);
					let dst_slice = dst.slice(j, h, ..);
					math::softmax_part2(o_slice, norm, dst_slice);
				}
			}
		}

		#[inline(never)]
		fn attention_f<T: Copy + HasDType + FromToF64>(
			&self, o: &SliceSet, q: &SliceSet, k: &SliceSet, v: &SliceSet, params: &AttentionParams,
		) {
			let o = self.cast_slice_set::<T>(o);
			let q = self.cast_slice_set::<T>(q);
			let k = self.cast_slice_set::<T>(k);
			let v = self.cast_slice_set::<T>(v);

			const Bq: usize = 64; // Number of outputs processed in one tile.
			const Bkv: usize = 13; // Number of inputs processed in one tile.
			let H = params.heads;

			let o_size = Bq * H * params.v_features;
			let prev_l_size = Bq * H;
			let prev_m_size = Bq * H;
			let scores_size = H * Bkv;

			let o_off = 0;
			let o_end = o_off + o_size;

			let prev_m_off = o_end;
			let prev_m_end = prev_m_off + prev_m_size;

			let prev_l_off = prev_m_end;
			let prev_l_end = prev_l_off + prev_l_size;

			let scores_off = prev_l_end;
			let scores_end = scores_off + scores_size;

			let mem_size = scores_end;

			let scratch_space = vec![Default::default(); mem_size];

			let acc = View3D {
				data: &scratch_space[o_off..o_end],
				seq_len: Bq,
				seq_stride: H * params.v_features,
				head_shift: 0,
				heads: H,
				features: params.v_features,
			};
			let prev_m = View2D {
				data: &scratch_space[prev_m_off..prev_m_end],
				cols: H,
			};
			let prev_l = View2D {
				data: &scratch_space[prev_l_off..prev_l_end],
				cols: H,
			};
			let scores = View2D {
				data: &scratch_space[scores_off..scores_end],
				cols: Bkv,
			};

			let q = View3D {
				data: q.buffer,
				seq_len: q.count,
				seq_stride: q.stride,
				head_shift: 0,
				heads: H,
				features: params.qk_features,
			};
			let k = View3D {
				data: k.buffer,
				seq_len: k.count,
				seq_stride: k.stride,
				head_shift: params.k_shift,
				heads: H >> params.k_shift,
				features: params.qk_features,
			};
			let v = View3D {
				data: v.buffer,
				seq_len: v.count,
				seq_stride: v.stride,
				head_shift: params.v_shift,
				heads: H >> params.v_shift,
				features: params.v_features,
			};
			let o = View3D {
				data: o.buffer,
				seq_len: o.count,
				seq_stride: o.stride,
				head_shift: 0,
				heads: H,
				features: params.v_features,
			};

			// TODO masking, scale

			let seq_len = o.seq_len;
			for j in (0..seq_len).step_by(Bq) {
				let je = (j + Bq).min(seq_len);
				let q = q.sub_sequence(j..je);

				for i in (0..seq_len).step_by(Bkv) {
					let ie = (i + Bkv).min(seq_len);
					let k = k.sub_sequence(i..ie);
					let v = v.sub_sequence(i..ie);

					// First tile will initialize `acc`, `prev_m`, `prev_l`
					if i == 0 {
						Self::attention_tile::<T, true>(acc, q, k, v, prev_m, prev_l, scores);
					} else {
						Self::attention_tile::<T, false>(acc, q, k, v, prev_m, prev_l, scores);
					}
				}

				let o = o.sub_sequence(j..je);
				Self::attention_finish::<T>(o, acc, prev_l);
			}
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
				write!(f, "{:.7}", val)?;
			}
			Ok(())
		}
	}*/
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn new_buffer(self: Rc<Self>, dtype: DType, elems: usize) -> Rc<DeviceBuffer> {
		let executor = match dtype {
			f32::dtype => &self.f32_executor,
			_ => todo!("Unsupported dtype for CPUDevice: {}", dtype),
		};
		let align = dtype.bytes().min(1);
		let size = dtype.array_bytes(elems).expect("Invalid size for CPUBuffer");
		let layout = std::alloc::Layout::from_size_align(size, align)
			.expect("Couldn't create layout for CPUBuffer");
		let memory = unsafe { std::alloc::alloc(layout) };
		let memory = NonNull::new(memory).expect("Failed to allocate memory for CPUBuffer");
		Rc::new(DeviceBuffer {
			executor: NonNull::from(executor),
			dtype,
			elems,
			device_data: memory.as_ptr(),
			device: ManuallyDrop::new(self.clone()),
			device_is_cpu: true,
		})
	}

	fn drop_buffer(self: Rc<Self>, dtype: DType, elems: usize, device_data: *mut u8) {
		let align = dtype.bytes().min(1);
		let size = dtype.array_bytes(elems).unwrap();
		let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
		unsafe { std::alloc::dealloc(device_data, layout) }
	}
}

#[cfg(false)]
impl Executor for CPUDevice {
	fn read_bin(&self, dst: &SliceSet, src: &mut dyn std::io::Read) -> std::io::Result<()> {
		let mut result = Ok(());
		match dst.dtype {
			f32::dtype => self.array_wise::<f32, 1>([dst], |[d]| {
				if result.is_ok() {
					let ptr = d.as_ptr() as *mut u8;
					let bytes = d.len() * std::mem::size_of::<f32>();
					let slice = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
					result = src.read_exact(slice);

					// We always store values as little-endian,
					// so conversion is needed for big-endian targets
					#[cfg(target_endian = "big")]
					{
						for elem in d {
							let le = elem.get();
							let bits = le.to_bits();
							let swapped = bits.swap_bytes();
							let be = f32::from_bits(swapped);
							elem.set(be);
						}
					}
				}
			}),
			_ => todo!(),
		}
		result
	}

	fn write_bin(&self, src: &SliceSet, dst: &mut dyn std::io::Write) -> std::io::Result<()> {
		#[cfg(target_endian = "big")]
		{
			todo!("Saving to binary file on big-endian targets is not implemented yet");
		}
		let mut result = Ok(());
		match src.dtype {
			f32::dtype => self.array_wise::<f32, 1>([src], |[s]| {
				if result.is_ok() {
					let ptr = s.as_ptr() as *const u8;
					let bytes = s.len() * std::mem::size_of::<f32>();
					let slice = unsafe { std::slice::from_raw_parts(ptr, bytes) };
					result = dst.write_all(slice);
				}
			}),
			_ => todo!(),
		}
		result
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
		match dst.dtype {
			f32::dtype => self.attention_f::<f32>(&dst, &q, &k, &v, params),
			_ => todo!(),
		}
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
