//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::{Cell, RefCell};
use std::hint::cold_path;
use std::mem::ManuallyDrop;
use std::ops::{Range, RangeFull};
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::HasDType;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};

pub mod float_executor;
pub mod math;
pub mod rng;
pub mod zip;

use rng::Rng;

use crate::tensor::device::cpu::float_executor::FloatExecutor;
use crate::tensor::device::{DeviceBuffer, DeviceError};
use crate::tensor::{DType, Device};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub enum ViewError {
	InvalidDType,
	NotOnCPUDevice,
}

//--------------------------------------------------------------------------------------------------

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

//--------------------------------------------------------------------------------------------------

pub struct CPUDevice {
	pub name: String,
	pub rng: Rc<RefCell<Rng>>,
	pub f32_executor: FloatExecutor<f32>,
}

impl CPUDevice {
	pub fn new() -> Rc<Self> {
		Self::new_named("CPU".to_string())
	}

	pub fn new_named(name: String) -> Rc<Self> {
		let rng = Rc::new(RefCell::new(Rng::new_default()));
		let f32_rng = rng.clone();
		Rc::new(Self {
			name,
			rng,
			f32_executor: FloatExecutor::new(f32_rng),
		})
	}

	pub fn ensure_can_view<'a, T: HasDType>(buf: &DeviceBuffer) -> Result<(), ViewError> {
		if buf.dtype != T::dtype {
			cold_path();
			return Err(ViewError::InvalidDType);
		}
		debug_assert!(T::dtype.bytes() == std::mem::size_of::<T>());
		if !buf.device_is_cpu {
			cold_path();
			return Err(ViewError::NotOnCPUDevice);
		}
		Ok(())
	}

	/// Returns a slice view of the buffer on CPU device.
	///
	/// # Errors
	/// If the buffer's dtype does not match `T` or if the buffer is not on CPU device.
	pub fn view<'a, T: HasDType>(buf: &DeviceBufferRef<'a>) -> Result<&'a [T], ViewError> {
		Self::ensure_can_view::<T>(buf)?;
		let data = buf.device_data;
		let elems = buf.elems;
		Ok(unsafe { std::slice::from_raw_parts(data.cast(), elems) })
	}

	pub fn view_mut<'a, T: HasDType>(
		buf: &mut DeviceBufferRefMut<'a>,
	) -> Result<&'a mut [T], ViewError> {
		Self::ensure_can_view::<T>(buf)?;
		let data = buf.device_data;
		let elems = buf.elems;
		Ok(unsafe { std::slice::from_raw_parts_mut(data.cast(), elems) })
	}

	/*
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

	#[inline(never)]
	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<DeviceBuffer>, DeviceError> {
		let executor = match dtype {
			f32::dtype => &self.f32_executor,
			_ => {
				cold_path();
				return Err(DeviceError::UnsupportedDType);
			},
		};
		let align = dtype.bytes().min(1);
		let Some(size) = dtype.array_bytes(elems) else {
			cold_path();
			return Err(DeviceError::AllocationFailed);
		};
		let Ok(layout) = std::alloc::Layout::from_size_align(size, align) else {
			cold_path();
			return Err(DeviceError::AllocationFailed);
		};
		let memory = unsafe { std::alloc::alloc(layout) };
		let Some(memory) = NonNull::new(memory) else {
			cold_path();
			return Err(DeviceError::AllocationFailed);
		};
		Ok(Rc::new(DeviceBuffer {
			executor: NonNull::from(executor),
			dtype,
			elems,
			device_data: memory.as_ptr(),
			device: ManuallyDrop::new(self.clone()),
			device_is_cpu: true,
			borrow_count: Cell::new(0),
		}))
	}

	unsafe fn drop_buffer(self: Rc<Self>, dtype: DType, elems: usize, device_data: *mut u8) {
		let align = dtype.bytes().min(1);
		let size = dtype.array_bytes(elems).unwrap();
		let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
		unsafe { std::alloc::dealloc(device_data, layout) }
	}
}

#[cfg(false)]
impl Executor for CPUDevice {
	fn gemm(&self, dst: &MatrixSet, dst_weight: f64, a: &MatrixSet, b: &MatrixSet, ab_weight: f64) {
		match dst.slice_set.dtype {
			f32::dtype => self.gemm_f::<f32>(&dst, dst_weight, &a, &b, ab_weight),
			_ => todo!(),
		}
	}

	fn attention(
		&self,
		dst: &SliceSet,
		q: &SliceSet,
		k: &SliceSet,
		v: &SliceSet,
		params: &AttentionParams,
	) {
		match dst.dtype {
			f32::dtype => self.attention_f::<f32>(&dst, &q, &k, &v, params),
			_ => todo!(),
		}
	}

	fn format(
		&self,
		f: &mut std::fmt::Formatter,
		buffer: &Buffer,
		dtype: DType,
		offset: usize,
		len: usize,
		stride: usize,
	) -> std::fmt::Result {
		match dtype {
			f32::dtype => self.format_f::<f32>(f, buffer, offset, len, stride),
			_ => todo!(),
		}
	}
}
