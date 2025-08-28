//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::device::DeviceBuffer;
use crate::tensor::device::buffer::{
	DeviceBufferVMT, KernelElemArg, KernelOutput, KernelReduceArg,
};
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::kernel::expr::DynExpr;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::map::{IndexToOffset, Map, ND, Select};
use crate::tensor::{HasDType, TensorOpError, generic};
use crate::util::mycell::{BorrowGuard, BorrowMutGuard};

use super::math::{self, FromToF64};

//--------------------------------------------------------------------------------------------------

/*trait Slice2D<T> {
	fn slice(&self, dim0: usize, dim1: std::ops::RangeFull) -> &[T];
}

impl<T: Copy + HasDType + FromToF64> Slice2D<T> for generic::Tensor<ND<2>, &[T]> {
	fn slice(&self, dim0: usize, _dim1: std::ops::RangeFull) -> &[T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		self.buf().get(span).unwrap()
	}
}*/

trait Slice3D<T> {
	fn slice(&self, dim0: usize, dim1: usize, dim2: std::ops::RangeFull) -> &[T];
}

impl<T: Copy + HasDType + FromToF64> Slice3D<T> for generic::Tensor<ND<3>, &[T]> {
	fn slice(&self, dim0: usize, dim1: usize, _dim2: std::ops::RangeFull) -> &[T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let map = map.select(0, dim1).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		self.buf().get(span).unwrap()
	}
}

trait SliceMut2D<T> {
	fn slice_mut(&mut self, dim0: usize, dim1: std::ops::RangeFull) -> &mut [T];
}

impl<T: Copy + HasDType + FromToF64> SliceMut2D<T> for generic::Tensor<ND<2>, &mut [T]> {
	fn slice_mut(&mut self, dim0: usize, _dim1: std::ops::RangeFull) -> &mut [T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		unsafe { self.buf_mut() }.get_mut(span).unwrap()
	}
}

trait SliceMut3D<T> {
	fn slice_mut(&mut self, dim0: usize, dim1: usize, dim2: std::ops::RangeFull) -> &mut [T];
}

impl<T: Copy + HasDType + FromToF64> SliceMut3D<T> for generic::Tensor<ND<3>, &mut [T]> {
	fn slice_mut(&mut self, dim0: usize, dim1: usize, _dim2: std::ops::RangeFull) -> &mut [T] {
		let map = self.map();
		let map = map.select(0, dim0).unwrap();
		let map = map.select(0, dim1).unwrap();
		let span = map.span();
		assert!(map.dims[0].size == span.len());
		unsafe { self.buf_mut() }.get_mut(span).unwrap()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct EvalExpr<'a, T: 'static + Copy + HasDType + FromToF64> {
	elemwise_args: &'a [KernelElemArg],
	reduce_args: &'a [KernelReduceArg],
	scalar_args: &'a [f64],
	reduction_size: usize,
	device: &'a CPUDevice,
	phantom: std::marker::PhantomData<T>,
}

impl<'a, T: 'static + Copy + HasDType + FromToF64> EvalExpr<'a, T> {
	pub unsafe fn eval_expr(&self, expr: &DynExpr, j: usize, i: usize, k: usize) -> f64 {
		unsafe {
			match expr {
				DynExpr::ElemwiseTensorArg(index) => {
					assert!(k == 0);
					let elemwise_arg = &self.elemwise_args[*index];
					elemwise_arg
						.device_data
						.add(
							elemwise_arg.offset_bytes
								+ j * elemwise_arg.stride_bytes[0]
								+ i * elemwise_arg.stride_bytes[1],
						)
						.cast::<T>()
						.read()
						.to_f64()
				},
				DynExpr::ReduceTensorArg(index) => {
					let reduce_arg = &self.reduce_args[*index];
					assert!(k < self.reduction_size);
					reduce_arg
						.device_data
						.add(
							reduce_arg.offset_bytes
								+ j * reduce_arg.stride_bytes[0]
								+ i * reduce_arg.stride_bytes[1]
								+ k * reduce_arg.stride_bytes[2],
						)
						.cast::<T>()
						.read()
						.to_f64()
				},
				DynExpr::ScalarArg(index) => self.scalar_args[*index],

				DynExpr::RandnExpr() => {
					let mut rng = self.device.rng.borrow_mut();
					let val = rng.get_normal_clamped();
					val
				},

				DynExpr::SumExpr(a) => {
					assert!(!self.reduce_args.is_empty());
					let a = a.as_ref();
					let mut sum = 0.0;
					for k in 0..self.reduction_size {
						let value = self.eval_expr(a, j, i, k);
						sum += value;
					}
					sum
				},
				DynExpr::MaxExpr(a) => {
					assert!(!self.reduce_args.is_empty());
					let a = a.as_ref();
					(0..self.reduction_size)
						.map(|k| self.eval_expr(a, j, i, k))
						.fold(f64::NEG_INFINITY, f64::max)
				},

				DynExpr::NegExpr(a) => {
					let a = self.eval_expr(a, j, i, k);
					-a
				},
				DynExpr::ExpExpr(a) => {
					let a = self.eval_expr(a, j, i, k);
					a.exp()
				},
				DynExpr::AbsExpr(a) => {
					let a = self.eval_expr(a, j, i, k);
					a.abs()
				},
				DynExpr::SqrtExpr(a) => {
					let a = self.eval_expr(a, j, i, k);
					a.sqrt()
				},
				DynExpr::LnExpr(a) => {
					let a = self.eval_expr(a, j, i, k);
					a.ln().max(-1000.0)
				},
				DynExpr::AddExpr(a, b) => {
					let a = self.eval_expr(a, j, i, k);
					let b = self.eval_expr(b, j, i, k);
					a + b
				},
				DynExpr::SubExpr(a, b) => {
					let a = self.eval_expr(a, j, i, k);
					let b = self.eval_expr(b, j, i, k);
					a - b
				},
				DynExpr::MulExpr(a, b) => {
					let a = self.eval_expr(a, j, i, k);
					let b = self.eval_expr(b, j, i, k);
					a * b
				},
				DynExpr::RecipExpr(a) => {
					let a = self.eval_expr(a, j, i, k);
					1.0 / a
				},
			}
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub(super) struct CPUFloatVMT<T: Copy + HasDType + FromToF64> {
	vmt: DeviceBufferVMT,
	phantom: std::marker::PhantomData<T>,
}

impl<T: 'static + Copy + HasDType + FromToF64> CPUFloatVMT<T> {
	pub fn new(device: &Rc<MaybeUninit<CPUDevice>>, kernel_runner: Rc<KernelRunner>) -> Self {
		let device = device.as_ptr();
		let device = unsafe { NonNull::new_unchecked(device.cast_mut()) };
		let device_is_cpu = true;
		let dtype = T::dtype;
		Self {
			vmt: unsafe {
				DeviceBufferVMT::new(
					device,
					device_is_cpu,
					dtype,
					kernel_runner,
					CPUDevice::drop_buffer,
					Self::read_float,
					Self::load_bin,
					Self::store_bin,
					Self::mm,
					Self::attention,
					Self::run_kernel,
				)
			},
			phantom: std::marker::PhantomData,
		}
	}

	unsafe fn cast_this<'a>(vmt: NonNull<DeviceBufferVMT>) -> &'a Self {
		debug_assert!(std::mem::offset_of!(Self, vmt) == 0);
		let vmt = vmt.cast::<Self>();
		unsafe { &*vmt.as_ptr() }
	}

	fn device(&self) -> &CPUDevice {
		let (device, _) = self.vmt.device_ptr().to_raw_parts();
		let cpu_device = device.cast::<CPUDevice>();
		unsafe { cpu_device.as_ref() }
	}

	pub fn view_contiguous<'t, 'buf>(
		tensor: &'t generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
	) -> Result<generic::Tensor<&'t ND<2>, &'t [T]>, ErrPack<TensorOpError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		if !feature_dim.is_contiguous() {
			return Err(TensorOpError::not_contiguous());
		}
		Ok(tensor.view()?)
	}

	pub fn view_contiguous_mut<'t, 'buf>(
		tensor: &'t mut generic::Tensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
	) -> Result<generic::Tensor<&'t ND<2>, &'t mut [T]>, ErrPack<TensorOpError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[1];
		if !feature_dim.is_contiguous() {
			return Err(TensorOpError::not_contiguous());
		}
		Ok(tensor.view_mut()?)
	}

	pub fn attention_tile<const FIRST: bool>(
		acc: &mut generic::Tensor<ND<3>, &mut [f64]>, // [output, head, vo_feature]

		q: &generic::Tensor<ND<3>, &[T]>, // [output, head, qk_feature]
		k: &generic::Tensor<ND<3>, &[T]>, // [input, head, qk_feature]
		v: &generic::Tensor<ND<3>, &[T]>, // [input, head, vo_feature]

		// `acc`, `prev_max` and `prev_sum` will be initialized when processing the first tile.
		prev_max: &mut generic::Tensor<ND<2>, &mut [f64]>, // [output, head]
		prev_sum: &mut generic::Tensor<ND<2>, &mut [f64]>, // [output, head]

		// Scratch space for storing scores. It doesn't need to be initialized.
		// On GPU, its shape will be [output, head, input]. However, we process outputs
		// sequentially, so we don't need separate space for each output.
		scores: &mut generic::Tensor<ND<2>, &mut [f64]>, // [head, input]
	) {
		let O = q.size(0).unwrap();
		let I = k.size(0).unwrap();
		let H = q.size(1).unwrap();
		let VO = v.size(2).unwrap();
		for j in 0..O {
			for i in 0..I {
				for h in 0..H {
					let q = q.slice(j, h, ..);
					let k = k.slice(i, h, ..);
					scores[[h, i]] = math::dot(q, k);
					// scores[h][i].set(math::dot(q, k))
				}
			}
			if FIRST {
				for h in 0..H {
					let prev_max = &mut prev_max[[j, h]];
					let prev_sum = &mut prev_sum[[j, h]];

					let scores = scores.slice_mut(h, ..);
					let scores = &mut scores[..I]; // TODO
					let (first_max, first_sum) = math::softmax_part1_(scores);

					*prev_max = first_max;
					*prev_sum = first_sum;

					let acc = acc.slice_mut(j, h, ..);
					for i in 0..1 {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64();
						for f in 0..VO {
							acc[f] = score * v[f].to_f64();
						}
					}
					for i in 1..I {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64();
						for f in 0..VO {
							acc[f] += score * v[f].to_f64();
						}
					}
				}
			} else {
				for h in 0..H {
					let prev_max = &mut prev_max[[j, h]];
					let prev_sum = &mut prev_sum[[j, h]];

					let scores = scores.slice_mut(h, ..);
					let scores = &mut scores[..I]; // TODO
					let (new_max, new_sum) = math::softmax_part1_(scores);

					let total_max = prev_max.max(new_max);
					*prev_max = total_max;

					let prev_weight = (*prev_max - total_max).exp();
					let new_weight = (new_max - total_max).exp();

					*prev_sum = (*prev_sum * prev_weight) + (new_sum * new_weight);

					let acc = acc.slice_mut(j, h, ..);
					for i in 0..1 {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64() * new_weight;
						for f in 0..VO {
							acc[f] = (acc[f] * prev_weight) + (score * v[f].to_f64());
						}
					}
					for i in 1..I {
						let v = v.slice(i, h, ..);
						let score = scores[i].to_f64() * new_weight;
						for f in 0..VO {
							acc[f] += score * v[f].to_f64();
						}
					}
				}
			}
		}
	}

	fn read_float<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		src: &generic::Tensor<ND<0>, BorrowGuard<'buf, DeviceBuffer>>,
	) -> Result<f64, ErrPack<TensorOpError>> {
		src.ensure_safe()?;
		let view = src.view::<T>()?;
		Ok(view[[]].to_f64())
	}

	fn load_bin<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		dst: &mut generic::Tensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
		src: &mut dyn std::io::Read,
	) -> Result<(), ErrPack<TensorOpError>> {
		let (map, buf) = Self::view_contiguous_mut(dst)?.into_parts();
		for j in 0..map.dims[0].size {
			let b = map.index_to_offset([j, 0]).unwrap();
			let e = b + map.dims[1].size;
			let buf = &mut buf[b..e];
			let byte_slice = unsafe {
				std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), size_of_val(buf))
			};
			src.read_exact(byte_slice)?;
		}
		Ok(())
	}

	fn store_bin<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		src: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		dst: &mut dyn std::io::Write,
	) -> Result<(), ErrPack<TensorOpError>> {
		let (map, buf) = Self::view_contiguous(src)?.into_parts();
		for j in 0..map.dims[0].size {
			let b = map.index_to_offset([j, 0]).unwrap();
			let e = b + map.dims[1].size;
			let buf = &buf[b..e];
			let byte_slice = unsafe {
				std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), std::mem::size_of_val(buf))
			};
			dst.write_all(byte_slice)?;
		}
		Ok(())
	}

	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::many_single_char_names)]
	fn mm<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		o: &mut generic::Tensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
		a: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		b: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		scale: f64,
	) -> Result<(), ErrPack<TensorOpError>> {
		let m = o.map().dims[0].size;
		let n = o.map().dims[1].size;
		let k = a.map().dims[1].size;

		assert!(a.map().dims[0].size == m);
		assert!(b.map().dims[0].size == k);
		assert!(b.map().dims[1].size == n);

		o.ensure_safe()?;
		let o_row_stride = o.map().dims[0].stride;
		let o_col_stride = o.map().dims[1].stride;
		let mut o = o.view_mut::<T>()?;
		let o_off = o.map().offset;
		let o = unsafe { &mut o.buf_mut()[o_off..] };

		a.ensure_safe()?;
		let a_row_stride = a.map().dims[0].stride;
		let a_col_stride = a.map().dims[1].stride;
		let a = a.view::<T>()?;
		let a = &a.buf()[a.map().offset..];

		b.ensure_safe()?;
		let b_row_stride = b.map().dims[0].stride;
		let b_col_stride = b.map().dims[1].stride;
		let b = b.view::<T>()?;
		let b = &b.buf()[b.map().offset..];

		for j in 0..m {
			for i in 0..n {
				let mut t = 0.0;
				for k in 0..k {
					let a = a[j * a_row_stride + k * a_col_stride];
					let b = b[k * b_row_stride + i * b_col_stride];
					t += a.to_f64() * b.to_f64();
				}
				let t = T::from_f64(t * scale);
				o[j * o_row_stride + i * o_col_stride] = t;
			}
		}

		Ok(()) // TODO
	}

	fn attention<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		// [inputs, qo_heads, o_features]
		_o: &mut generic::Tensor<ND<3>, BorrowMutGuard<'buf, DeviceBuffer>>,
		// [inputs, qo_heads, qk_features]
		_q: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>,
		// [inputs, k_heads, qk_features]
		_k: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>,
		// [inputs, v_heads, vo_features]
		_v: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>,
	) {
		todo!("CPUFloatExecutor::attention is not implemented yet");
	}

	unsafe fn run_kernel(
		this: NonNull<DeviceBufferVMT>,
		kernel_data: &KernelData,
		o: *const KernelOutput,
		elemwise_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		scalar_args: *const f64,
		reduction_size: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		let this = unsafe { Self::cast_this(this) };
		let expr = kernel_data.expr.as_ref();
		unsafe {
			let eval_expr = EvalExpr::<T> {
				elemwise_args: std::slice::from_raw_parts(
					elemwise_args,
					kernel_data.elemwise_count,
				),
				reduce_args: std::slice::from_raw_parts(reduce_args, kernel_data.reduce_count),
				scalar_args: std::slice::from_raw_parts(scalar_args, kernel_data.scalar_count),
				reduction_size,
				device: this.device(),
				phantom: std::marker::PhantomData,
			};
			let o = &*o;
			for j in 0..o.size[0] {
				for i in 0..o.size[1] {
					let o = o
						.device_data
						.add(o.offset_bytes + j * o.stride_bytes[0] + i * o.stride_bytes[1])
						.cast::<T>();
					let v = eval_expr.eval_expr(expr, j, i, 0);
					o.write(T::from_f64(v));
				}
			}
		}
		Ok(())
	}
}
