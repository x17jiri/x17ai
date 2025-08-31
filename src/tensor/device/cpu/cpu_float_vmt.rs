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
use crate::tensor::device::cpu::{CPUDevice, math};
use crate::tensor::device::kernel::expr::DynExpr;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::map::{Map, ND, Select};
use crate::tensor::{HasDType, TensorOpError, generic};
use crate::util::FromToF64;
use crate::util::mycell::{BorrowGuard, BorrowMutGuard};

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
					let mut max = f64::NEG_INFINITY;
					for k in 0..self.reduction_size {
						let value = self.eval_expr(a, j, i, k);
						if value > max {
							max = value;
						}
					}
					max
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
					Self::load_from_cpu_memory,
					Self::store_to_cpu_memory,
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

	pub fn view_contiguous<'t, 'buf, const N: usize>(
		tensor: &'t generic::Tensor<ND<N>, BorrowGuard<'buf, DeviceBuffer>>,
	) -> Result<generic::Tensor<&'t ND<N>, &'t [T]>, ErrPack<TensorOpError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[N - 1];
		if !feature_dim.is_contiguous() {
			return Err(TensorOpError::not_contiguous());
		}
		Ok(tensor.view()?)
	}

	pub fn view_contiguous_mut<'t, 'buf, const N: usize>(
		tensor: &'t mut generic::Tensor<ND<N>, BorrowMutGuard<'buf, DeviceBuffer>>,
	) -> Result<generic::Tensor<&'t ND<N>, &'t mut [T]>, ErrPack<TensorOpError>>
	where
		T: 'static,
	{
		tensor.ensure_safe()?;
		let feature_dim = tensor.map().dims[N - 1];
		if !feature_dim.is_contiguous() {
			return Err(TensorOpError::not_contiguous());
		}
		Ok(tensor.view_mut()?)
	}

	fn read_float<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		src: &generic::Tensor<ND<0>, BorrowGuard<'buf, DeviceBuffer>>,
	) -> Result<f64, ErrPack<TensorOpError>> {
		src.ensure_safe()?;
		let view = src.view::<T>()?;
		Ok(view[[]].to_f64())
	}

	fn load_from_cpu_memory<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		src: &[u8],
		dst: &mut generic::Tensor<ND<1>, BorrowMutGuard<'buf, DeviceBuffer>>,
	) -> Result<(), ErrPack<TensorOpError>> {
		let (map, buf) = Self::view_contiguous_mut(dst)?.into_parts();
		let dst_slice = &mut buf[map.span()];
		let src_slice = unsafe {
			std::slice::from_raw_parts(
				src.as_ptr().cast::<T>(),
				src.len() / std::mem::size_of::<T>(),
			)
		};
		if dst_slice.len() != src_slice.len() {
			todo!("better error");
		}
		dst_slice.copy_from_slice(src_slice);
		Ok(())
	}

	fn store_to_cpu_memory<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		src: &generic::Tensor<ND<1>, BorrowGuard<'buf, DeviceBuffer>>,
		dst: &mut [u8],
	) -> Result<(), ErrPack<TensorOpError>> {
		let (map, buf) = Self::view_contiguous(src)?.into_parts();
		let src_slice = &buf[map.span()];
		let dst_slice = unsafe {
			std::slice::from_raw_parts_mut(
				dst.as_mut_ptr().cast::<T>(),
				dst.len() / std::mem::size_of::<T>(),
			)
		};
		if dst_slice.len() != src_slice.len() {
			todo!("better error");
		}
		dst_slice.copy_from_slice(src_slice);
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
