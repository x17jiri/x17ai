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
	AttentionArgs, DeviceBufferVMT, KernelElemArg, KernelOutput, KernelReduceArg, MatMulArgs,
};
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::cpu::math::Float;
use crate::tensor::device::kernel::expr::DynExpr;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::GenericTensor;
use crate::tensor::generic::map::ND;
use crate::tensor::{HasDType, TensorOpError};
use crate::util::LossyInto;
use crate::util::mycell::{BorrowGuard, BorrowMutGuard};

//--------------------------------------------------------------------------------------------------

pub struct EvalExpr<
	'a,
	T: 'static + HasDType + Float,
	U: 'static + HasDType + Float + From<T> + LossyInto<T>,
> {
	elemwise_args: &'a [KernelElemArg],
	reduce_args: &'a [KernelReduceArg],
	scalar_args: &'a [f64],
	reduction_size: usize,
	phantom_t: std::marker::PhantomData<T>,
	phantom_u: std::marker::PhantomData<U>,
}

impl<'a, T: 'static + HasDType + Float, U: 'static + HasDType + Float + From<T> + LossyInto<T>>
	EvalExpr<'a, T, U>
{
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
pub(super) struct CPUFloatVMT<
	T: 'static + HasDType + Float,
	U: 'static + HasDType + Float + From<T> + LossyInto<T>,
> {
	vmt: DeviceBufferVMT,
	phantom_t: std::marker::PhantomData<T>,
	phantom_u: std::marker::PhantomData<U>,
}

#[allow(clippy::unnecessary_wraps)]
impl<T: 'static + HasDType + Float, U: 'static + HasDType + Float + From<T> + LossyInto<T>>
	CPUFloatVMT<T, U>
{
	pub fn new(device: &Rc<MaybeUninit<CPUDevice>>, kernel_runner: Rc<KernelRunner>) -> Self {
		let device = device.as_ptr();
		let device = unsafe { NonNull::new_unchecked(device.cast_mut()) };
		let device_is_cpu = true;
		Self {
			vmt: unsafe {
				DeviceBufferVMT::new(
					device,
					device_is_cpu,
					T::dtype,
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
			phantom_t: std::marker::PhantomData,
			phantom_u: std::marker::PhantomData,
		}
	}

	unsafe fn read_float(
		_this: NonNull<DeviceBufferVMT>,
		dev_src: (ND<0>, &DeviceBuffer),
	) -> Result<f64, ErrPack<TensorOpError>> {
		let (map, buf) = dev_src;
		debug_assert!(buf.vmt().device_is_cpu());
		debug_assert!(buf.vmt().dtype() == T::dtype);
		let mem = buf.device_data().cast::<T>();
		let val = unsafe { mem.add(map.offset).read() };
		Ok(val.to_f64())
	}

	fn load_from_cpu_memory(
		_this: NonNull<DeviceBufferVMT>,
		cpu_src: NonNull<u8>,
		dev_dst: (ND<0>, &DeviceBuffer),
		count: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		let (map, buf) = dev_dst;
		let buf = unsafe {
			std::slice::from_raw_parts_mut(buf.device_data().as_ptr().cast::<T>(), buf.elems())
		};
		let dst_slice = &mut buf[map.offset..map.offset + count];
		let src_slice = unsafe { std::slice::from_raw_parts(cpu_src.as_ptr().cast::<T>(), count) };
		dst_slice.copy_from_slice(src_slice);
		Ok(())
	}

	fn store_to_cpu_memory(
		_this: NonNull<DeviceBufferVMT>,
		dev_src: (ND<0>, &DeviceBuffer),
		cpu_dst: NonNull<u8>,
		count: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		let (map, buf) = dev_src;
		let buf = unsafe {
			std::slice::from_raw_parts(buf.device_data().as_ptr().cast::<T>(), buf.elems())
		};
		let src_slice = &buf[map.offset..map.offset + count];
		let dst_slice =
			unsafe { std::slice::from_raw_parts_mut(cpu_dst.as_ptr().cast::<T>(), count) };
		dst_slice.copy_from_slice(src_slice);
		Ok(())
	}

	unsafe fn mm(
		_this: NonNull<DeviceBufferVMT>,
		args: &MatMulArgs,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			let a = args.a_buf.cast::<T>().add(args.a_offset);
			let b = args.b_buf.cast::<T>().add(args.b_offset);
			let o = args.o_buf.cast::<T>().add(args.o_offset);
			for j in 0..args.o_rows {
				for i in 0..args.o_cols {
					let mut t = 0.0;
					for k in 0..args.a_cols {
						let a = a.add(j * args.a_row_stride + k * args.a_col_stride).read();
						let b = b.add(k * args.b_row_stride + i * args.b_col_stride).read();
						t += a.to_f64() * b.to_f64();
					}
					let t = T::from_f64(t * args.scale);
					o.add(j * args.o_row_stride + i * args.o_col_stride).write(t);
				}
			}
		}
		Ok(()) // TODO
	}

	fn attention(
		_this: NonNull<DeviceBufferVMT>,
		args: &AttentionArgs,
	) -> Result<(), ErrPack<TensorOpError>> {
		super::attention::attention::<T, U>(args)
	}

	unsafe fn run_kernel(
		_this: NonNull<DeviceBufferVMT>,
		kernel_data: &KernelData,
		o: *const KernelOutput,
		elemwise_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		scalar_args: *const f64,
		reduction_size: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		let expr = kernel_data.expr.as_ref();
		unsafe {
			let eval_expr = EvalExpr::<T, U> {
				elemwise_args: std::slice::from_raw_parts(
					elemwise_args,
					kernel_data.elemwise_count,
				),
				reduce_args: std::slice::from_raw_parts(reduce_args, kernel_data.reduce_count),
				scalar_args: std::slice::from_raw_parts(scalar_args, kernel_data.scalar_count),
				reduction_size,
				phantom_t: std::marker::PhantomData,
				phantom_u: std::marker::PhantomData,
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
