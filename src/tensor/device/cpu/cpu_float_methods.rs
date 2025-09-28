//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;

use crate::ErrPack;
use crate::tensor::device::cpu::math::{Float, FromToF64};
use crate::tensor::device::kernel::expr::DynExpr;
use crate::tensor::device::kernel::runner::KernelData;
use crate::tensor::device::{KernelElemArg, KernelReduceArg, MatMulArgs};
use crate::tensor::{DType, HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::LossyInto;

pub use attention::attention;

mod attention;

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
						.buf
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
						.buf
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

pub unsafe fn read_float(
	buf: NonNull<u8>,
	dtype: DType,
	offset: usize,
) -> Result<f64, ErrPack<TensorOpError>> {
	match dtype {
		f32::dtype => unsafe {
			let ptr = buf.cast::<f32>().add(offset);
			Ok(ptr.read().to_f64())
		},
		f64::dtype => unsafe {
			let ptr = buf.cast::<f64>().add(offset);
			Ok(ptr.read().to_f64())
		},
		_ => {
			cold_path();
			Err(UnsupportedDTypeError.into())
		},
	}
}

pub unsafe fn write_float(
	buf: NonNull<u8>,
	dtype: DType,
	offset: usize,
	value: f64,
) -> Result<(), ErrPack<TensorOpError>> {
	match dtype {
		dtype if dtype == f32::dtype => unsafe {
			let ptr = buf.cast::<f32>().add(offset);
			ptr.write(f32::from_f64(value));
			Ok(())
		},
		dtype if dtype == f64::dtype => unsafe {
			let ptr = buf.cast::<f64>().add(offset);
			ptr.write(f64::from_f64(value));
			Ok(())
		},
		_ => {
			cold_path();
			Err(UnsupportedDTypeError.into())
		},
	}
}

pub unsafe fn mm<T: 'static + HasDType + Float>(
	args: &MatMulArgs,
	scale: f64,
) -> Result<(), ErrPack<TensorOpError>> {
	unsafe {
		for j in 0..args.o_rows {
			for i in 0..args.o_cols {
				let mut t = 0.0;
				for k in 0..args.a_cols {
					let a_offset = args.a_offset + j * args.a_row_stride + k * args.a_col_stride;
					let b_offset = args.b_offset + k * args.b_row_stride + i * args.b_col_stride;
					t += read_float(args.a_buf, args.a_dtype, a_offset)?
						* read_float(args.b_buf, args.b_dtype, b_offset)?;
				}
				let t = T::from_f64(t * scale);
				let o_offset = args.o_offset + j * args.o_row_stride + i * args.o_col_stride;
				write_float(args.o_buf, args.o_dtype, o_offset, t.to_f64())?;
			}
		}
	}
	Ok(())
}

#[allow(clippy::unnecessary_wraps)]
impl<T: 'static + HasDType + Float, U: 'static + HasDType + Float + From<T> + LossyInto<T>>
	CPUFloatVMT<T, U>
{
	unsafe fn run_kernel(
		_this: &DeviceBufferVMT,
		kernel_data: &KernelData,
		o: &KernelOutput,
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
			for j in 0..o.size[0] {
				for i in 0..o.size[1] {
					let o = o
						.buf
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
