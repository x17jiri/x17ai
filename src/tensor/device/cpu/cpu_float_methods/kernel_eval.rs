//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::cpu::cpu_float_methods::KahanAcc;
use crate::tensor::device::kernel::expr::DynExpr;
use crate::tensor::device::kernel::runner::KernelData;
use crate::tensor::device::{KernelElemArg, KernelOutput, KernelReduceArg};

//--------------------------------------------------------------------------------------------------

pub unsafe fn run_kernel(
	kernel_data: &KernelData,
	o: &KernelOutput,
	elemwise_args: &[KernelElemArg],
	reduce_args: &[KernelReduceArg],
	scalar_args: &[f64],
	reduction_size: usize,
	dtype_config: &[u64],
) -> Result<(), ErrPack<TensorOpError>> {
	let expr = kernel_data.expr.as_ref();
	unsafe {
		let eval_expr = EvalExpr {
			kernel_data,
			elemwise_args,
			reduce_args,
			scalar_args,
			reduction_size,
			dtype_config,
		};
		for j in 0..o.size[0] {
			for i in 0..o.size[1] {
				let value = eval_expr.eval_expr(expr, j, i, 0)?;
				CPUDevice::__write_float(
					o.buf,
					kernel_data.output_dtype(dtype_config),
					o.offset_bytes + j * o.stride_bytes[0] + i * o.stride_bytes[1],
					value,
				)?;
			}
		}
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------

pub struct EvalExpr<'a> {
	kernel_data: &'a KernelData,
	elemwise_args: &'a [KernelElemArg],
	reduce_args: &'a [KernelReduceArg],
	scalar_args: &'a [f64],
	reduction_size: usize,
	dtype_config: &'a [u64],
}

impl<'a> EvalExpr<'a> {
	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::indexing_slicing)]
	pub unsafe fn eval_expr(
		&self,
		expr: &DynExpr,
		j: usize,
		i: usize,
		k: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		unsafe {
			match expr {
				DynExpr::ElemwiseTensorArg(index) => {
					assert!(k == 0);
					let elemwise_arg = &self.elemwise_args[*index];
					let dtype = self.kernel_data.elemwise_dtype(self.dtype_config, *index);
					CPUDevice::__read_float(
						elemwise_arg.buf,
						dtype,
						elemwise_arg.offset_bytes
							+ j * elemwise_arg.stride_bytes[0]
							+ i * elemwise_arg.stride_bytes[1],
					)
				},
				DynExpr::ReduceTensorArg(index) => {
					let reduce_arg = &self.reduce_args[*index];
					let dtype = self.kernel_data.reduce_dtype(self.dtype_config, *index);
					assert!(k < self.reduction_size);
					CPUDevice::__read_float(
						reduce_arg.buf,
						dtype,
						reduce_arg.offset_bytes
							+ j * reduce_arg.stride_bytes[0]
							+ i * reduce_arg.stride_bytes[1]
							+ k * reduce_arg.stride_bytes[2],
					)
				},
				DynExpr::ScalarArg(index) => Ok(self.scalar_args[*index]),

				DynExpr::SumExpr(a) => {
					assert!(!self.reduce_args.is_empty());
					let a = a.as_ref();
					let mut sum = KahanAcc::<f64>::new();
					for k in 0..self.reduction_size {
						let value = self.eval_expr(a, j, i, k)?;
						sum.acc_(value);
					}
					Ok(sum.value())
				},
				DynExpr::MaxExpr(a) => {
					assert!(!self.reduce_args.is_empty());
					let a = a.as_ref();
					let mut max = f64::NEG_INFINITY;
					for k in 0..self.reduction_size {
						let value = self.eval_expr(a, j, i, k)?;
						if value > max {
							max = value;
						}
					}
					Ok(max)
				},

				DynExpr::NegExpr(a) => {
					let a = self.eval_expr(a, j, i, k)?;
					Ok(-a)
				},
				DynExpr::ExpExpr(a) => {
					let a = self.eval_expr(a, j, i, k)?;
					Ok(a.exp())
				},
				DynExpr::AbsExpr(a) => {
					let a = self.eval_expr(a, j, i, k)?;
					Ok(a.abs())
				},
				DynExpr::SqrtExpr(a) => {
					let a = self.eval_expr(a, j, i, k)?;
					Ok(a.sqrt())
				},
				DynExpr::LnExpr(a) => {
					let a = self.eval_expr(a, j, i, k)?;
					Ok(a.ln().max(-1000.0))
				},
				DynExpr::AddExpr(a, b) => {
					let a = self.eval_expr(a, j, i, k)?;
					let b = self.eval_expr(b, j, i, k)?;
					Ok(a + b)
				},
				DynExpr::SubExpr(a, b) => {
					let a = self.eval_expr(a, j, i, k)?;
					let b = self.eval_expr(b, j, i, k)?;
					Ok(a - b)
				},
				DynExpr::MulExpr(a, b) => {
					let a = self.eval_expr(a, j, i, k)?;
					let b = self.eval_expr(b, j, i, k)?;
					Ok(a * b)
				},
				DynExpr::RecipExpr(a) => {
					let a = self.eval_expr(a, j, i, k)?;
					Ok(1.0 / a)
				},
			}
		}
	}
}

//--------------------------------------------------------------------------------------------------
