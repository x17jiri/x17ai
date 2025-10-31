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
use crate::tensor::device::kernel::{DynExpr, DynExprKind, DynKernelCall};

//--------------------------------------------------------------------------------------------------

pub unsafe fn run_kernel(data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>> {
	let expr = data.generate_expr();
	let reduced_expr = expr.pre_reduce();
	let o = &data.output;
	let reduction_size = o.size[2];
	let o_width = if reduced_expr.is_some() && o.stride_bytes[2] == 0 { 1 } else { reduction_size };
	let eval_expr = EvalExpr { data };
	unsafe {
		for z in 0..o.size[0] {
			for y in 0..o.size[1] {
				let reduced_value = if let Some(reduced_expr) = reduced_expr {
					eval_expr.reduce_expr(reduced_expr, z, y, reduction_size)?
				} else {
					0.0
				};
				for x in 0..o_width {
					let value = eval_expr.eval_expr(expr.as_ref(), z, y, x, reduced_value)?;
					CPUDevice::__write_float(
						o.buf,
						data.output_dtype(),
						o.offset_bytes
							+ z * o.stride_bytes[0]
							+ y * o.stride_bytes[1]
							+ x * o.stride_bytes[2],
						value,
					)?;
				}
			}
		}
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------

pub struct EvalExpr<'a> {
	data: &'a DynKernelCall<'a>,
}

impl<'a> EvalExpr<'a> {
	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::indexing_slicing)]
	pub unsafe fn reduce_expr(
		&self,
		expr: &DynExpr,
		z: usize,
		y: usize,
		width: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		unsafe {
			match &expr.kind {
				DynExprKind::SumExpr(a) => {
					let a = a.as_ref();
					let mut sum = KahanAcc::<f64>::new();
					for x in 0..width {
						let value = self.eval_expr(a, z, y, x, 0.0)?;
						sum.acc_(value);
					}
					Ok(sum.value())
				},
				DynExprKind::MaxExpr(a) => {
					let a = a.as_ref();
					let mut max = f64::NEG_INFINITY;
					for x in 0..width {
						let value = self.eval_expr(a, z, y, x, 0.0)?;
						if value > max {
							max = value;
						}
					}
					Ok(max)
				},

				DynExprKind::ElemwiseTensorArg(..)
				| DynExprKind::ReduceTensorArg(..)
				| DynExprKind::ScalarArg(..)
				| DynExprKind::NegExpr(..)
				| DynExprKind::ExpExpr(..)
				| DynExprKind::AbsExpr(..)
				| DynExprKind::SqrtExpr(..)
				| DynExprKind::LnExpr(..)
				| DynExprKind::AddExpr(..)
				| DynExprKind::SubExpr(..)
				| DynExprKind::MulExpr(..)
				| DynExprKind::RecipExpr(..) => {
					panic!("Not a reduction expression");
				},
			}
		}
	}

	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::indexing_slicing)]
	pub unsafe fn eval_expr(
		&self,
		expr: &DynExpr,
		z: usize,
		y: usize,
		x: usize,
		reduced_value: f64,
	) -> Result<f64, ErrPack<TensorOpError>> {
		unsafe {
			match &expr.kind {
				&DynExprKind::ElemwiseTensorArg(index) => {
					let elemwise_arg = &self.data.tensor_args[index];
					let dtype = self.data.arg_dtype(index);
					CPUDevice::__read_float(
						elemwise_arg.buf,
						dtype,
						elemwise_arg.offset_bytes
							+ z * elemwise_arg.stride_bytes[0]
							+ y * elemwise_arg.stride_bytes[1]
							+ x * elemwise_arg.stride_bytes[2],
					)
				},
				&DynExprKind::ReduceTensorArg(index) => {
					let index = self.data.tensor_args.len() - self.data.reduce_count + index;
					let reduce_arg = &self.data.tensor_args[index];
					let dtype = self.data.arg_dtype(index);
					CPUDevice::__read_float(
						reduce_arg.buf,
						dtype,
						reduce_arg.offset_bytes
							+ z * reduce_arg.stride_bytes[0]
							+ y * reduce_arg.stride_bytes[1]
							+ x * reduce_arg.stride_bytes[2],
					)
				},
				&DynExprKind::ScalarArg(index) => Ok(self.data.scalar_args[index]),

				DynExprKind::SumExpr(..) | DynExprKind::MaxExpr(..) => Ok(reduced_value),

				DynExprKind::NegExpr(a) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					Ok(-a)
				},
				DynExprKind::ExpExpr(a) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					Ok(a.exp())
				},
				DynExprKind::AbsExpr(a) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					Ok(a.abs())
				},
				DynExprKind::SqrtExpr(a) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					Ok(a.sqrt())
				},
				DynExprKind::LnExpr(a) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					Ok(a.ln().max(-1000.0))
				},
				DynExprKind::AddExpr(a, b) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					let b = self.eval_expr(b, z, y, x, reduced_value)?;
					Ok(a + b)
				},
				DynExprKind::SubExpr(a, b) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					let b = self.eval_expr(b, z, y, x, reduced_value)?;
					Ok(a - b)
				},
				DynExprKind::MulExpr(a, b) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					let b = self.eval_expr(b, z, y, x, reduced_value)?;
					Ok(a * b)
				},
				DynExprKind::RecipExpr(a) => {
					let a = self.eval_expr(a, z, y, x, reduced_value)?;
					Ok(1.0 / a)
				},
			}
		}
	}
}

//--------------------------------------------------------------------------------------------------
