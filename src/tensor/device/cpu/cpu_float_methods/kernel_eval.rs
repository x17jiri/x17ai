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
use crate::tensor::device::kernel::expr::{DynExpr, DynKernelCall};

//--------------------------------------------------------------------------------------------------

pub unsafe fn run_kernel(data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>> {
	let expr = data.generate_expr();
	unsafe {
		let eval_expr = EvalExpr { data };
		let o = data.output();
		for j in 0..o.size[0] {
			for i in 0..o.size[1] {
				let value = eval_expr.eval_expr(expr.as_ref(), j, i, 0)?;
				CPUDevice::__write_float(
					o.buf,
					data.output_dtype(),
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
	data: &'a DynKernelCall<'a>,
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
					let elemwise_arg = &self.data.elemwise_args()[*index];
					let dtype = self.data.elemwise_dtype(*index);
					CPUDevice::__read_float(
						elemwise_arg.buf,
						dtype,
						elemwise_arg.offset_bytes
							+ j * elemwise_arg.stride_bytes[0]
							+ i * elemwise_arg.stride_bytes[1],
					)
				},
				DynExpr::ReduceTensorArg(index) => {
					let reduce_arg = &self.data.reduce_args()[*index];
					let dtype = self.data.reduce_dtype(*index);
					assert!(k < self.data.output().reduction_size);
					CPUDevice::__read_float(
						reduce_arg.buf,
						dtype,
						reduce_arg.offset_bytes
							+ j * reduce_arg.stride_bytes[0]
							+ i * reduce_arg.stride_bytes[1]
							+ k * reduce_arg.stride_bytes[2],
					)
				},
				DynExpr::ScalarArg(index) => Ok(self.data.scalar_args()[*index]),

				DynExpr::SumExpr(a) => {
					assert!(!self.data.reduce_args().is_empty());
					let a = a.as_ref();
					let mut sum = KahanAcc::<f64>::new();
					for k in 0..self.data.output().reduction_size {
						let value = self.eval_expr(a, j, i, k)?;
						sum.acc_(value);
					}
					Ok(sum.value())
				},
				DynExpr::MaxExpr(a) => {
					assert!(!self.data.reduce_args().is_empty());
					let a = a.as_ref();
					let mut max = f64::NEG_INFINITY;
					for k in 0..self.data.output().reduction_size {
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
