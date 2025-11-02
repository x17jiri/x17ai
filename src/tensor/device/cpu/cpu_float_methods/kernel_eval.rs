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
use crate::tensor::device::kernel::{
	DynExpr, DynExprArgKind, DynExprBinaryKind, DynExprKind, DynExprReduction,
	DynExprReductionKind, DynExprUnaryKind, DynKernelCall,
};

//--------------------------------------------------------------------------------------------------

/// # Safety
///
/// TODO
pub unsafe fn run_kernel(data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>> {
	let expr = data.generate_expr();
	let reduction = expr.find_reduction();
	let o = &data.output;
	let reduction_size = o.size[2];
	let o_width = if reduction.is_some() && o.stride_bytes[2] == 0 { 1 } else { reduction_size };
	let eval_expr = EvalExpr { data };
	unsafe {
		for z in 0..o.size[0] {
			for y in 0..o.size[1] {
				let reduced_value = if let Some(reduction) = reduction {
					eval_expr.reduce_expr(reduction, z, y, reduction_size)?
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
		reduction_expr: &DynExprReduction,
		z: usize,
		y: usize,
		width: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		let a = reduction_expr.expr.as_ref();
		unsafe {
			match &reduction_expr.kind {
				DynExprReductionKind::Sum => {
					let mut sum = KahanAcc::<f64>::new();
					for x in 0..width {
						let value = self.eval_expr(a, z, y, x, 0.0)?;
						sum.acc_(value);
					}
					Ok(sum.value())
				},
				DynExprReductionKind::Max => {
					let mut max = f64::NEG_INFINITY;
					for x in 0..width {
						let value = self.eval_expr(a, z, y, x, 0.0)?;
						if value > max {
							max = value;
						}
					}
					Ok(max)
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
				DynExprKind::Arg(a) => match a.kind {
					DynExprArgKind::ElemwiseTensor => {
						let elemwise_arg = &self.data.tensor_args[a.index];
						let dtype = self.data.arg_dtype(a.index);
						CPUDevice::__read_float(
							elemwise_arg.buf,
							dtype,
							elemwise_arg.offset_bytes
								+ z * elemwise_arg.stride_bytes[0]
								+ y * elemwise_arg.stride_bytes[1]
								+ x * elemwise_arg.stride_bytes[2],
						)
					},
					DynExprArgKind::ReduceTensor => {
						let index = self.data.tensor_args.len() - self.data.reduce_count + a.index;
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
					DynExprArgKind::Scalar => Ok(self.data.scalar_args[a.index]),
				},

				DynExprKind::Reduction(..) => Ok(reduced_value),

				DynExprKind::Unary(un) => match un.kind {
					DynExprUnaryKind::Neg => {
						let a = self.eval_expr(un.expr.as_ref(), z, y, x, reduced_value)?;
						Ok(-a)
					},
					DynExprUnaryKind::Exp => {
						let a = self.eval_expr(un.expr.as_ref(), z, y, x, reduced_value)?;
						Ok(a.exp())
					},
					DynExprUnaryKind::Abs => {
						let a = self.eval_expr(un.expr.as_ref(), z, y, x, reduced_value)?;
						Ok(a.abs())
					},
					DynExprUnaryKind::Sqrt => {
						let a = self.eval_expr(un.expr.as_ref(), z, y, x, reduced_value)?;
						Ok(a.sqrt())
					},
					DynExprUnaryKind::Ln => {
						let a = self.eval_expr(un.expr.as_ref(), z, y, x, reduced_value)?;
						Ok(a.ln().max(-1000.0))
					},
					DynExprUnaryKind::Recip => {
						let a = self.eval_expr(un.expr.as_ref(), z, y, x, reduced_value)?;
						Ok(1.0 / a)
					},
				},
				DynExprKind::Binary(bin) => match bin.kind {
					DynExprBinaryKind::Add => {
						let a = self.eval_expr(bin.lhs.as_ref(), z, y, x, reduced_value)?;
						let b = self.eval_expr(bin.rhs.as_ref(), z, y, x, reduced_value)?;
						Ok(a + b)
					},
					DynExprBinaryKind::Sub => {
						let a = self.eval_expr(bin.lhs.as_ref(), z, y, x, reduced_value)?;
						let b = self.eval_expr(bin.rhs.as_ref(), z, y, x, reduced_value)?;
						Ok(a - b)
					},
					DynExprBinaryKind::Mul => {
						let a = self.eval_expr(bin.lhs.as_ref(), z, y, x, reduced_value)?;
						let b = self.eval_expr(bin.rhs.as_ref(), z, y, x, reduced_value)?;
						Ok(a * b)
					},
				},
			}
		}
	}
}

//--------------------------------------------------------------------------------------------------
