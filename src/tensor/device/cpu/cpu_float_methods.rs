//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;

use crate::ErrPack;
use crate::tensor::device::MatMulArgs;
use crate::tensor::device::cpu::math::{Float, FromToF64};
use crate::tensor::{DType, HasDType, TensorOpError, UnsupportedDTypeError};

pub use attention::attention;
pub use kernel_eval::run_kernel;

mod attention;
mod kernel_eval;

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
	offset_bytes: usize,
	value: f64,
) -> Result<(), ErrPack<TensorOpError>> {
	match dtype {
		dtype if dtype == f32::dtype => unsafe {
			let ptr = buf.cast::<u8>().add(offset_bytes).cast::<f32>();
			ptr.write(f32::from_f64(value));
			Ok(())
		},
		dtype if dtype == f64::dtype => unsafe {
			let ptr = buf.cast::<u8>().add(offset_bytes).cast::<f64>();
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
				let o_offset_bytes = args.o_dtype.array_bytes_unchecked(o_offset);
				write_float(args.o_buf, args.o_dtype, o_offset_bytes, t.to_f64())?;
			}
		}
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------
