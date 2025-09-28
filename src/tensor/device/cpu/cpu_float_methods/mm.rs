//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::MatMulArgs;
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::cpu::cpu_float_methods::KahanAcc;

//--------------------------------------------------------------------------------------------------

pub unsafe fn mm(args: &MatMulArgs, scale: f64) -> Result<(), ErrPack<TensorOpError>> {
	unsafe {
		for j in 0..args.o_rows {
			for i in 0..args.o_cols {
				let mut t = KahanAcc::<f64>::new();
				for k in 0..args.a_cols {
					let a_offset = args.a_offset + j * args.a_row_stride + k * args.a_col_stride;
					let a_offset_bytes = args.a_dtype.array_bytes_unchecked(a_offset);
					let b_offset = args.b_offset + k * args.b_row_stride + i * args.b_col_stride;
					let b_offset_bytes = args.b_dtype.array_bytes_unchecked(b_offset);
					t.acc_(
						CPUDevice::__read_float(args.a_buf, args.a_dtype, a_offset_bytes)?
							* CPUDevice::__read_float(args.b_buf, args.b_dtype, b_offset_bytes)?,
					);
				}
				t.scale_(scale);
				let o_offset = args.o_offset + j * args.o_row_stride + i * args.o_col_stride;
				let o_offset_bytes = args.o_dtype.array_bytes_unchecked(o_offset);
				CPUDevice::__write_float(args.o_buf, args.o_dtype, o_offset_bytes, t.value())?;
			}
		}
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------
