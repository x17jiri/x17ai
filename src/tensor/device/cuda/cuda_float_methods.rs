//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;

use crate::ErrPack;
use crate::tensor::device::DeviceBuffer;
use crate::tensor::device::cpu::math::{Float, FromToF64};
use crate::tensor::device::cuda::CudaDevice;
use crate::tensor::device::kernel::runner::KernelData;
use crate::tensor::generic::map::ND;
use crate::tensor::{HasDType, TensorOpError};
use crate::util::LossyInto;

//--------------------------------------------------------------------------------------------------

pub unsafe fn read_float<T: HasDType + Default + FromToF64>(
	device: &CudaDevice,
	buf: &DeviceBuffer,
	offset: usize,
) -> Result<f64, ErrPack<TensorOpError>> {
	debug_assert!(buf.dtype() == T::dtype);
	let val = T::default();
	unsafe {
		device.cuda_stream.store_to_cpu_memory(
			buf.memory(),
			NonNull::from(&val).cast(),
			offset,
			std::mem::size_of::<T>(),
		)?;
	}
	Ok(val.to_f64())
}

//--------------------------------------------------------------------------------------------------
