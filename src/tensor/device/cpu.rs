//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::device::cpu::cpu_float_methods::FromToF64;
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::device::kernel::DynKernelCall;
use crate::tensor::{HasDType, TensorOpError};
use crate::util::mycell::{self, BorrowGuard};

pub mod cpu_float_methods;

use crate::ErrPack;
use crate::tensor::device::{
	AttentionArgs, DevBufAllocFailedError, DeviceBuffer, DevicePtr, MatMulArgs,
};
use crate::tensor::{DType, Device};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BufAsSliceError {
	InvalidDType,
	NotOnCPUDevice,
}
//--------------------------------------------------------------------------------------------------

pub struct CPUDevice {
	name: String,
}

impl CPUDevice {
	pub fn new() -> Rc<Self> {
		Self::new_named("CPU".to_string())
	}

	pub fn new_named(name: String) -> Rc<Self> {
		Rc::new(Self { name })
	}

	pub fn buf_as_slice<'guard, 'buf, T: HasDType>(
		buf: &BorrowGuard<'buf, DeviceBuffer>,
	) -> Result<&'guard [T], BufAsSliceError> {
		if buf.dtype() != T::dtype {
			cold_path();
			return Err(BufAsSliceError::InvalidDType);
		}
		debug_assert!(T::dtype.bytes() == std::mem::size_of::<T>());
		if !buf.device().is_cpu() {
			cold_path();
			return Err(BufAsSliceError::NotOnCPUDevice);
		}
		unsafe { Ok(std::slice::from_raw_parts(buf.device_ptr().as_ptr::<T>(), buf.elems())) }
	}

	unsafe fn __read_float(
		buf: DevicePtr,
		dtype: DType,
		offset_bytes: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		match dtype {
			f32::dtype => unsafe {
				let ptr = buf.as_ptr::<u8>().add(offset_bytes).cast::<f32>();
				Ok(ptr.read().to_f64())
			},
			f64::dtype => unsafe {
				let ptr = buf.as_ptr::<u8>().add(offset_bytes).cast::<f64>();
				Ok(ptr.read().to_f64())
			},
			_ => {
				cold_path();
				Err(UnsupportedDTypeError.into())
			},
		}
	}

	unsafe fn __write_float(
		buf: DevicePtr,
		dtype: DType,
		offset_bytes: usize,
		value: f64,
	) -> Result<(), ErrPack<TensorOpError>> {
		match dtype {
			dtype if dtype == f32::dtype => unsafe {
				let ptr = buf.as_ptr::<u8>().add(offset_bytes).cast::<f32>();
				ptr.write(f32::from_f64(value));
				Ok(())
			},
			dtype if dtype == f64::dtype => unsafe {
				let ptr = buf.as_ptr::<u8>().add(offset_bytes).cast::<f64>();
				ptr.write(f64::from_f64(value));
				Ok(())
			},
			_ => {
				cold_path();
				Err(UnsupportedDTypeError.into())
			},
		}
	}
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn is_cpu(&self) -> bool {
		true
	}

	#[inline(never)]
	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<mycell::RefCell<DeviceBuffer>>, DevBufAllocFailedError> {
		// TODO - do I need `NewDeviceBufferError` as error type? I never return unsupported dtype
		if let Some(size) = dtype.array_bytes(elems)
			&& let Ok(layout) = std::alloc::Layout::from_size_align(size, dtype.align())
			&& let Some(memory) = NonNull::new(unsafe { std::alloc::alloc(layout) })
		{
			Ok(Rc::new(mycell::RefCell::new(unsafe {
				DeviceBuffer::new(DevicePtr::new(memory.as_ptr().cast()), dtype, elems, self)
			})))
		} else {
			cold_path();
			Err(DevBufAllocFailedError::AllocationFailed)
		}
	}

	unsafe fn drop_buffer(&self, device_ptr: DevicePtr, dtype: DType, elems: usize) {
		unsafe {
			let layout = std::alloc::Layout::from_size_align(
				dtype.array_bytes_unchecked(elems),
				dtype.align(),
			)
			.unwrap_unchecked();
			std::alloc::dealloc(device_ptr.as_ptr::<u8>(), layout);
		}
	}

	unsafe fn read_float(
		&self,
		buf: &DeviceBuffer,
		offset: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		unsafe {
			let offset_bytes = buf.dtype().array_bytes_unchecked(offset);
			Self::__read_float(buf.device_ptr(), buf.dtype(), offset_bytes)
		}
	}

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: &DeviceBuffer,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			std::ptr::copy_nonoverlapping(
				src.as_ptr(),
				dst.device_ptr().as_ptr::<u8>().add(offset_bytes),
				count_bytes,
			);
		}
		Ok(())
	}

	unsafe fn download_data(
		&self,
		src: &DeviceBuffer,
		dst: NonNull<u8>,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			std::ptr::copy_nonoverlapping(
				src.device_ptr().as_ptr::<u8>().add(offset_bytes),
				dst.as_ptr(),
				count_bytes,
			);
		}
		Ok(())
	}

	unsafe fn mm(&self, args: &MatMulArgs, scale: f64) -> Result<(), ErrPack<TensorOpError>> {
		let internal_dtype = common_dtype(args.internal_dtype, f64::dtype)?;
		if internal_dtype != f64::dtype {
			cold_path();
			return Err(UnsupportedDTypeError.into());
		}
		unsafe { cpu_float_methods::mm(args, scale) }
	}

	unsafe fn attention(&self, _args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement attention for CPUDevice");
	}

	unsafe fn run_elemwise_kernel(
		&self,
		data: &DynKernelCall,
	) -> Result<(), ErrPack<TensorOpError>> {
		assert!(data.reduce_count == 0);
		let internal_dtype = common_dtype(data.internal_dtype(), f64::dtype)?;
		if internal_dtype != f64::dtype {
			cold_path();
			return Err(UnsupportedDTypeError.into());
		}
		unsafe { cpu_float_methods::run_kernel(data) }
	}

	unsafe fn run_reduce_kernel(&self, data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>> {
		assert!(data.reduce_count > 0);
		let internal_dtype = common_dtype(data.internal_dtype(), f64::dtype)?;
		if internal_dtype != f64::dtype {
			cold_path();
			return Err(UnsupportedDTypeError.into());
		}
		unsafe { cpu_float_methods::run_kernel(data) }
	}
}

//--------------------------------------------------------------------------------------------------
