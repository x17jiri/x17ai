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
use crate::tensor::error::UnsupportedDTypeError;
use crate::tensor::{HasDType, TensorOpError};
use crate::util::intrusive_rc::IntrusiveRc;
use crate::util::intrusive_ref_cell::BorrowGuard;

pub mod cpu_float_methods;

use crate::ErrPack;
use crate::tensor::device::{AttentionArgs, DeviceAllocError, DeviceBuffer, DevicePtr, MatMulArgs};
use crate::tensor::{DType, Device};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NotOnCPUDeviceError;

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

	pub fn buf_as_slice<'buf, 'guard, T: HasDType>(
		buf: &'guard BorrowGuard<'buf, DeviceBuffer>,
	) -> Result<&'guard [T], NotOnCPUDeviceError> {
		if !buf.device().is_cpu() {
			cold_path();
			return Err(NotOnCPUDeviceError);
		}
		unsafe {
			Ok(std::slice::from_raw_parts(
				buf.device_ptr().as_ptr::<T>(),
				buf.byte_len() / std::mem::size_of::<T>(),
			))
		}
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

const BUFFER_ALIGN: usize = std::mem::align_of::<u64>();

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
		bytes: usize,
	) -> Result<IntrusiveRc<DeviceBuffer>, DeviceAllocError> {
		let struct_layout = std::alloc::Layout::new::<DeviceBuffer>();
		if let Ok(buffer_layout) = std::alloc::Layout::from_size_align(bytes, BUFFER_ALIGN)
			&& let Ok((layout, buffer_offset)) = struct_layout.extend(buffer_layout)
			&& let Some(memory) = NonNull::new(unsafe { std::alloc::alloc(layout) })
		{
			let inst = memory.cast::<DeviceBuffer>();
			unsafe {
				inst.write(DeviceBuffer::new(
					DevicePtr::new(memory.add(buffer_offset).as_ptr().cast()),
					bytes,
					self,
				));
				Ok(IntrusiveRc::new(inst))
			}
		} else {
			cold_path();
			Err(DeviceAllocError)
		}
	}

	unsafe fn drop_buffer(
		self: Rc<Self>,
		device_ptr: DevicePtr,
		bytes: usize,
		inst: NonNull<DeviceBuffer>,
	) {
		unsafe {
			let struct_layout = std::alloc::Layout::new::<DeviceBuffer>();
			let buffer_layout =
				std::alloc::Layout::from_size_align(bytes, BUFFER_ALIGN).unwrap_unchecked();
			let (layout, buffer_offset) = struct_layout.extend(buffer_layout).unwrap_unchecked();
			debug_assert!(
				device_ptr.as_ptr::<u8>() == inst.cast::<u8>().add(buffer_offset).as_ptr()
			);
			std::alloc::dealloc(inst.as_ptr().cast(), layout);
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
		let internal_dtype = common_dtype(args.internal_dtype, f64::dtype);
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
		let internal_dtype = common_dtype(data.internal_dtype(), f64::dtype);
		if internal_dtype != f64::dtype {
			cold_path();
			return Err(UnsupportedDTypeError.into());
		}
		unsafe { cpu_float_methods::run_kernel(data) }
	}

	unsafe fn run_reduce_kernel(&self, data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>> {
		assert!(data.reduce_count > 0);
		let internal_dtype = common_dtype(data.internal_dtype(), f64::dtype);
		if internal_dtype != f64::dtype {
			cold_path();
			return Err(UnsupportedDTypeError.into());
		}
		unsafe { cpu_float_methods::run_kernel(data) }
	}
}

//--------------------------------------------------------------------------------------------------
