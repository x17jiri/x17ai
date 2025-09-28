//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::device::dtype::common_dtype;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::{HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::mycell::{self, BorrowGuard};

pub mod cpu_float_methods;
pub mod math;

use crate::ErrPack;
use crate::tensor::device::{
	AttentionArgs, DerivesDeviceBase, DeviceBase, DeviceBuffer, KernelElemArg, KernelOutput,
	KernelReduceArg, MatMulArgs, NewDeviceBufferError,
};
use crate::tensor::{DType, Device};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BufAsSliceError {
	InvalidDType,
	NotOnCPUDevice,
}
//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct CPUDevice {
	base: DeviceBase,
	name: String,
}

unsafe impl DerivesDeviceBase for CPUDevice {}

impl CPUDevice {
	pub fn new() -> Rc<Self> {
		Self::new_named("CPU".to_string())
	}

	pub fn new_named(name: String) -> Rc<Self> {
		let kernel_runner = Rc::new(KernelRunner::new());

		DeviceBase::new_device(Self {
			base: DeviceBase::new(true, kernel_runner),
			name,
		})
	}

	pub fn buf_as_slice<'guard, 'buf, T: HasDType>(
		buf: &BorrowGuard<'buf, DeviceBuffer>,
	) -> Result<&'guard [T], BufAsSliceError> {
		if buf.dtype() != T::dtype {
			cold_path();
			return Err(BufAsSliceError::InvalidDType);
		}
		debug_assert!(T::dtype.bytes() == std::mem::size_of::<T>());
		if !buf.device_base().is_cpu() {
			cold_path();
			return Err(BufAsSliceError::NotOnCPUDevice);
		}
		let memory = buf.memory();
		let elems = buf.elems();
		let slice = unsafe { std::slice::from_raw_parts(memory.as_ptr().cast(), elems) };
		Ok(slice)
	}
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	#[inline(never)]
	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<mycell::RefCell<DeviceBuffer>>, NewDeviceBufferError> {
		// TODO - do I need `NewDeviceBufferError` as error type? I never return unsupported dtype
		if let Some(size) = dtype.array_bytes(elems)
			&& let Ok(layout) = std::alloc::Layout::from_size_align(size, dtype.align())
			&& let Some(memory) = NonNull::new(unsafe { std::alloc::alloc(layout) })
		{
			Ok(Rc::new(mycell::RefCell::new(unsafe {
				DeviceBuffer::new(memory, dtype, elems, self)
			})))
		} else {
			cold_path();
			Err(NewDeviceBufferError::AllocationFailed)
		}
	}

	unsafe fn drop_buffer(&self, memory: NonNull<u8>, dtype: DType, elems: usize) {
		unsafe {
			let layout = std::alloc::Layout::from_size_align(
				dtype.array_bytes_unchecked(elems),
				dtype.align(),
			)
			.unwrap_unchecked();
			std::alloc::dealloc(memory.as_ptr(), layout);
		}
	}

	unsafe fn read_float(
		&self,
		buf: &DeviceBuffer,
		offset: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		unsafe { cpu_float_methods::read_float(buf.memory(), buf.dtype(), offset) }
	}

	unsafe fn load_from_cpu_memory(
		&self,
		cpu_src: NonNull<u8>,
		dev_dst: &DeviceBuffer,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			std::ptr::copy_nonoverlapping(
				cpu_src.as_ptr(),
				dev_dst.memory().add(offset_bytes).as_ptr(),
				count_bytes,
			);
		}
		Ok(())
	}

	unsafe fn store_to_cpu_memory(
		&self,
		dev_src: &DeviceBuffer,
		cpu_dst: NonNull<u8>,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			std::ptr::copy_nonoverlapping(
				dev_src.memory().add(offset_bytes).as_ptr(),
				cpu_dst.as_ptr(),
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
		unsafe { cpu_float_methods::mm::<f64>(args, scale) }
	}

	unsafe fn attention(&self, args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement attention for CPUDevice");
	}

	unsafe fn run_kernel(
		&self,
		kernel_data: &KernelData,
		o: &KernelOutput,
		elemwise_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		scalar_args: *const f64,
		reduction_size: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement run_kernel for CPUDevice");
	}
}

//--------------------------------------------------------------------------------------------------
