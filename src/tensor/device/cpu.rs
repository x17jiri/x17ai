//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::map::ND;
use crate::tensor::{HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::mycell::{self, BorrowGuard};

pub mod attention;
pub mod cpu_float_vmt;
pub mod math;

use crate::ErrPack;
use crate::tensor::device::cpu::cpu_float_vmt::CPUFloatVMT;
use crate::tensor::device::{
	AttentionArgs, DeviceBuffer, KernelElemArg, KernelOutput, KernelReduceArg, MatMulArgs,
	NewDeviceBufferError,
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
	f32_vmt: CPUFloatVMT<f32, f64>,
}

impl CPUDevice {
	pub fn new() -> Rc<Self> {
		Self::new_named("CPU".to_string())
	}

	pub fn new_named(name: String) -> Rc<Self> {
		let kernel_runner = Rc::new(KernelRunner::new());

		let mut rc_uninit = Rc::new_uninit();
		let instance = Self {
			name,
			f32_vmt: CPUFloatVMT::new(&rc_uninit, kernel_runner),
		};
		unsafe {
			Rc::get_mut_unchecked(&mut rc_uninit).write(instance);
			rc_uninit.assume_init()
		}
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
		// TODO - do i need `NewDeviceBufferError` as error type? I never return unsupported dtype
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

	unsafe fn drop_buffer(&self, data: NonNull<u8>, dtype: DType, elems: usize) {
		unsafe {
			let layout = std::alloc::Layout::from_size_align(
				dtype.array_bytes_unchecked(elems),
				dtype.align(),
			)
			.unwrap_unchecked();
			std::alloc::dealloc(data.as_ptr(), layout);
		}
	}

	unsafe fn read_float(
		&self,
		(map, buf): (ND<0>, &DeviceBuffer),
	) -> Result<f64, ErrPack<TensorOpError>> {
		match buf.dtype() {
			dtype if dtype == f32::dtype => unsafe {
				cpu_float_vmt::read_float::<f32>((map, buf.memory()))
			},
			dtype if dtype == f64::dtype => unsafe {
				cpu_float_vmt::read_float::<f64>((map, buf.memory()))
			},
			_ => {
				cold_path();
				Err(UnsupportedDTypeError.into())
			},
		}
	}

	unsafe fn load_from_cpu_memory(
		&self,
		cpu_src: NonNull<u8>,
		dev_dst: (ND<0>, &DeviceBuffer),
		count: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement load_from_cpu_memory for CPUDevice");
	}

	unsafe fn store_to_cpu_memory(
		&self,
		dev_src: (ND<0>, &DeviceBuffer),
		cpu_dst: NonNull<u8>,
		count: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement store_to_cpu_memory for CPUDevice");
	}

	unsafe fn mm(&self, args: &MatMulArgs, scale: f64) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement mm for CPUDevice");
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
