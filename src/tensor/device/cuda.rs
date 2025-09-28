//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::device::cuda::cuda_shim::{CudaError, CudaStream};
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::device::{
	AttentionArgs, DerivesDeviceBase, DeviceBase, DeviceBuffer, KernelElemArg, KernelOutput,
	KernelReduceArg, MatMulArgs, NewDeviceBufferError,
};
use crate::tensor::{DType, Device, HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::mycell;

pub mod cuda_float_methods;
pub mod cuda_shim;

//--------------------------------------------------------------------------------------------------

pub struct CompiledKernel {
	handle: *const std::ffi::c_void,
}

#[repr(C)]
pub struct CudaDevice {
	base: DeviceBase,
	cuda_stream: CudaStream,
	compiled_kernels: Vec<Option<Box<CompiledKernel>>>,
	name: String,
}

unsafe impl DerivesDeviceBase for CudaDevice {}

impl CudaDevice {
	pub fn new() -> Result<Rc<Self>, CudaError> {
		Self::new_named("CUDA".to_string())
	}

	pub fn new_named(name: String) -> Result<Rc<Self>, CudaError> {
		let cuda_stream = CudaStream::new()?;
		let kernel_runner = Rc::new(KernelRunner::new());

		Ok(DeviceBase::new_device(Self {
			base: DeviceBase::new(true, kernel_runner),
			cuda_stream,
			compiled_kernels: Vec::new(),
			name,
		}))
	}
}

impl Device for CudaDevice {
	fn name(&self) -> &str {
		&self.name
	}

	#[inline(never)]
	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<mycell::RefCell<DeviceBuffer>>, NewDeviceBufferError> {
		if let Some(size) = dtype.array_bytes(elems)
			&& let Ok(memory) = unsafe { self.cuda_stream.alloc(size) }
		{
			Ok(Rc::new(mycell::RefCell::new(unsafe {
				DeviceBuffer::new(memory, dtype, elems, self)
			})))
		} else {
			cold_path();
			Err(NewDeviceBufferError::AllocationFailed)
		}
	}

	unsafe fn drop_buffer(&self, memory: NonNull<u8>, _dtype: DType, _elems: usize) {
		unsafe { self.cuda_stream.free(memory) }
	}

	unsafe fn read_float(
		&self,
		buf: &DeviceBuffer,
		offset: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		match buf.dtype() {
			f32::dtype => unsafe { cuda_float_methods::read_float::<f32>(self, buf, offset) },
			f64::dtype => unsafe { cuda_float_methods::read_float::<f64>(self, buf, offset) },
			_ => {
				cold_path();
				Err(UnsupportedDTypeError.into())
			},
		}
	}

	unsafe fn load_from_cpu_memory(
		&self,
		cpu_src: NonNull<u8>,
		dev_dst: &DeviceBuffer,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			self.cuda_stream.load_from_cpu_memory(
				cpu_src,
				dev_dst.memory(),
				offset_bytes,
				count_bytes,
			)?;
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
			self.cuda_stream.store_to_cpu_memory(
				dev_src.memory(),
				cpu_dst,
				offset_bytes,
				count_bytes,
			)?;
		}
		Ok(())
	}

	unsafe fn mm(&self, args: &MatMulArgs, scale: f64) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement mm for CudaDevice");
	}

	unsafe fn attention(&self, args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement attention for CudaDevice");
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
		let Some(Some(compiled_kernel)) = self.compiled_kernels.get(kernel_data.id) else {
			cold_path();
			todo!("CudaDevice::run_kernel: need to compile kernel");
		};
		let _compiled_kernel = compiled_kernel.as_ref();
		todo!("CudaDevice::run_kernel is not implemented yet");
	}
}
