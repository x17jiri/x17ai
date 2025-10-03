//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::boxed::ThinBox;
use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use hashbrown::HashTable;

use crate::ErrPack;
use crate::tensor::device::cpu::cpu_float_methods::FromToF64;
use crate::tensor::device::cuda::cuda_shim::{CudaError, CudaStream};
use crate::tensor::device::kernel::registry::KernelMap;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::device::{
	AttentionArgs, DerivesDeviceBase, DeviceBase, DeviceBuffer, KernelElemArg, KernelOutput,
	KernelReduceArg, MatMulArgs, NewDeviceBufferError,
};
use crate::tensor::{DType, Device, HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::mycell;

pub mod cuda_shim;

//--------------------------------------------------------------------------------------------------

struct CompiledKernel {
	handle: *const std::ffi::c_void,
	dtype_config_len: usize,
	// followed by dtype_config array
}

impl CompiledKernel {
	fn dtype_config<'a>(&self) -> &'a [u64] {
		unsafe {
			std::slice::from_raw_parts(
				// TODO - if Self has less alignment than u64, this may be unaligned
				(self as *const CompiledKernel).add(1) as *const u64,
				self.dtype_config_len,
			)
		}
	}
}

struct CompiledKernelEntry {
	pub key_hash: u64,
	pub value: ThinBox<CompiledKernel>,
}

#[repr(C)]
pub struct CudaDevice {
	base: DeviceBase,
	cuda_stream: CudaStream,
	compiled_kernels: Vec<Option<Box<HashTable<CompiledKernelEntry>>>>,
	name: String,
}

impl CudaDevice {
	pub fn new() -> Result<Rc<Self>, CudaError> {
		Self::new_named("CUDA".to_string())
	}

	pub fn new_named(name: String) -> Result<Rc<Self>, CudaError> {
		Ok(Rc::new(Self {
			base: DeviceBase {
				kernel_runner: Rc::new(KernelRunner::new()),
				is_cpu: false,
			},
			cuda_stream: CudaStream::new()?,
			compiled_kernels: Vec::new(),
			name,
		}))
	}

	unsafe fn __read_float<T: HasDType + Default + FromToF64>(
		&self,
		buf: &DeviceBuffer,
		offset: usize,
	) -> Result<f64, ErrPack<TensorOpError>> {
		debug_assert!(buf.dtype() == T::dtype);
		let val = T::default();
		unsafe {
			self.cuda_stream.store_to_cpu_memory(
				buf.memory(),
				NonNull::from(&val).cast(),
				offset,
				std::mem::size_of::<T>(),
			)?;
		}
		Ok(val.to_f64())
	}
}

unsafe impl DerivesDeviceBase for CudaDevice {}

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
			f32::dtype => unsafe { self.__read_float::<f32>(buf, offset) },
			f64::dtype => unsafe { self.__read_float::<f64>(buf, offset) },
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

	unsafe fn mm(&self, _args: &MatMulArgs, _scale: f64) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement mm for CudaDevice");
	}

	unsafe fn attention(&self, _args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement attention for CudaDevice");
	}

	unsafe fn run_kernel(
		&self,
		kernel_data: &KernelData,
		o: &KernelOutput,
		elemwise_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		scalar_args: *const f64,
		dtype_config: *const u64,
	) -> Result<(), ErrPack<TensorOpError>> {
		let dtype_config = unsafe { kernel_data.dtype_config(dtype_config) };
		let dtype_config_hash = KernelMap::hash_key(dtype_config);
		if let Some(Some(compiled_kernel_table)) = self.compiled_kernels.get(kernel_data.id)
			&& let Some(compiled_kernel) = compiled_kernel_table.find(dtype_config_hash, |entry| {
				entry.key_hash == dtype_config_hash && entry.value.dtype_config() == dtype_config
			}) {
			todo!("CudaDevice::run_kernel(): run kernel");
		} else {
			cold_path();
			todo!("CudaDevice::run_kernel(): compile and run kernel");
		};
	}
}
