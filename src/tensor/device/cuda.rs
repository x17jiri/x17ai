//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::{cold_path, likely};
use std::ptr::NonNull;
use std::rc::Rc;

use hashbrown::HashTable;

use crate::ErrPack;
use crate::tensor::device::cpu::cpu_float_methods::FromToF64;
use crate::tensor::device::cuda::cuda_shim::{CudaError, CudaKernel, CudaStream};
use crate::tensor::device::kernel::DynKernelCall;
use crate::tensor::device::{
	AttentionArgs, DeviceBuffer, DevicePtr, MatMulArgs, NewDeviceBufferError,
};
use crate::tensor::{DType, Device, HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::hasher::HashWord;
use crate::util::mycell;

pub mod cuda_shim;

//--------------------------------------------------------------------------------------------------

struct CompiledKernel {
	kernel: CudaKernel,
	key: Box<[HashWord]>, // TODO - allocate `key` inline at the end of the struct
}

struct CompiledKernelEntry {
	pub key_hash: u64,
	pub value: Box<CompiledKernel>,
}

pub struct CudaDevice {
	cuda_stream: CudaStream,
	hash_random_state: crate::util::hasher::RandomState,
	compiled_kernels: HashTable<CompiledKernelEntry>,
	name: String,
}

impl CudaDevice {
	pub fn new(device_id: usize) -> Result<Rc<Self>, CudaError> {
		Self::new_named(device_id, format!("CUDA Device {}", device_id))
	}

	pub fn new_named(device_id: usize, name: String) -> Result<Rc<Self>, CudaError> {
		Ok(Rc::new(Self {
			cuda_stream: CudaStream::new(device_id)?,
			hash_random_state: crate::util::hasher::RandomState::new(),
			compiled_kernels: HashTable::with_capacity(20),
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
			self.cuda_stream.download_data(
				buf.device_ptr(),
				NonNull::from(&val).cast(),
				offset,
				std::mem::size_of::<T>(),
			)?;
		}
		Ok(val.to_f64())
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

	unsafe fn drop_buffer(&self, device_ptr: DevicePtr, _dtype: DType, _elems: usize) {
		unsafe { self.cuda_stream.free(device_ptr) }
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

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: &DeviceBuffer,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			self.cuda_stream.upload_data(src, dst.device_ptr(), offset_bytes, count_bytes)?;
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
			self.cuda_stream.download_data(src.device_ptr(), dst, offset_bytes, count_bytes)?;
		}
		Ok(())
	}

	unsafe fn mm(&self, _args: &MatMulArgs, _scale: f64) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement mm for CudaDevice");
	}

	unsafe fn attention(&self, _args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>> {
		todo!("implement attention for CudaDevice");
	}

	unsafe fn run_kernel(&self, data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>> {
		let key = data.key;
		let key_hash = self.hash_random_state.hash_one(key);
		if let Some(kernel) = self.compiled_kernels.find(key_hash, |item| {
			item.key_hash == key_hash && likely(item.value.key.as_ref() == key)
		}) {
			unsafe {
				kernel.value.kernel.run(
					std::ptr::from_ref(data.output),
					data.elemwise_args.as_ptr(),
					data.reduce_args.as_ptr(),
					data.scalar_args.as_ptr(),
				)?;
				Ok(())
			}
		} else {
			cold_path();
			todo!("CudaDevice::run_kernel(): compile and run kernel");
		}
	}
}
