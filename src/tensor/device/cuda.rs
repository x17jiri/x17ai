//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hint::{cold_path, likely};
use std::ptr::NonNull;
use std::rc::Rc;

use hashbrown::HashTable;
use regex::Regex;

use crate::ErrPack;
use crate::tensor::device::cpu::cpu_float_methods::FromToF64;
use crate::tensor::device::cuda::cuda_shim::{
	CudaCapability, CudaCppError, CudaCube, CudaError, CudaKernel, CudaLaunchConfig, CudaStream,
	Ptx, cuda_compile,
};
use crate::tensor::device::kernel::DynKernelCall;
use crate::tensor::device::{
	AttentionArgs, DeviceBuffer, DevicePtr, MatMulArgs, NewDeviceBufferError,
};
use crate::tensor::{DType, Device, HasDType, TensorOpError, UnsupportedDTypeError};
use crate::util::hasher::HashWord;
use crate::util::{ToBoxedSlice, mycell};

pub mod cuda_shim;

//--------------------------------------------------------------------------------------------------

struct CompiledKernel {
	d1: Option<CudaKernel>,
	d2: Option<CudaKernel>,
	key: Box<[HashWord]>, // TODO - allocate `key` inline at the end of the struct
}

struct CompiledKernelEntry {
	pub key_hash: u64,
	pub value: Box<CompiledKernel>,
}

pub struct CudaDevice {
	cuda_stream: CudaStream,
	hash_random_state: crate::util::hasher::RandomState,
	compiled_kernels: RefCell<HashTable<CompiledKernelEntry>>,
	name: String,
	template_regex: Regex,
}

impl CudaDevice {
	pub fn new(device_id: usize) -> Result<Rc<Self>, CudaCppError> {
		Self::new_named(device_id, format!("CUDA Device {device_id}"))
	}

	pub fn new_named(device_id: usize, name: String) -> Result<Rc<Self>, CudaCppError> {
		Ok(Rc::new(Self {
			cuda_stream: CudaStream::new(device_id)?,
			hash_random_state: crate::util::hasher::RandomState::new(),
			compiled_kernels: RefCell::new(HashTable::with_capacity(20)),
			name,
			template_regex: Regex::new(r"\{\{([A-Za-z0-9_]+)\}\}").unwrap(),
		}))
	}

	pub fn capability(&self) -> CudaCapability {
		self.cuda_stream.capability()
	}

	pub fn compile(&self, src: &str) -> Result<Ptx, CudaCppError> {
		cuda_compile(self.capability(), src)
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

	fn render(&self, template: &str, vars: &HashMap<&str, String>) -> Result<String, CudaError> {
		let mut fail = false;
		let rendered = self.template_regex.replace_all(template, |caps: &regex::Captures| {
			if let Some(val) = vars.get(&caps[1]) {
				val.as_str()
			} else {
				fail = true;
				""
			}
		});
		if fail {
			cold_path();
			let err = CudaError {
				msg: "Failed to render template: missing variable",
			};
			return Err(err);
		}
		Ok(rendered.into_owned())
	}

	fn compile_d1(data: &DynKernelCall) -> Result<CudaKernel, ErrPack<TensorOpError>> {
		todo!("implement compile_d1 for CudaDevice");
	}

	fn compile_d2(_data: &DynKernelCall) -> Result<CudaKernel, ErrPack<TensorOpError>> {
		todo!("implement compile_d2 for CudaDevice");
	}

	fn get_kernel(&self, data: &DynKernelCall) -> Result<&CudaKernel, ErrPack<TensorOpError>> {
		let Ok(mut compiled_kernels) = self.compiled_kernels.try_borrow_mut() else {
			cold_path();
			let err = CudaError {
				msg: "Internal error: compiled_kernels is already borrowed",
			};
			return Err(err.into());
		};
		let key = data.key;
		let key_hash = self.hash_random_state.hash_one(key);
		let found = compiled_kernels.find_mut(key_hash, |item| {
			item.key_hash == key_hash && likely(item.value.key.as_ref() == key)
		});
		let mut entry = //.
			if let Some(found) = found {
				NonNull::from_mut(found.value.as_mut())
			} else {
				cold_path();
				NonNull::from_mut(
					compiled_kernels
						.insert_unique(
							key_hash,
							CompiledKernelEntry {
								key_hash,
								value: Box::new(CompiledKernel {
									d1: None,
									d2: None,
									key: key.to_boxed_slice(),
								}),
							},
							|item| item.key_hash,
						)
						.get_mut().value.as_mut(),
				)
			};
		let entry = unsafe { entry.as_mut() };
		if data.output.size[0] == 1 {
			if entry.d1.is_none() {
				entry.d1 = Some(Self::compile_d1(data)?);
			}
			Ok(unsafe { entry.d1.as_ref().unwrap_unchecked() })
		} else {
			if entry.d2.is_none() {
				entry.d2 = Some(Self::compile_d2(data)?);
			}
			Ok(unsafe { entry.d2.as_ref().unwrap_unchecked() })
		}
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
		/*
		let kernel_bytes = std::fs::read("/home/spock/prog/x17ai/kernel.cu").unwrap();
		let kernel_str = String::from_utf8_lossy_owned(kernel_bytes);
		let ptx = self.compile(&kernel_str).ok().unwrap();
		let module = self.cuda_stream.load_module(&ptx).ok().unwrap();
		let kernel = module.get_kernel("x17ai_kernel").ok().unwrap();
		*/
		let kernel = self.get_kernel(data)?;
		let blocks = data.output.size[1].div_ceil(256);
		let config = CudaLaunchConfig {
			grid_dim: CudaCube { x: blocks, y: 1, z: 1 },
			block_dim: CudaCube { x: 256, y: 1, z: 1 },
			shared_mem_bytes: 0,
		};
		unsafe {
			self.cuda_stream.run_kernel(
				kernel,
				&config,
				&[
					std::ptr::from_ref(data.output).cast(),
					std::ptr::from_ref(data.elemwise_args).cast(),
				],
			)?;
		}
		Ok(())
	}
}
