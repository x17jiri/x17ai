//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::device::DeviceBuffer;
use crate::tensor::device::buffer::{
	AttentionArgs, DeviceBufferVMT, DeviceBufferVMTData, KernelElemArg, KernelOutput,
	KernelReduceArg, MatMulArgs,
};
use crate::tensor::device::cpu::math::Float;
use crate::tensor::device::cuda::CudaDevice;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::map::ND;
use crate::tensor::{HasDType, TensorOpError};
use crate::util::LossyInto;

//--------------------------------------------------------------------------------------------------

pub struct CompiledKernel {
	handle: *const std::ffi::c_void,
}

#[repr(C)]
pub(super) struct CudaFloatVMT<
	T: 'static + HasDType + Float,
	U: 'static + HasDType + Float + From<T> + LossyInto<T>,
> {
	vmt: DeviceBufferVMT,
	compiled_kernels: Vec<Option<Box<CompiledKernel>>>,
	phantom_t: std::marker::PhantomData<T>,
	phantom_u: std::marker::PhantomData<U>,
}

#[allow(clippy::unnecessary_wraps)]
impl<T: 'static + HasDType + Float, U: 'static + HasDType + Float + From<T> + LossyInto<T>>
	CudaFloatVMT<T, U>
{
	pub fn new(device: &Rc<MaybeUninit<CudaDevice>>, kernel_runner: Rc<KernelRunner>) -> Self {
		let device = device.as_ptr();
		Self {
			vmt: unsafe {
				DeviceBufferVMT::new(DeviceBufferVMTData {
					device: NonNull::new_unchecked(device.cast_mut()),
					device_is_cpu: false,
					dtype: T::dtype,
					kernel_runner,
					drop_buffer: CudaDevice::drop_buffer,
					read_float: Self::read_float,
					load_from_cpu_memory: Self::load_from_cpu_memory,
					store_to_cpu_memory: Self::store_to_cpu_memory,
					mm: Self::mm,
					attention: Self::attention,
					run_kernel: Self::run_kernel,
				})
			},
			compiled_kernels: Vec::new(),
			phantom_t: std::marker::PhantomData,
			phantom_u: std::marker::PhantomData,
		}
	}

	unsafe fn cast_this(vmt: &DeviceBufferVMT) -> &Self {
		debug_assert!(std::mem::offset_of!(Self, vmt) == 0);
		unsafe { vmt.cast::<Self>() }
	}

	fn device(&self) -> &CudaDevice {
		unsafe { self.vmt.cast_device::<CudaDevice>() }
	}

	unsafe fn read_float(
		this: &DeviceBufferVMT,
		dev_src: (ND<0>, &DeviceBuffer),
	) -> Result<f64, ErrPack<TensorOpError>> {
		let dev = unsafe { Self::cast_this(this) }.device();
		let (map, buf) = dev_src;
		let val = T::default();
		unsafe {
			dev.cuda_stream.store_to_cpu_memory(
				buf.device_data(),
				NonNull::from(&val).cast(),
				map.offset,
				std::mem::size_of::<T>(),
			)?;
		}
		Ok(val.to_f64())
	}

	unsafe fn load_from_cpu_memory(
		this: &DeviceBufferVMT,
		cpu_src: NonNull<u8>,
		dev_dst: (ND<0>, &DeviceBuffer),
		count: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		let dev = unsafe { Self::cast_this(this) }.device();
		let (map, buf) = dev_dst;
		unsafe {
			dev.cuda_stream.load_from_cpu_memory(
				cpu_src,
				buf.device_data(),
				map.offset * std::mem::size_of::<T>(),
				count * std::mem::size_of::<T>(),
			)?;
		}
		Ok(())
	}

	unsafe fn store_to_cpu_memory(
		this: &DeviceBufferVMT,
		dev_src: (ND<0>, &DeviceBuffer),
		cpu_dst: NonNull<u8>,
		count: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		let dev = unsafe { Self::cast_this(this) }.device();
		let (map, buf) = dev_src;
		unsafe {
			dev.cuda_stream.store_to_cpu_memory(
				buf.device_data(),
				cpu_dst,
				map.offset * std::mem::size_of::<T>(),
				count * std::mem::size_of::<T>(),
			)?;
		}
		Ok(())
	}

	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::many_single_char_names)]
	unsafe fn mm(
		_this: &DeviceBufferVMT,
		_args: &MatMulArgs,
		_scale: f64,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("CUDAFloatVMT::mm is not implemented yet");
	}

	unsafe fn attention(
		_this: &DeviceBufferVMT,
		_args: &AttentionArgs,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("CUDAFloatVMT::attention is not implemented yet");
	}

	unsafe fn run_kernel(
		this: &DeviceBufferVMT,
		kernel_data: &KernelData,
		_o: &KernelOutput,
		_elemwise_args: *const KernelElemArg,
		_reduce_args: *const KernelReduceArg,
		_scalar_args: *const f64,
		_reduction_size: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		let this = unsafe { Self::cast_this(this) };
		let Some(Some(compiled_kernel)) = this.compiled_kernels.get(kernel_data.id) else {
			cold_path();
			todo!("CUDAFloatVMT::run_kernel: need to compile kernel");
		};
		let _compiled_kernel = compiled_kernel.as_ref();
		todo!("CUDAFloatVMT::run_kernel is not implemented yet");
	}
}
