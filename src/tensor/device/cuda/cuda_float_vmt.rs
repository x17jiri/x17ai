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
	AttentionArgs, DeviceBufferVMT, KernelElemArg, KernelOutput, KernelReduceArg,
};
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::cpu::math::Float;
use crate::tensor::device::cuda::CUDADevice;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::GenericTensor;
use crate::tensor::generic::map::ND;
use crate::tensor::{HasDType, TensorOpError};
use crate::util::LossyInto;
use crate::util::mycell::{BorrowGuard, BorrowMutGuard};

//--------------------------------------------------------------------------------------------------

pub struct CompiledKernel {
	handle: *const std::ffi::c_void,
}

pub(super) struct CUDAFloatVMT<
	T: 'static + HasDType + Float,
	U: 'static + HasDType + Float + From<T> + LossyInto<T>,
> {
	vmt: DeviceBufferVMT,
	compiled_kernels: Vec<Option<Box<CompiledKernel>>>,
	phantom_t: std::marker::PhantomData<T>,
	phantom_u: std::marker::PhantomData<U>,
}

impl<T: 'static + HasDType + Float, U: 'static + HasDType + Float + From<T> + LossyInto<T>>
	CUDAFloatVMT<T, U>
{
	pub fn new(device: &Rc<MaybeUninit<CPUDevice>>, kernel_runner: Rc<KernelRunner>) -> Self {
		let device = device.as_ptr();
		let device = unsafe { NonNull::new_unchecked(device.cast_mut()) };
		let device_is_cpu = true;
		let dtype = T::dtype;
		Self {
			vmt: unsafe {
				DeviceBufferVMT::new(
					device,
					device_is_cpu,
					dtype,
					kernel_runner,
					CUDADevice::drop_buffer,
					Self::read_float,
					Self::load_from_cpu_memory,
					Self::store_to_cpu_memory,
					Self::mm,
					Self::attention,
					Self::run_kernel,
				)
			},
			compiled_kernels: Vec::new(),
			phantom_t: std::marker::PhantomData,
			phantom_u: std::marker::PhantomData,
		}
	}

	unsafe fn cast_this<'a>(vmt: NonNull<DeviceBufferVMT>) -> &'a Self {
		debug_assert!(std::mem::offset_of!(Self, vmt) == 0);
		let vmt = vmt.cast::<Self>();
		unsafe { &*vmt.as_ptr() }
	}

	fn device(&self) -> &CPUDevice {
		let (device, _) = self.vmt.device_ptr().to_raw_parts();
		let cpu_device = device.cast::<CPUDevice>();
		unsafe { cpu_device.as_ref() }
	}

	fn read_float<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		_src: &GenericTensor<ND<0>, BorrowGuard<'buf, DeviceBuffer>>,
	) -> Result<f64, ErrPack<TensorOpError>> {
		todo!("CUDAFloatVMT::read_float is not implemented yet");
	}

	fn load_from_cpu_memory<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		_src: &[u8],
		_dst: &mut GenericTensor<ND<1>, BorrowMutGuard<'buf, DeviceBuffer>>,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("CUDAFloatVMT::load_from_cpu_memory is not implemented yet");
	}

	fn store_to_cpu_memory<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		_src: &GenericTensor<ND<1>, BorrowGuard<'buf, DeviceBuffer>>,
		_dst: &mut [u8],
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("CUDAFloatVMT::store_to_cpu_memory is not implemented yet");
	}

	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::many_single_char_names)]
	fn mm<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		_o: &mut GenericTensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
		_a: &GenericTensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		_b: &GenericTensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		_scale: f64,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("CUDAFloatVMT::mm is not implemented yet");
	}

	fn attention<'buf>(
		_this: NonNull<DeviceBufferVMT>,
		_args: &AttentionArgs,
	) -> Result<(), ErrPack<TensorOpError>> {
		todo!("CUDAFloatVMT::attention is not implemented yet");
	}

	unsafe fn run_kernel(
		this: NonNull<DeviceBufferVMT>,
		kernel_data: &KernelData,
		_o: *const KernelOutput,
		_elemwise_args: *const KernelElemArg,
		_reduce_args: *const KernelReduceArg,
		_scalar_args: *const f64,
		_reduction_size: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		let this = unsafe { Self::cast_this(this) };
		let Some(Some(compiled_kernel)) = this.compiled_kernels.get(kernel_data.id) else {
			cold_path();
			todo!("CUDAFloatExecutor::run_kernel: need to compile kernel");
		};
		let _compiled_kernel = compiled_kernel.as_ref();
		todo!("CUDAFloatExecutor::run_kernel is not implemented yet");
	}
}
