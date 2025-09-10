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
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut, DeviceBufferVMT};
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::cpu::math::Float;
use crate::tensor::device::cuda::CUDADevice;
use crate::tensor::device::executor::{
	Executor, ExecutorError, KernelElemArg, KernelOutput, KernelReduceArg,
};
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::map::ND;
use crate::tensor::{HasDType, generic};
use crate::util::LossyInto;

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

	fn read_bin<'buf>(
		&self,
		_dst: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		_src: &mut dyn std::io::Read,
	) -> Result<(), ErrPack<ExecutorError>> {
		todo!("CUDAFloatExecutor::read_bin is not implemented yet");
	}

	fn write_bin<'buf>(
		&self,
		_src: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_dst: &mut dyn std::io::Write,
	) -> Result<(), ErrPack<ExecutorError>> {
		todo!("CUDAFloatExecutor::write_bin is not implemented yet");
	}

	fn randn_clamped<'buf>(
		&self,
		_o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		todo!("CUDAFloatExecutor::randn_clamped is not implemented yet");
	}

	fn sum_all<'buf>(
		&self,
		_a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<f64, ErrPack<ExecutorError>> {
		// TODO - rework the API to return a tensor instead of a scalar
		todo!("CUDAFloatExecutor::sum_all is not implemented yet");
	}

	fn approx_eq<'buf>(
		&self,
		_a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_eps: f64,
	) -> Result<bool, ErrPack<ExecutorError>> {
		todo!("CUDAFloatExecutor::approx_eq is not implemented yet");
	}

	fn softmax<'buf>(
		&self,
		_out: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		_inp: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		todo!("CUDAFloatExecutor::softmax is not implemented yet");
	}

	fn softmax_<'buf>(
		&self,
		_t: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>> {
		todo!("CUDAFloatExecutor::softmax_ is not implemented yet");
	}

	#[allow(clippy::panic_in_result_fn)]
	#[allow(clippy::many_single_char_names)]
	fn mm<'buf>(
		&self,
		_o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		_a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		_scale: f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		todo!("CUDAFloatExecutor::mm is not implemented yet");
	}

	fn attention(
		&self,
		_o: &mut generic::Tensor<ND<3>, DeviceBufferRefMut>, // [inputs, qo_heads, vo_features]
		_q: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, qo_heads, qk_features]
		_k: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, k_heads, qk_features]
		_v: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, v_heads, vo_features]
	) {
		todo!("CUDAFloatExecutor::attention is not implemented yet");
	}

	unsafe fn run_kernel(
		&self,
		kernel_data: &KernelData,
		o: *const KernelOutput,
		elem_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		const_args: *const f64,
	) -> Result<(), ErrPack<ExecutorError>> {
		let Some(Some(compiled_kernel)) = self.compiled_kernels.get(kernel_data.id) else {
			cold_path();
			todo!("CUDAFloatExecutor::run_kernel: need to compile kernel");
		};
		let compiled_kernel = compiled_kernel.as_ref();
		todo!("CUDAFloatExecutor::run_kernel is not implemented yet");
	}
}
