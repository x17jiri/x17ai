//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;

use crate::tensor::device::executor::{KernelElemArg, KernelOutput, KernelReduceArg};

//--------------------------------------------------------------------------------------------------

#[link(name = "cuda_shim")]
unsafe extern "C" {
	// Returns 0 on success
	fn x17ai_cuda_init() -> std::ffi::c_int;

	fn x17ai_cuda_alloc_f32(count: i64) -> *mut std::ffi::c_void;
	fn x17ai_cuda_free(ptr: *mut std::ffi::c_void);

	fn x17ai_cude_new_kernel(
		source: *const std::ffi::c_char,
		len: usize,
	) -> *const std::ffi::c_void;
	fn x17ai_cuda_del_kernel(kernel: *const std::ffi::c_void);

	fn x17ai_cuda_run_kernel(
		kernel: *const std::ffi::c_void,
		o: *const KernelOutput,
		elem_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		const_args: *const f64,
	) -> std::ffi::c_int;
}

//--------------------------------------------------------------------------------------------------

pub struct CudaInitError;
pub struct CudaAllocError;
pub struct CudaNewKernelError;
pub struct CudaRunKernelError;

pub fn init() -> Result<(), CudaInitError> {
	let err = unsafe { x17ai_cuda_init() };
	if err == 0 {
		Ok(())
	} else {
		Err(CudaInitError) //
	}
}

#[allow(clippy::option_if_let_else)]
pub unsafe fn alloc_f32(count: usize) -> Result<NonNull<u8>, CudaAllocError> {
	let ptr = unsafe { x17ai_cuda_alloc_f32(count) }.cast();
	if let Some(nonnull) = NonNull::new(ptr) {
		Ok(nonnull)
	} else {
		Err(CudaAllocError) //
	}
}

pub unsafe fn free(ptr: *mut u8) {
	unsafe { x17ai_cuda_free(ptr.cast()) };
}

pub struct CudaKernelHandle(NonNull<std::ffi::c_void>);

pub unsafe fn new_kernel(source: &str) -> Result<CudaKernelHandle, CudaNewKernelError> {
	let kernel = unsafe { x17ai_cude_new_kernel(source.as_ptr().cast(), source.len()) };
	if let Some(nonnull) = NonNull::new(kernel.cast_mut()) {
		Ok(CudaKernelHandle(nonnull))
	} else {
		Err(CudaNewKernelError) //
	}
}

pub unsafe fn del_kernel(handle: CudaKernelHandle) {
	unsafe { x17ai_cuda_del_kernel(handle.0.as_ptr()) };
}

pub unsafe fn run_kernel(
	handle: &CudaKernelHandle,
	o: *const KernelOutput,
	elem_args: *const KernelElemArg,
	reduce_args: *const KernelReduceArg,
	const_args: *const f64,
) -> Result<(), CudaRunKernelError> {
	let err =
		unsafe { x17ai_cuda_run_kernel(handle.0.as_ptr(), o, elem_args, reduce_args, const_args) };
	if err == 0 {
		Ok(())
	} else {
		Err(CudaRunKernelError) //
	}
}

//--------------------------------------------------------------------------------------------------
