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

	fn x17ai_cuda_alloc(bytes: i64) -> *mut std::ffi::c_void;
	fn x17ai_cuda_free(ptr: *mut std::ffi::c_void);

	fn x17ai_cuda_new_kernel(
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

#[derive(Clone, Copy, Debug)]
pub struct CudaInitError;

#[derive(Clone, Copy, Debug)]
pub struct CudaAllocError;

#[derive(Clone, Copy, Debug)]
pub struct CudaNewKernelError;

#[derive(Clone, Copy, Debug)]
pub struct CudaRunKernelError;

pub fn init() -> Result<(), CudaInitError> {
	let err = unsafe { x17ai_cuda_init() };
	if err == 0 {
		Ok(())
	} else {
		Err(CudaInitError) //
	}
}

/// # Safety
///
/// The allocated block of memory may or may not be initialized.
#[allow(clippy::option_if_let_else)]
pub unsafe fn alloc(bytes: usize) -> Result<NonNull<u8>, CudaAllocError> {
	if let Ok(size) = bytes.try_into()
		&& let ptr = unsafe { x17ai_cuda_alloc(size) }.cast()
		&& let Some(nonnull) = NonNull::new(ptr)
	{
		Ok(nonnull)
	} else {
		Err(CudaAllocError) //
	}
}

/// # Safety
///
/// The pointer must be a valid pointer returned by `alloc_f32`.
pub unsafe fn free(ptr: *mut u8) {
	unsafe { x17ai_cuda_free(ptr.cast()) };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKernelHandle(NonNull<std::ffi::c_void>);

#[allow(clippy::option_if_let_else)]
pub fn new_kernel(source: &str) -> Result<CudaKernelHandle, CudaNewKernelError> {
	let kernel = unsafe { x17ai_cuda_new_kernel(source.as_ptr().cast(), source.len()) };
	if let Some(nonnull) = NonNull::new(kernel.cast_mut()) {
		Ok(CudaKernelHandle(nonnull))
	} else {
		Err(CudaNewKernelError) //
	}
}

/// # Safety
///
/// The handle must be a valid handle returned by `new_kernel`.
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
