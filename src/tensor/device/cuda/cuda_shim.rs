//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;

//--------------------------------------------------------------------------------------------------

#[link(name = "cuda_shim")]
unsafe extern "C" {
	fn x17ai_cuda_init() -> std::ffi::c_int;
	fn x17ai_cuda_alloc(size: u64) -> *mut std::ffi::c_void;
	fn x17ai_cuda_free(ptr: *mut std::ffi::c_void);
}

//--------------------------------------------------------------------------------------------------

pub struct CudaInitError;
pub struct CudaAllocError;

pub fn cuda_init() -> Result<(), CudaInitError> {
	if unsafe { x17ai_cuda_init() } == 0 {
		Ok(())
	} else {
		Err(CudaInitError) //
	}
}

pub unsafe fn cuda_alloc(size: usize) -> Result<NonNull<u8>, CudaAllocError> {
	let ptr = unsafe { x17ai_cuda_alloc(size as u64) }.cast();
	if let Some(nonnull) = NonNull::new(ptr) {
		Ok(nonnull)
	} else {
		Err(CudaAllocError) //
	}
}

pub unsafe fn cuda_free(ptr: *mut u8) {
	unsafe { x17ai_cuda_free(ptr.cast()) };
}

//--------------------------------------------------------------------------------------------------
