//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::buffer::{KernelElemArg, KernelOutput, KernelReduceArg};

//--------------------------------------------------------------------------------------------------

#[repr(C)]
struct PointerResult {
	// TODO - the decision of ok or err should be based on `result == null`,
	// not on `error == null`.
	// That way:
	// - we can avoid reading `error` most of the time.
	// - we can safely convert result to NonNull in Rust.
	// - We can use something like `return (stream, "Err: Cuda returned null")` in the C code.
	result: *mut std::ffi::c_void,
	error: *const std::ffi::c_char,
}

struct VoidResult {
	error: *const std::ffi::c_char,
}

#[link(name = "cuda_shim")]
unsafe extern "C" {
	// Returns 0 on success
	fn x17ai_cuda_open_stream() -> PointerResult;
	fn x17ai_cuda_close_stream(stream: *mut std::ffi::c_void) -> VoidResult;

	fn x17ai_cuda_alloc(stream: *mut std::ffi::c_void, bytes: usize) -> PointerResult;
	fn x17ai_cuda_free(stream: *mut std::ffi::c_void, ptr: *mut std::ffi::c_void) -> VoidResult;

	pub fn x17ai_cuda_load_from_cpu_memory(
		stream: *mut std::ffi::c_void,
		cpu_src: *const u8,
		cuda_dst: *mut u8,
		offset_bytes: usize,
		size_bytes: usize,
	) -> VoidResult;

	pub fn x17ai_cuda_store_to_cpu_memory(
		stream: *mut std::ffi::c_void,
		cuda_src: *const u8,
		cpu_dst: *mut u8,
		offset_bytes: usize,
		size_bytes: usize,
	) -> VoidResult;

	fn x17ai_cuda_new_kernel(source: *const std::ffi::c_char, len: usize) -> PointerResult;
	fn x17ai_cuda_del_kernel(kernel: *const std::ffi::c_void);

	fn x17ai_cuda_run_kernel(
		kernel: *const std::ffi::c_void,
		o: *const KernelOutput,
		elem_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		const_args: *const f64,
	) -> VoidResult;
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct CudaError {
	pub msg: &'static str,
}

impl From<CudaError> for ErrPack<TensorOpError> {
	fn from(err: CudaError) -> Self {
		TensorOpError::device_error(err.msg)
	}
}

//--------------------------------------------------------------------------------------------------

pub struct CudaStream {
	ptr: NonNull<std::ffi::c_void>,
}

impl CudaStream {
	pub fn new() -> Result<Self, CudaError> {
		let ptr = unsafe { x17ai_cuda_open_stream() };
		if let Some(nonnull) = NonNull::new(ptr) {
			Ok(Self { ptr: nonnull })
		} else {
			cold_path();
			Err(CudaError { msg: "Failed to open CUDA stream" })
		}
	}

	/// # Safety
	///
	/// The allocated block of memory may or may not be initialized.
	pub unsafe fn alloc(&self, bytes: usize) -> Result<NonNull<u8>, CudaError> {
		let ptr = unsafe { x17ai_cuda_alloc(self.ptr.as_ptr(), bytes) }.cast();
		if let Some(nonnull) = NonNull::new(ptr) {
			Ok(nonnull)
		} else {
			cold_path();
			Err(CudaError { msg: "Failed to allocate CUDA memory" })
		}
	}

	/// # Safety
	///
	/// The pointer must be a valid pointer returned by `alloc`.
	pub unsafe fn free(&self, ptr: NonNull<u8>) {
		unsafe { x17ai_cuda_free(self.ptr.as_ptr(), ptr.as_ptr().cast()) };
	}

	pub unsafe fn load_from_cpu_memory(
		&self,
		cpu_src: NonNull<u8>,
		cuda_dst: NonNull<u8>,
		offset_bytes: usize,
		size_bytes: usize,
	) -> Result<(), CudaError> {
		let err = unsafe {
			x17ai_cuda_load_from_cpu_memory(
				self.ptr.as_ptr(),
				cpu_src.as_ptr(),
				cuda_dst.as_ptr(),
				offset_bytes,
				size_bytes,
			)
		};
		if err == 0 {
			Ok(())
		} else {
			cold_path();
			Err(CudaError {
				msg: "Failed to load from CPU memory to CUDA memory",
			})
		}
	}

	pub unsafe fn store_to_cpu_memory(
		&self,
		cuda_src: NonNull<u8>,
		cpu_dst: NonNull<u8>,
		offset_bytes: usize,
		size_bytes: usize,
	) -> Result<(), CudaError> {
		let err = unsafe {
			x17ai_cuda_store_to_cpu_memory(
				self.ptr.as_ptr(),
				cuda_src.as_ptr(),
				cpu_dst.as_ptr(),
				offset_bytes,
				size_bytes,
			)
		};
		if err == 0 {
			Ok(())
		} else {
			cold_path();
			Err(CudaError {
				msg: "Failed to store from CUDA memory to CPU memory",
			})
		}
	}
}

impl Drop for CudaStream {
	fn drop(&mut self) {
		unsafe { x17ai_cuda_close_stream(self.ptr.as_ptr()) };
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct CudaNewKernelError;

#[derive(Clone, Copy, Debug)]
pub struct CudaRunKernelError;

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
