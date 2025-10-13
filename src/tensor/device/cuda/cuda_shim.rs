//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::{c_char, c_void};
use std::hint::cold_path;
use std::ptr::NonNull;

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::{KernelElemArg, KernelOutput, KernelReduceArg};
use crate::util::ffi_buffer::FfiBuffer;

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct StaticCppString {
	data: *const c_char,
	len: usize,
}

impl StaticCppString {
	pub fn to_string(&self) -> String {
		let data = self.data.cast::<u8>();
		let len = self.len;
		let slice = unsafe { std::slice::from_raw_parts(data, len) };
		String::from_utf8_lossy(slice).into_owned()
	}
}

//--------------------------------------------------------------------------------------------------

#[repr(transparent)]
struct DevicePtr(u64);

//--------------------------------------------------------------------------------------------------

// The decision of whether the result is ok or not
// is based on `result == null`, not on `error == null`.
// That way:
// - we can avoid reading `error` most of the time.
// - we can safely convert result to NonNull in Rust.
// - we can use something like `Ok(result, "Err: Cuda returned null")` in the C++ code,
// avoiding one condition.
#[repr(C)]
struct PtrResult {
	result: *mut c_void,
	error: *const StaticCppString,
}

impl PtrResult {
	pub fn into_result<U, F>(self, map: F) -> Result<U, CudaError>
	where
		F: FnOnce(NonNull<c_void>) -> U,
	{
		if let Some(nonnull) = NonNull::new(self.result) {
			Ok(map(nonnull))
		} else {
			cold_path();
			Err(CudaError { msg: unsafe { &*self.error } })
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct DevicePtrResult {
	result: DevicePtr,
	error: *const StaticCppString,
}

impl DevicePtrResult {
	pub fn into_result(self) -> Result<DevicePtr, CudaError> {
		if self.error.is_null() {
			Ok(self.result)
		} else {
			cold_path();
			Err(CudaError { msg: unsafe { &*self.error } })
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct VoidResult {
	error: *const StaticCppString,
}

impl VoidResult {
	pub fn into_result(self) -> Result<(), CudaError> {
		if self.error.is_null() {
			Ok(())
		} else {
			cold_path();
			Err(CudaError { msg: unsafe { &*self.error } })
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[link(name = "cuda_shim")]
unsafe extern "C" {
	pub fn x17ai_test();

	fn x17ai_cuda_open_context(device_id: usize) -> PtrResult;
	fn x17ai_cuda_close_context(device_id: usize) -> VoidResult;

	fn x17ai_cuda_open_stream(ctx: *mut c_void) -> PtrResult;
	fn x17ai_cuda_close_stream(stream: *mut c_void) -> VoidResult;

	fn x17ai_cuda_alloc(stream: *mut c_void, bytes: usize) -> DevicePtrResult;
	fn x17ai_cuda_free(stream: *mut c_void, ptr: DevicePtr) -> VoidResult;

	pub fn x17ai_cuda_upload_data(
		stream: *mut c_void,
		src: *const u8,
		dst: DevicePtr,
		offset_bytes: usize,
		size_bytes: usize,
	) -> VoidResult;

	pub fn x17ai_cuda_download_data(
		stream: *mut c_void,
		src: DevicePtr,
		dst: *mut u8,
		offset_bytes: usize,
		size_bytes: usize,
	) -> VoidResult;

	/// On success, returns 0 and fills `buffer` with the compiled PTX code.
	/// On failure, returns != 0 and fills `buffer` with the error message.
	fn x17ai_cuda_compile_kernel(
		stream: *mut c_void,
		source: *const c_char,
		ptx: FfiBuffer,
		log: FfiBuffer,
	) -> VoidResult;

	fn x17ai_cuda_new_kernel(source: *const c_char, len: usize) -> PtrResult;
	fn x17ai_cuda_del_kernel(kernel: *const c_void);

	fn x17ai_cuda_run_kernel(
		kernel: *const c_void,
		o: *const KernelOutput,
		elem_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		const_args: *const f64,
	) -> VoidResult;
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct CudaError {
	pub msg: &'static StaticCppString,
}

impl From<CudaError> for ErrPack<TensorOpError> {
	fn from(err: CudaError) -> Self {
		TensorOpError::device_error(err.msg.to_string())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct CudaStream {
	device_id: usize,
	ctx: NonNull<c_void>,
	stream: NonNull<c_void>,
}

impl CudaStream {
	pub fn new() -> Result<Self, CudaError> {
		let device_id = 0;
		let ctx = unsafe { x17ai_cuda_open_context(device_id) }.into_result(|ptr| ptr)?;
		let stream = unsafe { x17ai_cuda_open_stream(ctx.as_ptr()) }.into_result(|ptr| ptr)?;
		Ok(Self { device_id, ctx, stream })
	}

	/// # Safety
	///
	/// The allocated block of memory may or may not be initialized.
	pub unsafe fn alloc(&self, bytes: usize) -> Result<DevicePtr, CudaError> {
		unsafe { x17ai_cuda_alloc(self.stream.as_ptr(), bytes) }.into_result()
	}

	/// # Safety
	///
	/// The pointer must be a valid pointer returned by `alloc`.
	pub unsafe fn free(&self, ptr: DevicePtr) {
		unsafe { x17ai_cuda_free(self.stream.as_ptr(), ptr) };
	}

	/// # Safety
	///
	/// TODO
	pub unsafe fn upload_data(
		&self,
		cpu_src: NonNull<u8>,
		cuda_dst: NonNull<u8>,
		offset_bytes: usize,
		size_bytes: usize,
	) -> Result<(), CudaError> {
		unsafe {
			x17ai_cuda_upload_data(
				self.ptr.as_ptr(),
				cpu_src.as_ptr(),
				cuda_dst.as_ptr(),
				offset_bytes,
				size_bytes,
			)
		}
		.into_result()
	}

	/// # Safety
	///
	/// TODO
	pub unsafe fn download_data(
		&self,
		cuda_src: NonNull<u8>,
		cpu_dst: NonNull<u8>,
		offset_bytes: usize,
		size_bytes: usize,
	) -> Result<(), CudaError> {
		unsafe {
			x17ai_cuda_download_data(
				self.ptr.as_ptr(),
				cuda_src.as_ptr(),
				cpu_dst.as_ptr(),
				offset_bytes,
				size_bytes,
			)
		}
		.into_result()
	}

	pub fn compile_kernel(&self, source: &str) -> Result<Vec<u8>, ErrPack<TensorOpError>> {
		let mut c_source = Vec::with_capacity(source.len() + 1);
		c_source.extend_from_slice(source.as_bytes());
		c_source.push(0);
		let c_source_ptr = c_source.as_ptr().cast();

		let stream = self.ptr.as_ptr();
		let mut buffer = Vec::new();
		let err =
			unsafe { x17ai_cuda_compile_kernel(stream, c_source_ptr, FfiBuffer::new(&mut buffer)) };
		if err != 0 {
			cold_path();
			return Err(TensorOpError::device_error(String::from_utf8_lossy_owned(buffer)));
		}
		Ok(buffer)
	}
}

impl Drop for CudaStream {
	fn drop(&mut self) {
		unsafe {
			x17ai_cuda_close_stream(self.ctx.as_ptr(), self.stream.as_ptr());
			x17ai_cuda_close_context(self.device_id);
		};
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKernelHandle(NonNull<c_void>);

#[allow(clippy::option_if_let_else)]
pub fn new_kernel(source: &str) -> Result<CudaKernelHandle, CudaError> {
	unsafe { x17ai_cuda_new_kernel(source.as_ptr().cast(), source.len()) }
		.into_result(CudaKernelHandle)
}

/// # Safety
///
/// The handle must be a valid handle returned by `new_kernel`.
pub unsafe fn del_kernel(handle: CudaKernelHandle) {
	unsafe { x17ai_cuda_del_kernel(handle.0.as_ptr()) };
}

/// # Safety
///
/// TODO
pub unsafe fn run_kernel(
	handle: CudaKernelHandle,
	o: *const KernelOutput,
	elem_args: *const KernelElemArg,
	reduce_args: *const KernelReduceArg,
	const_args: *const f64,
) -> Result<(), CudaError> {
	unsafe { x17ai_cuda_run_kernel(handle.0.as_ptr(), o, elem_args, reduce_args, const_args) }
		.into_result()
}

//--------------------------------------------------------------------------------------------------
