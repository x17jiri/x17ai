//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::{c_char, c_int, c_void};
use std::hint::cold_path;
use std::ptr::NonNull;

use thin_vec::ThinVec;

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::{DevicePtr, KernelElemArg, KernelOutput, KernelReduceArg};
use crate::util::ffi_buffer::FfiBuffer;

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct CudaContextHandle;

#[repr(C)]
pub struct CudaStreamHandle;

#[repr(C)]
pub struct CudaDeviceData;

#[link(name = "cuda_shim")]
unsafe extern "C" {
	pub fn x17ai_test();

	/// On error, returns null and fills `err` with the error message.
	fn x17ai_cuda_open_context(device_id: usize, err: FfiBuffer) -> *mut CudaContextHandle;
	/// On error, returns != 0 and fills `err` with the error message.
	fn x17ai_cuda_close_context(device_id: usize, err: FfiBuffer) -> c_int;

	fn x17ai_cuda_open_stream(ctx: *mut CudaContextHandle, err: FfiBuffer)
	-> *mut CudaStreamHandle;
	fn x17ai_cuda_close_stream(stream: *mut CudaStreamHandle, err: FfiBuffer) -> c_int;

	fn x17ai_cuda_alloc(
		stream: *mut CudaStreamHandle,
		bytes: usize,
		err: FfiBuffer,
	) -> *mut CudaDeviceData;
	fn x17ai_cuda_free(stream: *mut CudaStreamHandle, ptr: *mut CudaDeviceData) -> c_int;

	pub fn x17ai_cuda_upload_data(
		stream: *mut CudaStreamHandle,
		src: *mut u8,
		dst: *mut CudaDeviceData,
		offset_bytes: usize,
		size_bytes: usize,
		err: FfiBuffer,
	) -> c_int;

	pub fn x17ai_cuda_download_data(
		stream: *mut CudaStream,
		src: *const CudaDeviceData,
		dst: *mut u8,
		offset_bytes: usize,
		size_bytes: usize,
		err: FfiBuffer,
	) -> c_int;

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

#[derive(Clone)]
pub struct CudaError {
	pub msg: ThinVec<u8>,
}

impl From<CudaError> for ErrPack<TensorOpError> {
	#[inline(never)]
	fn from(err: CudaError) -> Self {
		TensorOpError::device_error(String::from_utf8_lossy(&err.msg).into_owned())
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
		unsafe { x17ai_cuda_alloc(self.stream.as_ptr(), bytes) }
			.into_result(|ptr| DevicePtr::new(ptr.as_ptr().cast()))
	}

	/// # Safety
	///
	/// The pointer must be a valid pointer returned by `alloc`.
	pub unsafe fn free(&self, ptr: DevicePtr) {
		unsafe { x17ai_cuda_free(self.stream.as_ptr(), ptr.as_ptr::<c_void>()) };
	}

	/// # Safety
	///
	/// TODO
	pub unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: DevicePtr,
		offset_bytes: usize,
		size_bytes: usize,
	) -> Result<(), CudaError> {
		unsafe {
			x17ai_cuda_upload_data(
				self.stream.as_ptr(),
				src.as_ptr(),
				dst,
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
		src: DevicePtr,
		dst: NonNull<u8>,
		offset_bytes: usize,
		size_bytes: usize,
	) -> Result<(), CudaError> {
		unsafe {
			x17ai_cuda_download_data(
				self.stream.as_ptr(),
				src,
				dst.as_ptr(),
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

		let stream = self.stream.as_ptr();
		let mut ptx = Vec::new();
		let mut log = Vec::new();
		let result = unsafe {
			x17ai_cuda_compile_kernel(
				stream,
				c_source_ptr,
				FfiBuffer::new(&mut ptx),
				FfiBuffer::new(&mut log),
			)
		};
		if let Err(err) = result.into_result() {
			cold_path();
			return Err(TensorOpError::device_error(String::from_utf8_lossy_owned(buffer)));
		}
		Ok(buffer)
	}
}

impl Drop for CudaStream {
	fn drop(&mut self) {
		unsafe {
			x17ai_cuda_close_stream(self.stream.as_ptr());
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
