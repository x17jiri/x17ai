//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::{c_char, c_int, c_void};
use std::hint::cold_path;
use std::ptr::NonNull;

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
pub struct CudaKernelHandle;

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
	fn x17ai_cuda_free(
		stream: *mut CudaStreamHandle,
		ptr: *mut CudaDeviceData,
		err: FfiBuffer,
	) -> c_int;

	pub fn x17ai_cuda_upload_data(
		stream: *mut CudaStreamHandle,
		src: *mut u8,
		dst: *mut CudaDeviceData,
		offset_bytes: usize,
		size_bytes: usize,
		err: FfiBuffer,
	) -> c_int;

	pub fn x17ai_cuda_download_data(
		stream: *mut CudaStreamHandle,
		src: *const CudaDeviceData,
		dst: *mut u8,
		offset_bytes: usize,
		size_bytes: usize,
		err: FfiBuffer,
	) -> c_int;

	/// On success, `ptx` will contain the compiled PTX code terminated by a zero byte.
	/// On failure, `ptx` will be empty and `log` will contain error messages.
	/// Note that `log` may contain warnings even on success.
	fn x17ai_cuda_compile_kernel(
		stream: *mut CudaStreamHandle,
		source: *const c_char,
		ptx: FfiBuffer,
		log: FfiBuffer,
	);

	fn x17ai_cuda_new_kernel(
		ctx: *mut CudaContextHandle,
		ptx: *const c_char,
		err: FfiBuffer,
	) -> *mut CudaKernelHandle;
	fn x17ai_cuda_del_kernel(kernel: *mut CudaKernelHandle, err: FfiBuffer) -> c_int;

	fn x17ai_cuda_run_kernel(
		kernel: *mut CudaKernelHandle,
		o: *const KernelOutput,
		elem_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		const_args: *const f64,
		err: FfiBuffer,
	) -> c_int;
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct CudaError {
	pub msg: Vec<u8>,
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
	ctx: NonNull<CudaContextHandle>,
	stream: NonNull<CudaStreamHandle>,
}

impl CudaStream {
	pub fn new() -> Result<Self, CudaError> {
		let mut err = Vec::new();
		let device_id = 0;
		let Some(ctx) =
			NonNull::new(unsafe { x17ai_cuda_open_context(device_id, FfiBuffer::new(&mut err)) })
		else {
			cold_path();
			return Err(CudaError { msg: err });
		};
		let Some(stream) =
			NonNull::new(unsafe { x17ai_cuda_open_stream(ctx.as_ptr(), FfiBuffer::new(&mut err)) })
		else {
			cold_path();
			unsafe { x17ai_cuda_close_context(device_id, FfiBuffer::new(&mut err)) };
			return Err(CudaError { msg: err });
		};
		Ok(Self { device_id, ctx, stream })
	}

	/// # Safety
	///
	/// The allocated block of memory may or may not be initialized.
	pub unsafe fn alloc(&self, bytes: usize) -> Result<DevicePtr, CudaError> {
		let mut err = Vec::new();
		let Some(ptr) = NonNull::new(unsafe {
			x17ai_cuda_alloc(self.stream.as_ptr(), bytes, FfiBuffer::new(&mut err))
		}) else {
			cold_path();
			return Err(CudaError { msg: err });
		};
		Ok(DevicePtr::new(ptr.as_ptr().cast()))
	}

	/// # Safety
	///
	/// The pointer must be a valid pointer returned by `alloc`.
	pub unsafe fn free(&self, ptr: DevicePtr) {
		let mut err = Vec::new();
		unsafe {
			x17ai_cuda_free(
				self.stream.as_ptr(),
				ptr.as_ptr::<CudaDeviceData>(),
				FfiBuffer::new(&mut err),
			);
		}
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
		let mut err = Vec::new();
		let result = unsafe {
			x17ai_cuda_upload_data(
				self.stream.as_ptr(),
				src.as_ptr(),
				dst.as_ptr::<CudaDeviceData>(),
				offset_bytes,
				size_bytes,
				FfiBuffer::new(&mut err),
			)
		};
		if result != 0 {
			cold_path();
			return Err(CudaError { msg: err });
		}
		Ok(())
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
		let mut err = Vec::new();
		let result = unsafe {
			x17ai_cuda_download_data(
				self.stream.as_ptr(),
				src.as_ptr::<CudaDeviceData>(),
				dst.as_ptr(),
				offset_bytes,
				size_bytes,
				FfiBuffer::new(&mut err),
			)
		};
		if result != 0 {
			cold_path();
			return Err(CudaError { msg: err });
		}
		Ok(())
	}

	pub fn compile_kernel(&self, source: &str) -> Result<Vec<u8>, CudaError> {
		let mut c_source = Vec::with_capacity(source.len() + 1);
		c_source.extend_from_slice(source.as_bytes());
		c_source.push(0);

		let mut ptx = Vec::new();
		let mut log = Vec::new();
		unsafe {
			x17ai_cuda_compile_kernel(
				self.stream.as_ptr(),
				c_source.as_ptr().cast(),
				FfiBuffer::new(&mut ptx),
				FfiBuffer::new(&mut log),
			);
		}
		if ptx.is_empty() {
			cold_path();
			return Err(CudaError { msg: log });
		}
		Ok(ptx) // TODO - should we return log as well?
	}

	pub fn new_kernel(&self, mut source: Vec<u8>) -> Result<CudaKernel, CudaError> {
		if let Some(&last) = source.last()
			&& last == 0
		{
			// already null-terminated
		} else {
			source.push(0);
		}

		let mut err = Vec::new();
		let Some(handle) = NonNull::new(unsafe {
			x17ai_cuda_new_kernel(source.as_ptr().cast(), FfiBuffer::new(&mut err))
		}) else {
			cold_path();
			return Err(CudaError { msg: err });
		};
		Ok(CudaKernel { handle })
	}
}

impl Drop for CudaStream {
	fn drop(&mut self) {
		let mut err = Vec::new();
		unsafe {
			x17ai_cuda_close_stream(self.stream.as_ptr(), FfiBuffer::new(&mut err));
			x17ai_cuda_close_context(self.device_id, FfiBuffer::new(&mut err));
		};
	}
}

//--------------------------------------------------------------------------------------------------

#[repr(transparent)]
pub struct CudaKernel {
	pub handle: NonNull<CudaKernelHandle>,
}

impl Drop for CudaKernel {
	fn drop(&mut self) {
		let mut err = Vec::new();
		unsafe {
			x17ai_cuda_del_kernel(self.handle.as_ptr(), FfiBuffer::new(&mut err));
		};
	}
}

impl CudaKernel {
	/// # Safety
	///
	/// TODO
	pub unsafe fn run(
		&self,
		o: *const KernelOutput,
		elem_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		const_args: *const f64,
	) -> Result<(), CudaError> {
		let mut err = Vec::new();
		let result = unsafe {
			x17ai_cuda_run_kernel(
				self.handle.as_ptr(),
				o,
				elem_args,
				reduce_args,
				const_args,
				FfiBuffer::new(&mut err),
			)
		};
		if result != 0 {
			cold_path();
			return Err(CudaError { msg: err });
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
