//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::ffi::{c_int, c_void};
use std::hint::cold_path;
use std::ptr::NonNull;

use crate::device::DevicePtr;
use crate::util::ffi_buffer::FfiBuffer;
use crate::{ErrExtra, ErrPack, TensorOpError};

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct CudaContextHandle {
	refcnt_munus_one: usize, // TODO - use Atomic?
	_private: [u8; 0],
}

#[repr(C)]
pub struct CudaStreamHandle {
	_private: [u8; 0],
}

#[repr(C)]
pub struct CudaDeviceData {
	_private: [u8; 0],
}

unsafe extern "C" {
	fn x17ai_cuda_open_context(device_id: usize, err: FfiBuffer) -> *mut CudaContextHandle;

	fn x17ai_cuda_close_context(ctx: *mut CudaContextHandle, err: FfiBuffer) -> c_int;

	fn x17ai_cuda_context_ptr(ctx: *mut CudaContextHandle) -> *mut c_void;

	fn x17ai_cuda_open_stream(ctx: *mut CudaContextHandle, err: FfiBuffer)
	-> *mut CudaStreamHandle;

	fn x17ai_cuda_close_stream(stream: *mut CudaStreamHandle, err: FfiBuffer) -> c_int;

	fn x17ai_cuda_synchronize(stream: *mut CudaStreamHandle, err: FfiBuffer) -> c_int;

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

	fn x17ai_cuda_upload_data(
		stream: *mut CudaStreamHandle,
		src: *const u8,
		dst: *mut CudaDeviceData,
		offset_bytes: usize,
		size_bytes: usize,
		err: FfiBuffer,
	) -> c_int;

	fn x17ai_cuda_download_data(
		stream: *mut CudaStreamHandle,
		src: *mut CudaDeviceData,
		dst: *mut u8,
		offset_bytes: usize,
		size_bytes: usize,
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
	#[cold]
	fn from(err: CudaError) -> Self {
		Self {
			code: TensorOpError::Device,
			extra: Some(Box::new(ErrExtra {
				message: Cow::from(String::from_utf8_lossy(&err.msg).into_owned()),
				nested: None,
			})),
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct CudaStream {
	stream: NonNull<CudaStreamHandle>,
	ctx: NonNull<CudaContextHandle>,
}

impl CudaStream {
	pub fn new(device_id: usize) -> Result<Self, ErrPack<TensorOpError>> {
		let mut err = CudaError { msg: Vec::new() };

		let ctx = unsafe { x17ai_cuda_open_context(device_id, FfiBuffer::new(&mut err.msg)) };
		let Some(ctx) = NonNull::new(ctx) else {
			cold_path();
			return Err(err.into());
		};

		let stream = unsafe { x17ai_cuda_open_stream(ctx.as_ptr(), FfiBuffer::new(&mut err.msg)) };
		let Some(stream) = NonNull::new(stream) else {
			cold_path();
			unsafe { x17ai_cuda_close_context(ctx.as_ptr(), FfiBuffer::new(&mut err.msg)) };
			return Err(err.into());
		};

		Ok(Self { stream, ctx })
	}

	/// # Safety
	///
	/// The allocated block of memory may or may not be initialized.
	pub unsafe fn alloc(&self, bytes: usize) -> Result<DevicePtr, ErrPack<TensorOpError>> {
		let mut err = CudaError { msg: Vec::new() };

		let ptr =
			unsafe { x17ai_cuda_alloc(self.stream.as_ptr(), bytes, FfiBuffer::new(&mut err.msg)) };
		let Some(ptr) = NonNull::new(ptr) else {
			cold_path();
			return Err(err.into());
		};

		Ok(DevicePtr::new(ptr.as_ptr().cast()))
	}

	pub fn synchronize(&self) -> Result<(), ErrPack<TensorOpError>> {
		let mut err = CudaError { msg: Vec::new() };
		let result = unsafe { x17ai_cuda_synchronize(self.stream.as_ptr(), FfiBuffer::new(&mut err.msg)) };
		if result != 0 {
			cold_path();
			return Err(err.into());
		}
		Ok(())
	}

	/// # Safety
	///
	/// The pointer must be a valid pointer returned by `alloc`.
	pub unsafe fn free(&self, ptr: DevicePtr) {
		let mut err = CudaError { msg: Vec::new() };
		unsafe {
			x17ai_cuda_free(
				self.stream.as_ptr(),
				ptr.as_ptr::<CudaDeviceData>(),
				FfiBuffer::new(&mut err.msg),
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
	) -> Result<(), ErrPack<TensorOpError>> {
		let mut err = CudaError { msg: Vec::new() };
		let result = unsafe {
			x17ai_cuda_upload_data(
				self.stream.as_ptr(),
				src.as_ptr(),
				dst.as_ptr::<CudaDeviceData>(),
				offset_bytes,
				size_bytes,
				FfiBuffer::new(&mut err.msg),
			)
		};
		if result != 0 {
			cold_path();
			return Err(err.into());
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
	) -> Result<(), ErrPack<TensorOpError>> {
		let mut err = CudaError { msg: Vec::new() };
		let result = unsafe {
			x17ai_cuda_download_data(
				self.stream.as_ptr(),
				src.as_ptr::<CudaDeviceData>(),
				dst.as_ptr(),
				offset_bytes,
				size_bytes,
				FfiBuffer::new(&mut err.msg),
			)
		};
		if result != 0 {
			cold_path();
			return Err(err.into());
		}
		Ok(())
	}

	pub fn handle(&self) -> *mut CudaStreamHandle{
		self.stream.as_ptr()
	}

	pub fn cuda_context(&self) -> *mut c_void {
		unsafe { x17ai_cuda_context_ptr(self.ctx.as_ptr()) }
	}
}

impl Drop for CudaStream {
	fn drop(&mut self) {
		let mut err = CudaError { msg: Vec::new() };
		unsafe {
			x17ai_cuda_close_stream(self.stream.as_ptr(), FfiBuffer::new(&mut err.msg));
			x17ai_cuda_close_context(self.ctx.as_ptr(), FfiBuffer::new(&mut err.msg));
		};
	}
}

//--------------------------------------------------------------------------------------------------
