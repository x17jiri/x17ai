//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::ffi::c_void;
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

#[repr(C)]
pub struct DiagnosticBuffer {
	_private: [u8; 0],
}

#[repr(C)]
struct PtrResult<T> {
	value: *mut T,
	diagnostic: *mut DiagnosticBuffer,
}

#[repr(C)]
struct UsizeResult {
	value: usize,
	diagnostic: *mut DiagnosticBuffer,
}

unsafe extern "C" {
	fn x17ai_copy_diagnostic(diag: *mut DiagnosticBuffer, err: FfiBuffer);

	fn x17ai_cuda_open_context(device_id: usize) -> PtrResult<CudaContextHandle>;

	fn x17ai_cuda_close_context(ctx: *mut CudaContextHandle) -> UsizeResult;

	fn x17ai_cuda_context_ptr(ctx: *mut CudaContextHandle) -> *mut c_void;

	fn x17ai_cuda_open_stream(ctx: *mut CudaContextHandle) -> PtrResult<CudaStreamHandle>;

	fn x17ai_cuda_close_stream(stream: *mut CudaStreamHandle) -> UsizeResult;

	fn x17ai_cuda_synchronize(stream: *mut CudaStreamHandle) -> UsizeResult;

	fn x17ai_cuda_alloc(
		stream: *mut CudaStreamHandle,
		bytes: usize,
	) -> PtrResult<CudaDeviceData>;

	fn x17ai_cuda_free(
		stream: *mut CudaStreamHandle,
		ptr: *mut CudaDeviceData,
	) -> UsizeResult;

	fn x17ai_cuda_upload_data(
		stream: *mut CudaStreamHandle,
		src: *const u8,
		dst: *mut CudaDeviceData,
		offset_bytes: usize,
		size_bytes: usize,
	) -> UsizeResult;

	fn x17ai_cuda_download_data(
		stream: *mut CudaStreamHandle,
		src: *mut CudaDeviceData,
		dst: *mut u8,
		offset_bytes: usize,
		size_bytes: usize,
	) -> UsizeResult;
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

#[inline(never)]
#[cold]
fn ptr_result_to_error<T>(result: PtrResult<T>) -> ErrPack<TensorOpError> {
	diagnostic_to_error(result.diagnostic, "CUDA operation failed without diagnostic")
}

#[inline(never)]
#[cold]
fn usize_result_to_error(result: UsizeResult) -> ErrPack<TensorOpError> {
	diagnostic_to_error(
		result.diagnostic,
		&format!("CUDA operation failed without diagnostic; status {}", result.value),
	)
}

#[inline(never)]
#[cold]
fn usize_result_to_error_with_nested(
	result: UsizeResult,
	nested: ErrPack<TensorOpError>,
) -> ErrPack<TensorOpError> {
	let mut result = usize_result_to_error(result);
	if let Some(extra) = &mut result.extra {
		extra.nested = Some(Box::new(nested));
	} else {
		result.extra = Some(Box::new(ErrExtra {
			message: "CUDA operation failed without diagnostic".into(),
			nested: Some(Box::new(nested)),
		}));
	}
	result
}

#[inline(never)]
#[cold]
fn diagnostic_to_error(diagnostic: *mut DiagnosticBuffer, fallback: &str) -> ErrPack<TensorOpError> {
	let mut err = CudaError { msg: Vec::new() };
	unsafe {
		x17ai_copy_diagnostic(diagnostic, FfiBuffer::new(&mut err.msg));
	}
	if err.msg.is_empty() {
		err.msg.extend_from_slice(fallback.as_bytes());
	}
	err.into()
}

#[inline(never)]
#[cold]
fn discard_diagnostic(diag: *mut DiagnosticBuffer) {
	if !diag.is_null() {
		let mut msg = Vec::new();
		unsafe {
			x17ai_copy_diagnostic(diag, FfiBuffer::new(&mut msg));
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
		let ctx_result = unsafe { x17ai_cuda_open_context(device_id) };
		let Some(ctx) = NonNull::new(ctx_result.value) else {
			cold_path();
			return Err(ptr_result_to_error(ctx_result));
		};
		debug_assert!(ctx_result.diagnostic.is_null());

		let stream_result = unsafe { x17ai_cuda_open_stream(ctx.as_ptr()) };
		let Some(stream) = NonNull::new(stream_result.value) else {
			cold_path();
			let result = ptr_result_to_error(stream_result);
			let result2 = unsafe { x17ai_cuda_close_context(ctx.as_ptr()) };
			if result2.value != 0 {
				return Err(usize_result_to_error_with_nested(result2, result));
			}
			debug_assert!(result2.diagnostic.is_null());
			return Err(result);
		};
		debug_assert!(stream_result.diagnostic.is_null());

		Ok(Self { stream, ctx })
	}

	/// # Safety
	///
	/// The allocated block of memory may or may not be initialized.
	pub unsafe fn alloc(&self, bytes: usize) -> Result<DevicePtr, ErrPack<TensorOpError>> {
		let ptr_result = unsafe { x17ai_cuda_alloc(self.stream.as_ptr(), bytes) };
		let Some(ptr) = NonNull::new(ptr_result.value) else {
			cold_path();
			return Err(ptr_result_to_error(ptr_result));
		};
		debug_assert!(ptr_result.diagnostic.is_null());

		Ok(DevicePtr::new(ptr.as_ptr().cast()))
	}

	pub fn synchronize(&self) -> Result<(), ErrPack<TensorOpError>> {
		let result = unsafe { x17ai_cuda_synchronize(self.stream.as_ptr()) };
		if result.value != 0 {
			cold_path();
			return Err(usize_result_to_error(result));
		}
		debug_assert!(result.diagnostic.is_null());
		Ok(())
	}

	/// # Safety
	///
	/// The pointer must be a valid pointer returned by `alloc`.
	pub unsafe fn free(&self, ptr: DevicePtr) {
		let result = unsafe {
			x17ai_cuda_free(self.stream.as_ptr(), ptr.as_ptr::<CudaDeviceData>())
		};
		if result.value != 0 {
			cold_path();
			discard_diagnostic(result.diagnostic);
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
		let result = unsafe {
			x17ai_cuda_upload_data(
				self.stream.as_ptr(),
				src.as_ptr(),
				dst.as_ptr::<CudaDeviceData>(),
				offset_bytes,
				size_bytes,
			)
		};
		if result.value != 0 {
			cold_path();
			return Err(usize_result_to_error(result));
		}
		debug_assert!(result.diagnostic.is_null());
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
		let result = unsafe {
			x17ai_cuda_download_data(
				self.stream.as_ptr(),
				src.as_ptr::<CudaDeviceData>(),
				dst.as_ptr(),
				offset_bytes,
				size_bytes,
			)
		};
		if result.value != 0 {
			cold_path();
			return Err(usize_result_to_error(result));
		}
		debug_assert!(result.diagnostic.is_null());
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
		unsafe {
			let result = x17ai_cuda_close_stream(self.stream.as_ptr());
			if result.value != 0 {
				cold_path();
				discard_diagnostic(result.diagnostic);
			}

			let result = x17ai_cuda_close_context(self.ctx.as_ptr());
			if result.value != 0 {
				cold_path();
				discard_diagnostic(result.diagnostic);
			}
		};
	}
}

//--------------------------------------------------------------------------------------------------
