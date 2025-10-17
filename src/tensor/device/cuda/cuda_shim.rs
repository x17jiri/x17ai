//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::{c_char, c_int};
use std::hint::cold_path;
use std::ptr::NonNull;

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::{DevicePtr, KernelElemArg, KernelOutput, KernelReduceArg};
use crate::util::ffi_buffer::FfiBuffer;

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct CudaContextHandle {
	_private: [u8; 0],
}

#[repr(C)]
pub struct CudaStreamHandle {
	_private: [u8; 0],
}

#[repr(C)]
pub struct CudaModuleHandle {
	_private: [u8; 0],
}

#[repr(C)]
pub struct CudaKernelHandle {
	_private: [u8; 0],
}

#[repr(C)]
pub struct CudaDeviceData {
	_private: [u8; 0],
}

#[link(name = "cuda_shim")]
unsafe extern "C" {
	// Error behavior of these functions:
	// - functions returning pointers:
	//   - on success: return valid pointer
	//   - on error: return null and fill `err` with the error message
	// - functions returning c_int:
	//   - on success: return 0
	//   - on error: return != 0 and fill `err` with the error message

	fn x17ai_cuda_open_context(device_id: usize, err: FfiBuffer) -> *mut CudaContextHandle;

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
	fn x17ai_cuda_compile_module(
		stream: *mut CudaStreamHandle,
		source: *const c_char,
		ptx: FfiBuffer,
		log: FfiBuffer,
	);

	fn x17ai_cuda_load_module(
		ctx: *mut CudaContextHandle,
		ptx: *const c_char,
		err: FfiBuffer,
	) -> *mut CudaModuleHandle;

	fn x17ai_cuda_del_module(module: *mut CudaModuleHandle, err: FfiBuffer) -> c_int;

	fn x17ai_cuda_get_kernel(
		kernel: *mut CudaModuleHandle,
		name: *const c_char,
		err: FfiBuffer,
	) -> *mut CudaKernelHandle;

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
	pub fn new(device_id: usize) -> Result<Self, CudaError> {
		let mut err = CudaError { msg: Vec::new() };

		let ctx = unsafe { x17ai_cuda_open_context(device_id, FfiBuffer::new(&mut err.msg)) };
		let Some(ctx) = NonNull::new(ctx) else {
			cold_path();
			return Err(err);
		};

		let stream = unsafe { x17ai_cuda_open_stream(ctx.as_ptr(), FfiBuffer::new(&mut err.msg)) };
		let Some(stream) = NonNull::new(stream) else {
			cold_path();
			unsafe { x17ai_cuda_close_context(device_id, FfiBuffer::new(&mut err.msg)) };
			return Err(err);
		};

		Ok(Self { device_id, ctx, stream })
	}

	/// # Safety
	///
	/// The allocated block of memory may or may not be initialized.
	pub unsafe fn alloc(&self, bytes: usize) -> Result<DevicePtr, CudaError> {
		let mut err = CudaError { msg: Vec::new() };

		let ptr =
			unsafe { x17ai_cuda_alloc(self.stream.as_ptr(), bytes, FfiBuffer::new(&mut err.msg)) };
		let Some(ptr) = NonNull::new(ptr) else {
			cold_path();
			return Err(err);
		};

		Ok(DevicePtr::new(ptr.as_ptr().cast()))
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
	) -> Result<(), CudaError> {
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
			return Err(err);
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
			return Err(err);
		}
		Ok(())
	}

	pub fn compile_module(&self, source: &str) -> Result<Vec<u8>, CudaError> {
		let mut c_source = Vec::with_capacity(source.len() + 1);
		c_source.extend_from_slice(source.as_bytes());
		c_source.push(0);

		let mut ptx = Vec::new();
		let mut log = Vec::new();
		unsafe {
			x17ai_cuda_compile_module(
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

	pub fn load_module(&self, mut ptx: Vec<u8>) -> Result<CudaModule, CudaError> {
		if let Some(&last) = ptx.last()
			&& last == 0
		{
			// already null-terminated
		} else {
			ptx.push(0);
		}

		let mut err = CudaError { msg: Vec::new() };
		let handle = unsafe {
			x17ai_cuda_load_module(
				self.ctx.as_ptr(),
				ptx.as_ptr().cast(),
				FfiBuffer::new(&mut err.msg),
			)
		};
		let Some(handle) = NonNull::new(handle) else {
			cold_path();
			return Err(err);
		};
		Ok(CudaModule { handle })
	}
}

impl Drop for CudaStream {
	fn drop(&mut self) {
		let mut err = CudaError { msg: Vec::new() };
		unsafe {
			x17ai_cuda_close_stream(self.stream.as_ptr(), FfiBuffer::new(&mut err.msg));
			x17ai_cuda_close_context(self.device_id, FfiBuffer::new(&mut err.msg));
		};
	}
}

//--------------------------------------------------------------------------------------------------

#[repr(transparent)]
pub struct CudaModule {
	pub handle: NonNull<CudaModuleHandle>,
}

impl Drop for CudaModule {
	fn drop(&mut self) {
		let mut err = CudaError { msg: Vec::new() };
		unsafe {
			x17ai_cuda_del_module(self.handle.as_ptr(), FfiBuffer::new(&mut err.msg));
		};
	}
}

impl CudaModule {
	pub fn get_kernel(self, name: &str) -> Result<CudaKernel, CudaError> {
		let mut c_name = Vec::with_capacity(name.len() + 1);
		c_name.extend_from_slice(name.as_bytes());
		c_name.push(0);

		let mut err = CudaError { msg: Vec::new() };
		let kernel = unsafe {
			x17ai_cuda_get_kernel(
				self.handle.as_ptr(),
				c_name.as_ptr().cast(),
				FfiBuffer::new(&mut err.msg),
			)
		};
		let Some(kernel) = NonNull::new(kernel) else {
			cold_path();
			return Err(err);
		};
		Ok(CudaKernel { _module: self, kernel })
	}
}

pub struct CudaKernel {
	_module: CudaModule, // we need module to keep the kernel alive
	kernel: NonNull<CudaKernelHandle>,
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
		let mut err = CudaError { msg: Vec::new() };
		let result = unsafe {
			x17ai_cuda_run_kernel(
				self.kernel.as_ptr(),
				o,
				elem_args,
				reduce_args,
				const_args,
				FfiBuffer::new(&mut err.msg),
			)
		};
		if result != 0 {
			cold_path();
			return Err(err);
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
