//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::ffi::{c_char, c_int, c_void};
use std::hint::cold_path;
use std::ptr::NonNull;

use crate::tensor::TensorOpError;
use crate::tensor::device::DevicePtr;
use crate::util::ffi_buffer::FfiBuffer;
use crate::{ErrExtra, ErrPack};

//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
#[repr(C)]
pub struct CudaCapability {
	pub major: usize,
	pub minor: usize,
}

#[repr(C)]
pub struct CudaCube {
	pub x: usize,
	pub y: usize,
	pub z: usize,
}

#[repr(C)]
pub struct CudaLaunchConfig {
	pub grid_dim: CudaCube,
	pub block_dim: CudaCube,
	pub shared_mem_bytes: usize,
}

#[repr(C)]
pub struct CudaContextHandle {
	refcnt_munus_one: usize, // TODO - use Atomic?
	device_id: usize,
	capability: CudaCapability,
	warp_size: usize,
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

pub struct Ptx {
	code: String,
	log: String,
}

impl Ptx {
	pub fn code(&self) -> &str {
		&self.code
	}

	pub fn log(&self) -> &str {
		&self.log
	}
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

	fn x17ai_cuda_close_context(ctx: *mut CudaContextHandle, err: FfiBuffer) -> c_int;

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
	/// The zero byte is NOT included in the length of the buffer.
	///
	/// On failure, `ptx` will be empty and `log` will contain error messages.
	///
	/// Note that `log` may contain warnings even on success.
	fn x17ai_cuda_compile_module(
		device_capability: CudaCapability,
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
		stream: *mut CudaStreamHandle,
		kernel: *mut CudaKernelHandle,
		config: *const CudaLaunchConfig,
		args: *const *const c_void,
		err: FfiBuffer,
	) -> c_int;
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct CudaCppError {
	pub msg: Vec<u8>,
}

impl From<CudaCppError> for ErrPack<TensorOpError> {
	#[inline(never)]
	#[cold]
	fn from(err: CudaCppError) -> Self {
		ErrPack {
			code: TensorOpError::DeviceError,
			extra: Some(Box::new(ErrExtra {
				message: Cow::from(String::from_utf8_lossy(&err.msg).into_owned()),
				nested: None,
			})),
		}
	}
}

impl From<CudaCppError> for CudaError {
	#[inline(never)]
	#[cold]
	fn from(err: CudaCppError) -> Self {
		Self {
			extra: Some(Box::new(ErrExtra {
				message: Cow::from(String::from_utf8_lossy(&err.msg).into_owned()),
				nested: None,
			})),
		}
	}
}

pub struct CudaError {
	pub extra: Option<Box<ErrExtra>>,
}

impl CudaError {
	pub fn new<T: Into<Cow<'static, str>>>(message: T) -> Self {
		Self {
			extra: Some(Box::new(ErrExtra { message: message.into(), nested: None })),
		}
	}

	pub fn new2<T: Into<Cow<'static, str>>, N: 'static + std::error::Error + Send + Sync>(
		message: T,
		nested: N,
	) -> Self {
		Self {
			extra: Some(Box::new(ErrExtra {
				message: message.into(),
				nested: Some(Box::new(nested)),
			})),
		}
	}
}

impl From<CudaError> for ErrPack<TensorOpError> {
	fn from(err: CudaError) -> Self {
		Self {
			code: TensorOpError::DeviceError,
			extra: err.extra,
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct CudaStream {
	stream: NonNull<CudaStreamHandle>,
	ctx: NonNull<CudaContextHandle>,
}

impl CudaStream {
	pub fn new(device_id: usize) -> Result<Self, CudaCppError> {
		let mut err = CudaCppError { msg: Vec::new() };

		let ctx = unsafe { x17ai_cuda_open_context(device_id, FfiBuffer::new(&mut err.msg)) };
		let Some(ctx) = NonNull::new(ctx) else {
			cold_path();
			return Err(err);
		};

		let stream = unsafe { x17ai_cuda_open_stream(ctx.as_ptr(), FfiBuffer::new(&mut err.msg)) };
		let Some(stream) = NonNull::new(stream) else {
			cold_path();
			unsafe { x17ai_cuda_close_context(ctx.as_ptr(), FfiBuffer::new(&mut err.msg)) };
			return Err(err);
		};

		Ok(Self { stream, ctx })
	}

	pub fn capability(&self) -> CudaCapability {
		unsafe { self.ctx.as_ref().capability }
	}

	pub fn warp_size(&self) -> usize {
		unsafe { self.ctx.as_ref().warp_size }
	}

	/// # Safety
	///
	/// The allocated block of memory may or may not be initialized.
	pub unsafe fn alloc(&self, bytes: usize) -> Result<DevicePtr, CudaCppError> {
		let mut err = CudaCppError { msg: Vec::new() };

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
		let mut err = CudaCppError { msg: Vec::new() };
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
	) -> Result<(), CudaCppError> {
		let mut err = CudaCppError { msg: Vec::new() };
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
	) -> Result<(), CudaCppError> {
		let mut err = CudaCppError { msg: Vec::new() };
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

	pub fn load_module(&self, ptx: &Ptx) -> Result<CudaModule, CudaCppError> {
		if ptx.code.capacity() <= ptx.code.len()
			|| unsafe { ptx.code.as_ptr().wrapping_add(ptx.code.len()).cast::<u8>().read() } != 0
		{
			cold_path();
			return Err(CudaCppError {
				msg: b"PTX code is not null-terminated".to_vec(),
			});
		}

		let mut err = CudaCppError { msg: Vec::new() };
		let handle = unsafe {
			x17ai_cuda_load_module(
				self.ctx.as_ptr(),
				ptx.code.as_ptr().cast(),
				FfiBuffer::new(&mut err.msg),
			)
		};
		let Some(handle) = NonNull::new(handle) else {
			cold_path();
			return Err(err);
		};
		let mut ctx = self.ctx;
		unsafe { ctx.as_mut() }.refcnt_munus_one += 1;
		Ok(CudaModule { context: self.ctx, handle })
	}

	/// # Safety
	///
	/// TODO
	pub unsafe fn run_kernel(
		&self,
		kernel: &CudaKernel,
		config: &CudaLaunchConfig,
		args: &[*const ()],
	) -> Result<(), CudaCppError> {
		let mut err = CudaCppError { msg: Vec::new() };
		let result = unsafe {
			x17ai_cuda_run_kernel(
				self.stream.as_ptr(),
				kernel.kernel.as_ptr(),
				config as *const CudaLaunchConfig,
				args.as_ptr().cast(),
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

impl Drop for CudaStream {
	fn drop(&mut self) {
		let mut err = CudaCppError { msg: Vec::new() };
		unsafe {
			x17ai_cuda_close_stream(self.stream.as_ptr(), FfiBuffer::new(&mut err.msg));
			x17ai_cuda_close_context(self.ctx.as_ptr(), FfiBuffer::new(&mut err.msg));
		};
	}
}

//--------------------------------------------------------------------------------------------------

pub struct CudaModule {
	pub context: NonNull<CudaContextHandle>,
	pub handle: NonNull<CudaModuleHandle>,
}

impl Drop for CudaModule {
	fn drop(&mut self) {
		let mut err = CudaCppError { msg: Vec::new() };
		unsafe {
			x17ai_cuda_del_module(self.handle.as_ptr(), FfiBuffer::new(&mut err.msg));
			x17ai_cuda_close_context(self.context.as_ptr(), FfiBuffer::new(&mut err.msg));
		};
	}
}

impl CudaModule {
	pub fn get_kernel(self, name: &str) -> Result<CudaKernel, CudaCppError> {
		let mut c_name = Vec::with_capacity(name.len() + 1);
		c_name.extend_from_slice(name.as_bytes());
		c_name.push(0);

		let mut err = CudaCppError { msg: Vec::new() };
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

//--------------------------------------------------------------------------------------------------

pub fn cuda_compile(capability: CudaCapability, source: &str) -> Result<Ptx, CudaCppError> {
	let mut c_source = Vec::with_capacity(source.len() + 1);
	c_source.extend_from_slice(source.as_bytes());
	c_source.push(0);

	let mut ptx = Vec::new();
	let mut log = Vec::new();
	unsafe {
		x17ai_cuda_compile_module(
			capability,
			c_source.as_ptr().cast(),
			FfiBuffer::new(&mut ptx),
			FfiBuffer::new(&mut log),
		);
	}
	if ptx.is_empty() {
		cold_path();
		return Err(CudaCppError { msg: log });
	}

	let mut ptx = String::from_utf8_lossy_owned(ptx);
	ptx.push('\0');
	ptx.pop();
	let log = String::from_utf8_lossy_owned(log);
	Ok(Ptx { code: ptx, log })
}

//--------------------------------------------------------------------------------------------------
