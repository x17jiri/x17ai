//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ffi::{c_char, c_void, CString};
use std::hint::cold_path;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;

use crate::device::DevicePtr;
use crate::util::ffi_buffer::FfiBuffer;
use crate::{ErrPack, TensorOpError};

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
pub struct CudaTimerHandle {
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
pub struct DiagnosticBuffer {
	_private: [u8; 0],
}

#[repr(C)]
struct PtrResult<T> {
	value: *mut T,
	diagnostic: *mut DiagnosticBuffer,
}

unsafe extern "C" {
	fn x17ai_move_diagnostic(diag: *mut DiagnosticBuffer, err: FfiBuffer);

	fn x17ai_cuda_open_context(device_id: usize) -> PtrResult<CudaContextHandle>;

	fn x17ai_cuda_close_context(ctx: *mut CudaContextHandle) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_context_ptr(ctx: *mut CudaContextHandle) -> *mut c_void;

	fn x17ai_cuda_open_stream(ctx: *mut CudaContextHandle) -> PtrResult<CudaStreamHandle>;

	fn x17ai_cuda_close_stream(stream: *mut CudaStreamHandle) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_synchronize(stream: *mut CudaStreamHandle) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_create_timer(
		ctx: *mut CudaContextHandle,
		stream: *mut CudaStreamHandle
	) -> PtrResult<CudaTimerHandle>;

	fn x17ai_cuda_destroy_timer(timer: *mut CudaTimerHandle) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_timer_start(timer: *mut CudaTimerHandle) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_timer_stop(timer: *mut CudaTimerHandle) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_timer_elapsed_seconds(
		timer: *mut CudaTimerHandle,
		ms: *mut f64,
	) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_alloc(
		stream: *mut CudaStreamHandle,
		bytes: usize,
	) -> PtrResult<CudaDeviceData>;

	fn x17ai_cuda_free(
		stream: *mut CudaStreamHandle,
		ptr: *mut CudaDeviceData,
	) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_upload_data(
		stream: *mut CudaStreamHandle,
		src: *const u8,
		dst: *mut CudaDeviceData,
		offset_bytes: usize,
		size_bytes: usize,
	) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_download_data(
		stream: *mut CudaStreamHandle,
		src: *mut CudaDeviceData,
		dst: *mut u8,
		offset_bytes: usize,
		size_bytes: usize,
	) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_load_module(
		ctx: *mut CudaContextHandle,
		cubin_path: *const c_char,
	) -> PtrResult<CudaModuleHandle>;

	fn x17ai_cuda_del_module(module: *mut CudaModuleHandle) -> *mut DiagnosticBuffer;

	fn x17ai_cuda_get_kernel(
		module: *mut CudaModuleHandle,
		name: *const c_char,
		smem_size: usize,
	) -> PtrResult<CudaKernelHandle>;

	fn x17ai_cuda_launch_kernel(
		stream: *mut CudaStreamHandle, kernel: *mut CudaKernelHandle,
		grid_x: usize, grid_y: usize, grid_z: usize,
		block_x: usize, block_y: usize, block_z: usize,
		shared_mem_bytes: usize,
		args: *mut *mut c_void,
	) -> *mut DiagnosticBuffer;
}

//--------------------------------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
#[inline(never)]
fn ptr_result_to_error<T>(result: PtrResult<T>) -> ErrPack<TensorOpError> {
	diagnostic_to_error(result.diagnostic, "CUDA operation failed without diagnostic")
}

#[inline(never)]
fn diagnostic_to_error(diagnostic: *mut DiagnosticBuffer, fallback: &str) -> ErrPack<TensorOpError> {
	ErrPack::new(TensorOpError::Device, diagnostic_to_string(diagnostic, fallback))
}

pub(super) fn diagnostic_to_string(diagnostic: *mut DiagnosticBuffer, fallback: &str) -> String {
	let mut msg = Vec::new();
	unsafe {
		x17ai_move_diagnostic(diagnostic, FfiBuffer::new(&mut msg));
	}
	if msg.is_empty() {
		fallback.to_owned()
	} else {
		String::from_utf8_lossy(&msg).into_owned()
	}
}

#[inline(never)]
fn diagnostic_to_error_with_nested(
	diagnostic: *mut DiagnosticBuffer,
	nested: ErrPack<TensorOpError>,
) -> ErrPack<TensorOpError> {
	ErrPack::new_with_nested(
		TensorOpError::Device,
		diagnostic_to_string(diagnostic, "CUDA operation failed without diagnostic"),
		nested,
	)
}

#[inline(never)]
fn discard_diagnostic(diag: *mut DiagnosticBuffer) {
	if !diag.is_null() {
		let mut msg = Vec::new();
		unsafe {
			x17ai_move_diagnostic(diag, FfiBuffer::new(&mut msg));
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
			if !result2.is_null() {
				cold_path();
				return Err(diagnostic_to_error_with_nested(result2, result));
			}
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
		let diagnostic = unsafe { x17ai_cuda_synchronize(self.stream.as_ptr()) };
		if !diagnostic.is_null() {
			cold_path();
			return Err(diagnostic_to_error(
				diagnostic,
				"x17ai_cuda_synchronize() failed without diagnostic",
			));
		}
		Ok(())
	}

	/// # Safety
	///
	/// The pointer must be a valid pointer returned by `alloc`.
	pub unsafe fn free(&self, ptr: DevicePtr) {
		let diagnostic = unsafe {
			x17ai_cuda_free(self.stream.as_ptr(), ptr.as_ptr::<CudaDeviceData>())
		};
		if !diagnostic.is_null() {
			cold_path();
			discard_diagnostic(diagnostic);
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
		let diagnostic = unsafe {
			x17ai_cuda_upload_data(
				self.stream.as_ptr(),
				src.as_ptr(),
				dst.as_ptr::<CudaDeviceData>(),
				offset_bytes,
				size_bytes,
			)
		};
		if !diagnostic.is_null() {
			cold_path();
			return Err(diagnostic_to_error(
				diagnostic,
				"x17ai_cuda_upload_data() failed without diagnostic",
			));
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
		let diagnostic = unsafe {
			x17ai_cuda_download_data(
				self.stream.as_ptr(),
				src.as_ptr::<CudaDeviceData>(),
				dst.as_ptr(),
				offset_bytes,
				size_bytes,
			)
		};
		if !diagnostic.is_null() {
			cold_path();
			return Err(diagnostic_to_error(
				diagnostic,
				"x17ai_cuda_download_data() failed without diagnostic",
			));
		}
		Ok(())
	}

	pub fn handle(&self) -> *mut CudaStreamHandle {
		self.stream.as_ptr()
	}

	pub fn cuda_context(&self) -> *mut c_void {
		unsafe { x17ai_cuda_context_ptr(self.ctx.as_ptr()) }
	}

	pub fn load_module_from_cubin(
		&self,
		cubin_path: &Path,
	) -> Result<CudaModule, ErrPack<TensorOpError>> {
		let Ok(cubin_path) = CString::new(cubin_path.as_os_str().as_bytes()) else {
			cold_path();
			return Err(ErrPack::new(
				TensorOpError::Device,
				"CUDA cubin path contains an interior NUL byte",
			));
		};
		let module_result = unsafe {
			x17ai_cuda_load_module(self.ctx.as_ptr(), cubin_path.as_ptr())
		};
		let Some(module) = NonNull::new(module_result.value) else {
			cold_path();
			return Err(ptr_result_to_error(module_result));
		};
		debug_assert!(module_result.diagnostic.is_null());

		Ok(CudaModule { handle: module })
	}
}

impl Drop for CudaStream {
	fn drop(&mut self) {
		unsafe {
			let diagnostic = x17ai_cuda_close_stream(self.stream.as_ptr());
			if !diagnostic.is_null() {
				cold_path();
				discard_diagnostic(diagnostic);
			}

			let diagnostic = x17ai_cuda_close_context(self.ctx.as_ptr());
			if !diagnostic.is_null() {
				cold_path();
				discard_diagnostic(diagnostic);
			}
		};
	}
}

//--------------------------------------------------------------------------------------------------

pub struct CudaModule {
	handle: NonNull<CudaModuleHandle>,
}

impl CudaModule {
	pub fn get_kernel(
		self,
		name: &str,
		smem_size: usize,
	) -> Result<CudaKernel, ErrPack<TensorOpError>> {
		let Ok(name) = CString::new(name) else {
			cold_path();
			return Err(ErrPack::new(
				TensorOpError::Device,
				"CUDA kernel name contains an interior NUL byte",
			));
		};

		let kernel_result = unsafe {
			x17ai_cuda_get_kernel(
				self.handle.as_ptr(),
				name.as_ptr(),
				smem_size,
			)
		};
		let Some(kernel) = NonNull::new(kernel_result.value) else {
			cold_path();
			return Err(ptr_result_to_error(kernel_result));
		};
		debug_assert!(kernel_result.diagnostic.is_null());

		Ok(CudaKernel { _module: self, handle: kernel })
	}
}

impl Drop for CudaModule {
	fn drop(&mut self) {
		let diagnostic = unsafe { x17ai_cuda_del_module(self.handle.as_ptr()) };
		if !diagnostic.is_null() {
			cold_path();
			discard_diagnostic(diagnostic);
		}
	}
}

pub struct CudaKernel {
	_module: CudaModule,
	handle: NonNull<CudaKernelHandle>,
}

impl CudaKernel {
	pub fn handle(&self) -> *mut CudaKernelHandle {
		self.handle.as_ptr()
	}

	/// # Safety
	///
	/// `args` must contain one pointer per kernel parameter. Each pointer must point to a host
	/// value with the exact type and layout expected by the loaded CUDA kernel.
	pub unsafe fn launch(
		&self,
		stream: &CudaStream,
		grid_dim: [usize; 3],
		block_dim: [usize; 3],
		shared_mem_bytes: usize,
		args: &mut [*mut c_void],
	) -> Result<(), ErrPack<TensorOpError>> {
		let diagnostic = unsafe {
			x17ai_cuda_launch_kernel(
				stream.stream.as_ptr(), self.handle.as_ptr(),
				grid_dim[0], grid_dim[1], grid_dim[2],
				block_dim[0], block_dim[1], block_dim[2],
				shared_mem_bytes,
				args.as_mut_ptr(),
			)
		};
		if !diagnostic.is_null() {
			cold_path();
			return Err(diagnostic_to_error(
				diagnostic,
				"x17ai_cuda_launch_kernel() failed without diagnostic",
			));
		}
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct CudaEventTimer {
	timer: NonNull<CudaTimerHandle>,
}

impl CudaEventTimer {
	pub fn new(stream: &CudaStream) -> Result<Self, ErrPack<TensorOpError>> {
		let timer_result = unsafe {
			x17ai_cuda_create_timer(stream.ctx.as_ptr(), stream.stream.as_ptr())
		};
		let Some(timer) = NonNull::new(timer_result.value) else {
			cold_path();
			return Err(ptr_result_to_error(timer_result));
		};
		debug_assert!(timer_result.diagnostic.is_null());

		Ok(Self { timer })
	}

	pub fn start(&self) -> Result<(), ErrPack<TensorOpError>> {
		let diagnostic = unsafe { x17ai_cuda_timer_start(self.timer.as_ptr()) };
		if !diagnostic.is_null() {
			cold_path();
			return Err(diagnostic_to_error(
				diagnostic,
				"x17ai_cuda_timer_start() failed without diagnostic",
			));
		}
		Ok(())
	}

	pub fn stop(&self) -> Result<(), ErrPack<TensorOpError>> {
		let diagnostic = unsafe { x17ai_cuda_timer_stop(self.timer.as_ptr()) };
		if !diagnostic.is_null() {
			cold_path();
			return Err(diagnostic_to_error(
				diagnostic,
				"x17ai_cuda_timer_stop() failed without diagnostic",
			));
		}
		Ok(())
	}

	pub fn elapsed_seconds(&self) -> Result<f64, ErrPack<TensorOpError>> {
		let mut seconds = 0.0;
		let diagnostic = unsafe {
			x17ai_cuda_timer_elapsed_seconds(self.timer.as_ptr(), &raw mut seconds)
		};
		if !diagnostic.is_null() {
			cold_path();
			return Err(diagnostic_to_error(
				diagnostic,
				"x17ai_cuda_timer_elapsed_seconds() failed without diagnostic",
			));
		}
		Ok(seconds)
	}
}

impl Drop for CudaEventTimer {
	fn drop(&mut self) {
		let diagnostic = unsafe { x17ai_cuda_destroy_timer(self.timer.as_ptr()) };
		if !diagnostic.is_null() {
			cold_path();
			discard_diagnostic(diagnostic);
		}
	}
}

//--------------------------------------------------------------------------------------------------
