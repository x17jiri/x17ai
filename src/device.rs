use std::ptr::NonNull;

use crate::{DeviceAllocError, ErrPack, TensorOpError};

pub mod cpu;
pub mod cuda;
pub mod cuda_shim;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DevicePtr {
	ptr: *mut (),
}

impl DevicePtr {
	#[inline]
	pub fn new(ptr: *mut ()) -> Self {
		Self { ptr }
	}

	/// # Safety
	/// The pointer should only be used by device-specific code that knows what the pointer is.
	///
	/// It could be the device pointer casted to host pointer, or even some handle
	/// that can be used to get the real device pointer.
	///
	/// Since it can be a casted device pointer, dereferencing it may be undefined behavior.
	///
	/// And since it can be some handle, pointer arithmetic on it may not make sense.
	#[inline]
	pub unsafe fn as_ptr<T>(&self) -> *mut T {
		self.ptr.cast::<T>()
	}
}

pub trait Device {
	fn name(&self) -> &str;

	fn is_cpu(&self) -> bool {
		false
	}

	/// # Safety
	/// - The returned buffer can only be used by this device.
	unsafe fn new_buffer(&self, bytes: usize) -> Result<DevicePtr, DeviceAllocError>;

	/// # Safety
	/// TODO
	unsafe fn drop_buffer(&self, device_ptr: DevicePtr, bytes: usize);

	/// # Safety
	/// TODO
	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: DevicePtr,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>>;

	/// # Safety
	/// TODO
	unsafe fn download_data(
		&self,
		src: DevicePtr,
		dst: NonNull<u8>,
		bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>>;
}
