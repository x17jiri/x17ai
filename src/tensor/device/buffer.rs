//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;
use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::NewDeviceBufferError;
use crate::tensor::device::executor::ExecutorError;
use crate::tensor::device::kernel::runner::KernelData;
use crate::tensor::generic::buffer::Buffer;
use crate::tensor::generic::map::ND;
use crate::tensor::generic::{self};

use super::Device;
use super::dtype::DType;

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct KernelElemArg {
	pub stride_bytes: [usize; 2],
	pub offset_bytes: usize,
	pub device_data: *const u8,
}

#[repr(C)]
pub struct KernelReduceArg {
	pub reduction_size: usize,
	pub stride_bytes: [usize; 2],
	pub offset_bytes: usize,
	pub device_data: *const u8,
}

#[repr(C)]
pub struct KernelOutput {
	pub size: [usize; 2],
	pub stride_bytes: [usize; 2],
	pub offset_bytes: usize,
	pub device_data: *mut u8,
}

//--------------------------------------------------------------------------------------------------

pub struct DeviceBufferVMT {
	pub device: *const dyn Device,
	pub device_is_cpu: bool,
	pub dtype: DType,

	pub new_buffer: fn(
		this: NonNull<dyn Device>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<DeviceBuffer>, NewDeviceBufferError>,

	pub drop_buffer: unsafe fn(this: NonNull<DeviceBufferVMT>, elems: usize, device_data: *mut u8),

	pub read_bin: for<'buf> fn(
		this: NonNull<DeviceBufferVMT>,
		dst: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		src: &mut dyn std::io::Read,
	) -> Result<(), ErrPack<ExecutorError>>,

	pub write_bin: for<'buf> fn(
		this: NonNull<DeviceBufferVMT>,
		src: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		dst: &mut dyn std::io::Write,
	) -> Result<(), ErrPack<ExecutorError>>,

	pub randn_clamped: for<'buf> fn(
		this: NonNull<DeviceBufferVMT>,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
	) -> Result<(), ErrPack<ExecutorError>>,

	pub mm: for<'buf> fn(
		this: NonNull<DeviceBufferVMT>,
		o: &mut generic::Tensor<ND<2>, DeviceBufferRefMut<'buf>>,
		a: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		b: &generic::Tensor<ND<2>, DeviceBufferRef<'buf>>,
		scale: f64,
	) -> Result<(), ErrPack<ExecutorError>>,

	pub attention: fn(
		this: NonNull<DeviceBufferVMT>,
		o: &mut generic::Tensor<ND<3>, DeviceBufferRefMut>, // [inputs, qo_heads, vo_features]
		q: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, qo_heads, qk_features]
		k: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, k_heads, qk_features]
		v: &generic::Tensor<ND<3>, DeviceBufferRef>,        // [inputs, v_heads, vo_features]
	),

	pub run_kernel: unsafe fn(
		this: NonNull<DeviceBufferVMT>,
		kernel_data: &KernelData,
		o: *const KernelOutput,
		elemwise_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		scalar_args: *const f64,
	) -> Result<(), ErrPack<ExecutorError>>,
}

pub struct DeviceBuffer {
	pub device_data: *mut u8,
	pub elems: usize,
	pub read_count: Cell<usize>,
	pub write_count: Cell<usize>,
	pub vmt: NonNull<DeviceBufferVMT>,
}

impl DeviceBuffer {
	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let vmt = unsafe { self.vmt.as_ref() };
		let my_dev = vmt.device.cast::<u8>();

		let dev = std::ptr::from_ref(device);
		let dev = dev.cast::<u8>();

		my_dev == dev
	}

	pub fn try_borrow(&self) -> Result<DeviceBufferRef<'_>, BorrowError> {
		DeviceBufferRef::new(self)
	}

	pub fn try_borrow_mut(&self) -> Result<DeviceBufferRefMut<'_>, BorrowMutError> {
		DeviceBufferRefMut::new(self)
	}
}

impl Drop for DeviceBuffer {
	fn drop(&mut self) {
		unsafe {
			(self.vmt.as_ref().drop_buffer)(self.vmt, self.elems, self.device_data);
		}
	}
}

//--------------------------------------------------------------------------------------------------

impl Buffer for Rc<DeviceBuffer> {
	fn len(&self) -> usize {
		self.elems
	}
}

impl Buffer for &DeviceBuffer {
	fn len(&self) -> usize {
		self.elems
	}
}

impl<'a> Buffer for DeviceBufferRef<'a> {
	fn len(&self) -> usize {
		self.device_buffer.elems
	}
}

impl<'a> Buffer for DeviceBufferRefMut<'a> {
	fn len(&self) -> usize {
		self.device_buffer.elems
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BorrowError;

impl std::error::Error for BorrowError {}

impl std::fmt::Display for BorrowError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Cannot borrow the device buffer")
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BorrowMutError;

impl std::error::Error for BorrowMutError {}

impl std::fmt::Display for BorrowMutError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Cannot borrow the device buffer mutably")
	}
}

//--------------------------------------------------------------------------------------------------

pub struct DeviceBufferRef<'a> {
	device_buffer: &'a DeviceBuffer,
}

impl<'a> DeviceBufferRef<'a> {
	pub fn new(device_buffer: &'a DeviceBuffer) -> Result<Self, BorrowError> {
		let read_count = device_buffer.read_count.get();
		let write_count = device_buffer.write_count.get();

		if write_count != 0 {
			cold_path();
			return Err(BorrowError);
		}

		device_buffer.read_count.set(read_count + 1);
		Ok(Self { device_buffer })
	}

	pub unsafe fn new_unsafe(device_buffer: &'a DeviceBuffer, fail: &mut usize) -> Self {
		let read_count = device_buffer.read_count.get();
		let write_count = device_buffer.write_count.get();

		*fail |= write_count;

		device_buffer.read_count.set(read_count + 1);
		Self { device_buffer }
	}

	pub fn device_buffer(&self) -> &'a DeviceBuffer {
		self.device_buffer
	}
}

impl<'a> Clone for DeviceBufferRef<'a> {
	fn clone(&self) -> Self {
		let read_count = self.device_buffer.read_count.get();

		debug_assert!(read_count > 0, "DeviceBufferRef: invalid counter state");

		self.device_buffer.read_count.set(read_count + 1);
		Self { device_buffer: self.device_buffer }
	}
}

impl<'a> Drop for DeviceBufferRef<'a> {
	fn drop(&mut self) {
		let read_count = self.device_buffer.read_count.get();

		debug_assert!(read_count > 0, "DeviceBufferRef: invalid counter state");

		self.device_buffer.read_count.set(read_count - 1);
	}
}

impl<'a> std::ops::Deref for DeviceBufferRef<'a> {
	type Target = DeviceBuffer;

	#[inline]
	fn deref(&self) -> &DeviceBuffer {
		self.device_buffer
	}
}

impl<'a> From<DeviceBufferRefMut<'a>> for DeviceBufferRef<'a> {
	fn from(value: DeviceBufferRefMut<'a>) -> Self {
		let read_count = value.device_buffer.read_count.get();
		let write_count = value.device_buffer.write_count.get();

		debug_assert!(write_count > 0, "DeviceBufferRefMut: invalid counter state");

		value.device_buffer.read_count.set(read_count + 1);
		value.device_buffer.write_count.set(write_count - 1);
		let result = Self { device_buffer: value.device_buffer };

		std::mem::forget(value);
		result
	}
}

//--------------------------------------------------------------------------------------------------

pub fn check_borrows(c_fail: usize, m_fail: usize) -> Result<(), ErrPack<TensorOpError>> {
	if (c_fail | m_fail) != 0 {
		cold_path();
		#[allow(clippy::redundant_else)]
		if m_fail == 0 {
			return Err(ErrPack {
				code: TensorOpError::CannotBorrow,
				extra: None,
			});
		} else {
			return Err(ErrPack {
				code: TensorOpError::CannotBorrowMut,
				extra: None,
			});
		}
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------

pub struct DeviceBufferRefMut<'a> {
	device_buffer: &'a DeviceBuffer,
}

impl<'a> DeviceBufferRefMut<'a> {
	pub fn new(device_buffer: &'a DeviceBuffer) -> Result<Self, BorrowMutError> {
		let read_count = device_buffer.read_count.get();
		let write_count = device_buffer.write_count.get();

		if (read_count | write_count) != 0 {
			cold_path();
			return Err(BorrowMutError);
		}

		device_buffer.write_count.set(1);
		Ok(Self { device_buffer })
	}

	pub unsafe fn new_unsafe(device_buffer: &'a DeviceBuffer, fail: &mut usize) -> Self {
		let read_count = device_buffer.read_count.get();
		let write_count = device_buffer.write_count.get();

		*fail |= read_count | write_count;

		device_buffer.write_count.set(write_count + 1);
		Self { device_buffer }
	}

	pub fn device_buffer(&self) -> &'a DeviceBuffer {
		self.device_buffer
	}
}

impl<'a> Drop for DeviceBufferRefMut<'a> {
	fn drop(&mut self) {
		let write_count = self.device_buffer.write_count.get();

		debug_assert!(write_count > 0, "DeviceBufferRefMut: invalid counter state");

		self.device_buffer.write_count.set(write_count - 1);
	}
}

impl<'a> std::ops::Deref for DeviceBufferRefMut<'a> {
	type Target = DeviceBuffer;

	#[inline]
	fn deref(&self) -> &DeviceBuffer {
		self.device_buffer
	}
}

//--------------------------------------------------------------------------------------------------
