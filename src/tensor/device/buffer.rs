//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::device::{DeviceBase, DeviceVMT};
use crate::tensor::generic::buffer::Buffer;
use crate::tensor::generic::map::{ND, SizeAndStride};
use crate::util::mycell;

use super::Device;
use super::dtype::DType;

//--------------------------------------------------------------------------------------------------

/// I use this helper struct to make sure that
/// `DeviceBufferVMT` can only be created via the unsafe `new()` function.
pub struct DeviceBufferVMTData {
	pub device: NonNull<dyn Device>,
	pub device_is_cpu: bool,
	pub dtype: DType,
	pub kernel_runner: Rc<KernelRunner>,

	pub drop_buffer: DropBufferFn,
	pub read_float: ReadFloatFn,
	pub load_from_cpu_memory: LoadFromCPUMemoryFn,
	pub store_to_cpu_memory: StoreToCPUMemoryFn,
	pub mm: MMFn,
	pub attention: AttentionFn,
	pub run_kernel: RunKernelFn,
}

impl std::ops::Deref for DeviceBufferVMT {
	type Target = DeviceBufferVMTData;

	#[inline]
	fn deref(&self) -> &DeviceBufferVMTData {
		&self.data
	}
}

impl DeviceBufferVMT {
	/// # Safety
	///
	/// - `device` must be a valid pointer that outlives `self`
	/// - calling the provided functions with pointer to `self` as `this` must be safe
	pub unsafe fn new(data: DeviceBufferVMTData) -> Self {
		Self { data }
	}

	/// # Safety
	/// - `T` must be a struct that has `DeviceBufferVMT` as its first field
	///
	/// Example:
	/// ```
	/// #[repr(C)]
	/// struct CPUFloatVMT {
	/// 	vmt: DeviceBufferVMT,
	/// 	...
	/// }
	/// ```
	#[inline]
	pub unsafe fn cast<T>(&self) -> &T {
		unsafe { &*NonNull::from_ref(self).as_ptr().cast::<T>() }
	}

	#[inline]
	pub fn device(&self) -> &dyn Device {
		unsafe { self.data.device.as_ref() }
	}

	#[inline]
	pub fn rc_device(&self) -> Rc<dyn Device> {
		unsafe {
			let device = self.data.device.as_ptr();
			Rc::increment_strong_count(device);
			Rc::from_raw(device)
		}
	}

	/// # Safety
	/// - `T` must be the actual type of the device
	#[inline]
	pub unsafe fn cast_device<T: Device>(&self) -> &T {
		let (device, _) = self.data.device.to_raw_parts();
		let device = device.cast();
		unsafe { device.as_ref() }
	}
}

pub struct DeviceBuffer {
	device_buffer: NonNull<u8>,
	dtype: DType,
	elems: usize,
	device_is_cpu: bool,
	device: NonNull<DeviceBase>,
}

impl DeviceBuffer {
	#[inline]
	pub unsafe fn new(
		device_data: NonNull<u8>,
		elems: usize,
		vmt: NonNull<DeviceBufferVMT>,
	) -> Self {
		Self { device_data, elems, vmt }
	}

	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let my_vmt = unsafe { self.vmt.as_ref() };
		let (my_dev, _) = my_vmt.device.to_raw_parts();

		let (dev, _) = NonNull::from_ref(device).to_raw_parts();

		my_dev == dev
	}

	#[inline]
	pub fn device_data(&self) -> NonNull<u8> {
		self.device_data
	}

	#[inline]
	pub fn elems(&self) -> usize {
		self.elems
	}

	#[inline]
	pub fn vmt(&self) -> &DeviceBufferVMT {
		unsafe { self.vmt.as_ref() }
	}

	#[inline]
	pub fn dtype(&self) -> DType {
		self.vmt().dtype
	}
}

impl Drop for DeviceBuffer {
	fn drop(&mut self) {
		unsafe {
			let vmt = self.vmt();
			(vmt.drop_buffer)(vmt, self.elems, self.device_data);
		}
	}
}

//--------------------------------------------------------------------------------------------------

impl Buffer for Rc<mycell::RefCell<DeviceBuffer>> {
	fn len(&self) -> usize {
		self.elems
	}
}

impl Buffer for &mycell::RefCell<DeviceBuffer> {
	fn len(&self) -> usize {
		self.elems
	}
}

impl<'a> Buffer for mycell::BorrowGuard<'a, DeviceBuffer> {
	fn len(&self) -> usize {
		self.elems
	}
}

impl<'a> Buffer for mycell::BorrowMutGuard<'a, DeviceBuffer> {
	fn len(&self) -> usize {
		self.elems
	}
}

//--------------------------------------------------------------------------------------------------
