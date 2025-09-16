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
use crate::tensor::generic::GenericTensor;
use crate::tensor::generic::buffer::Buffer;
use crate::tensor::generic::map::{ND, SizeAndStride};
use crate::util::mycell::{self, BorrowGuard, BorrowMutGuard};

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
	pub stride_bytes: [usize; 3],
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

#[repr(C)]
pub struct MatMulArgs {
	pub o_row_stride: usize,
	pub o_col_stride: usize,
	pub o_rows: usize,
	pub o_cols: usize,
	pub o_offset: usize,
	pub o_buf: NonNull<u8>, // [o_rows, o_cols]

	pub a_row_stride: usize,
	pub a_col_stride: usize,
	// a_rows == o_rows
	pub a_cols: usize,
	pub a_offset: usize,
	pub a_buf: NonNull<u8>, // [o_rows, a_cols]

	pub b_row_stride: usize,
	pub b_col_stride: usize,
	// b_rows == a_cols
	// b_cols == o_cols
	pub b_offset: usize,
	pub b_buf: NonNull<u8>, // [a_cols, o_cols]

	pub scale: f64,
}

#[repr(C)]
pub struct AttentionArgs {
	pub q_count: usize,
	pub head_count: usize,
	pub q_width: usize,
	pub q_offset: usize,
	pub q_item_stride: usize,
	pub q_head_stride: usize,
	pub q: NonNull<u8>, // [q_count, head_count, q_width]

	pub k_count: usize,
	pub group_shift: usize,
	// k_width == q_width
	pub k_offset: usize,
	pub k_item_stride: usize,
	pub k_head_stride: usize,
	pub k: NonNull<u8>, // [kv_count, head_count >> group_shift, q_width]

	// v_count == k_count
	// v_head_count == head_count >> group_shift
	pub v_width: usize,
	pub v_offset: usize,
	pub v_item_stride: usize,
	pub v_head_stride: usize,
	pub v: NonNull<u8>, // [kv_count, head_count >> group_shift, v_width]

	// o_count == q_count
	// o_head_count == head_count
	// o_width == v_width
	pub o_offset: usize,
	pub o_head_stride: usize,
	pub o_item_stride: usize,
	pub o: NonNull<u8>, // [q_count, head_count, v_width]
}

#[rustfmt::skip]
impl AttentionArgs {
	pub fn q_map(&self) -> ND<3> {
		ND {
			dims: [
				SizeAndStride { size: self.q_count, stride: self.q_item_stride },
				SizeAndStride { size: self.head_count, stride: self.q_head_stride },
				SizeAndStride { size: self.q_width, stride: 1 },
			],
			offset: self.q_offset,
		}
	}
	pub fn k_map(&self) -> ND<3> {
		ND {
			dims: [
				SizeAndStride { size: self.k_count, stride: self.k_item_stride },
				SizeAndStride { size: self.head_count >> self.group_shift, stride: self.k_head_stride },
				SizeAndStride { size: self.q_width, stride: 1 },
			],
			offset: self.k_offset,
		}
	}
	pub fn v_map(&self) -> ND<3> {
		ND {
			dims: [
				SizeAndStride { size: self.k_count, stride: self.v_item_stride },
				SizeAndStride { size: self.head_count >> self.group_shift, stride: self.v_head_stride },
				SizeAndStride { size: self.v_width, stride: 1 },
			],
			offset: self.v_offset,
		}
	}
	pub fn o_map(&self) -> ND<3> {
		ND {
			dims: [
				SizeAndStride { size: self.q_count, stride: self.o_item_stride },
				SizeAndStride { size: self.head_count, stride: self.o_head_stride },
				SizeAndStride { size: self.v_width, stride: 1 },
			],
			offset: self.o_offset,
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub type DropBufferFn =
	unsafe fn(this: NonNull<DeviceBufferVMT>, elems: usize, device_data: NonNull<u8>);

pub type ReadFloatFn = unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	dev_src: (ND<0>, &DeviceBuffer),
) -> Result<f64, ErrPack<TensorOpError>>;

pub type LoadFromCPUMemoryFn = unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	cpu_src: NonNull<u8>,
	dev_dst: (ND<0>, &DeviceBuffer),
	count: usize,
) -> Result<(), ErrPack<TensorOpError>>;

pub type StoreToCPUMemoryFn = unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	dev_src: (ND<0>, &DeviceBuffer),
	cpu_dst: NonNull<u8>,
	count: usize,
) -> Result<(), ErrPack<TensorOpError>>;

pub type MMFn = unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	args: &MatMulArgs,
) -> Result<(), ErrPack<TensorOpError>>;

pub type AttentionFn = unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	args: &AttentionArgs,
) -> Result<(), ErrPack<TensorOpError>>;

pub type RunKernelFn = unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	kernel_data: &KernelData,
	o: *const KernelOutput,
	elemwise_args: *const KernelElemArg,
	reduce_args: *const KernelReduceArg,
	scalar_args: *const f64,
	reduction_size: usize,
) -> Result<(), ErrPack<TensorOpError>>;

pub struct DeviceBufferVMT {
	device: NonNull<dyn Device>,
	device_is_cpu: bool,
	dtype: DType,
	kernel_runner: Rc<KernelRunner>,

	drop_buffer: DropBufferFn,
	pub read_float: ReadFloatFn,
	pub load_from_cpu_memory: LoadFromCPUMemoryFn,
	pub store_to_cpu_memory: StoreToCPUMemoryFn,
	pub mm: MMFn,
	pub attention: AttentionFn,
	pub run_kernel: RunKernelFn,
}

impl DeviceBufferVMT {
	/// # Safety
	///
	/// - `device` must be a valid pointer that outlives `self`
	/// - calling the provided functions with pointer to `self` as `this` must be safe
	#[allow(clippy::too_many_arguments)]
	pub unsafe fn new(
		device: NonNull<dyn Device>,
		device_is_cpu: bool,
		dtype: DType,
		kernel_runner: Rc<KernelRunner>,

		drop_buffer: DropBufferFn,
		read_float: ReadFloatFn,
		load_from_cpu_memory: LoadFromCPUMemoryFn,
		store_to_cpu_memory: StoreToCPUMemoryFn,
		mm: MMFn,
		attention: AttentionFn,
		run_kernel: RunKernelFn,
	) -> Self {
		Self {
			device,
			device_is_cpu,
			dtype,
			kernel_runner,

			drop_buffer,
			read_float,
			load_from_cpu_memory,
			store_to_cpu_memory,
			mm,
			attention,
			run_kernel,
		}
	}

	#[inline]
	pub fn device(&self) -> &dyn Device {
		unsafe { self.device.as_ref() }
	}

	#[inline]
	pub fn rc_device(&self) -> Rc<dyn Device> {
		unsafe {
			let device = self.device.as_ptr();
			Rc::increment_strong_count(device);
			Rc::from_raw(device)
		}
	}

	#[inline]
	pub fn device_ptr(&self) -> NonNull<dyn Device> {
		self.device
	}

	#[inline]
	pub fn device_is_cpu(&self) -> bool {
		self.device_is_cpu
	}

	#[inline]
	pub unsafe fn cast_device<T: Device>(&self) -> &T {
		let (device, _) = self.device_ptr().to_raw_parts();
		let device = device.cast();
		unsafe { device.as_ref() }
	}

	#[inline]
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	#[inline]
	pub fn kernel_runner(&self) -> &KernelRunner {
		&self.kernel_runner
	}
}

pub struct DeviceBuffer {
	device_data: NonNull<u8>,
	elems: usize,
	vmt: NonNull<DeviceBufferVMT>,
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
			(self.vmt.as_ref().drop_buffer)(self.vmt, self.elems, self.device_data);
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
