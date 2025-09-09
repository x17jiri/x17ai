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

pub struct AttentionArgs {
	pub q_count: usize,
	pub head_count: usize,
	pub q_width: usize,
	pub q_offset: usize,
	pub q_item_stride: usize,
	pub q_head_stride: usize,
	pub q: *const u8, // [q_count, head_count, q_width]

	pub k_count: usize,
	pub group_shift: usize,
	// k_width == q_width
	pub k_offset: usize,
	pub k_item_stride: usize,
	pub k_head_stride: usize,
	pub k: *const u8, // [kv_count, head_count >> group_shift, q_width]

	// v_count == k_count
	// v_head_count == head_count >> group_shift
	pub v_width: usize,
	pub v_offset: usize,
	pub v_item_stride: usize,
	pub v_head_stride: usize,
	pub v: *const u8, // [kv_count, head_count >> group_shift, v_width]

	// o_count == q_count
	// o_head_count == head_count
	// o_width == v_width
	pub o_offset: usize,
	pub o_head_stride: usize,
	pub o_item_stride: usize,
	pub o: *mut u8, // [q_count, head_count, v_width]
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
	unsafe fn(this: NonNull<DeviceBufferVMT>, elems: usize, device_data: *mut u8);

pub type ReadFloatFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	src: &GenericTensor<ND<0>, BorrowGuard<'buf, DeviceBuffer>>,
) -> Result<f64, ErrPack<TensorOpError>>;

pub type LoadFromCPUMemoryFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	src: &[u8],
	dst: &mut GenericTensor<ND<1>, BorrowMutGuard<'buf, DeviceBuffer>>,
) -> Result<(), ErrPack<TensorOpError>>;

pub type StoreToCPUMemoryFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	src: &GenericTensor<ND<1>, BorrowGuard<'buf, DeviceBuffer>>,
	dst: &mut [u8],
) -> Result<(), ErrPack<TensorOpError>>;

pub type MMFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	o: &mut GenericTensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
	a: &GenericTensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
	b: &GenericTensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
	scale: f64,
) -> Result<(), ErrPack<TensorOpError>>;

pub type AttentionFn = for<'buf> unsafe fn(
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
	read_float: ReadFloatFn,
	load_from_cpu_memory: LoadFromCPUMemoryFn,
	store_to_cpu_memory: StoreToCPUMemoryFn,
	mm: MMFn,
	attention: AttentionFn,
	run_kernel: RunKernelFn,
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
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	#[inline]
	pub fn kernel_runner(&self) -> &KernelRunner {
		&self.kernel_runner
	}

	#[inline]
	pub fn read_float<'buf>(
		&self,
		src: &GenericTensor<ND<0>, BorrowGuard<'buf, DeviceBuffer>>,
	) -> Result<f64, ErrPack<TensorOpError>> {
		unsafe { (self.read_float)(self.into(), src) }
	}

	#[inline]
	pub fn load_from_cpu_memory<'buf>(
		&self,
		src: &[u8],
		dst: &mut GenericTensor<ND<1>, BorrowMutGuard<'buf, DeviceBuffer>>,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { (self.load_from_cpu_memory)(self.into(), src, dst) }
	}

	#[inline]
	pub fn store_to_cpu_memory<'buf>(
		&self,
		src: &GenericTensor<ND<1>, BorrowGuard<'buf, DeviceBuffer>>,
		dst: &mut [u8],
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { (self.store_to_cpu_memory)(self.into(), src, dst) }
	}

	#[inline]
	pub fn mm<'buf>(
		&self,
		o: &mut GenericTensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
		a: &GenericTensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		b: &GenericTensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		scale: f64,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { (self.mm)(self.into(), o, a, b, scale) }
	}

	#[inline]
	pub fn attention<'buf>(&self, args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { (self.attention)(self.into(), args) }
	}

	#[inline]
	pub unsafe fn run_kernel(
		&self,
		kernel_data: &KernelData,
		o: *const KernelOutput,
		elemwise_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		scalar_args: *const f64,
		reduction_size: usize,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			(self.run_kernel)(
				self.into(),
				kernel_data,
				o,
				elemwise_args,
				reduce_args,
				scalar_args,
				reduction_size,
			)
		}
	}
}

pub struct DeviceBuffer {
	device_data: *mut u8,
	elems: usize,
	vmt: NonNull<DeviceBufferVMT>,
}

impl DeviceBuffer {
	#[inline]
	pub unsafe fn new(device_data: *mut u8, elems: usize, vmt: NonNull<DeviceBufferVMT>) -> Self {
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
	pub fn device_data(&self) -> *mut u8 {
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
