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
use crate::tensor::generic::buffer::Buffer;
use crate::tensor::generic::map::ND;
use crate::tensor::generic::{self};
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

//--------------------------------------------------------------------------------------------------

pub type DropBufferFn =
	unsafe fn(this: NonNull<DeviceBufferVMT>, elems: usize, device_data: *mut u8);

pub type ReadFloatFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	src: &generic::Tensor<ND<0>, BorrowGuard<'buf, DeviceBuffer>>,
) -> Result<f64, ErrPack<TensorOpError>>;

pub type LoadBinFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	dst: &mut generic::Tensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
	src: &mut dyn std::io::Read,
) -> Result<(), ErrPack<TensorOpError>>;

pub type StoreBinFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	src: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
	dst: &mut dyn std::io::Write,
) -> Result<(), ErrPack<TensorOpError>>;

pub type MMFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	o: &mut generic::Tensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
	a: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
	b: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
	scale: f64,
) -> Result<(), ErrPack<TensorOpError>>;

pub type AttentionFn = for<'buf> unsafe fn(
	this: NonNull<DeviceBufferVMT>,
	o: &mut generic::Tensor<ND<3>, BorrowMutGuard<'buf, DeviceBuffer>>, /* [inputs, qo_heads,
	                                                                     * vo_features] */
	q: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>, // [inputs, qo_heads, qk_features]
	k: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>, // [inputs, k_heads, qk_features]
	v: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>, // [inputs, v_heads, vo_features]
);

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
	load_bin: LoadBinFn,
	store_bin: StoreBinFn,
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
		load_bin: LoadBinFn,
		store_bin: StoreBinFn,
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
			load_bin,
			store_bin,
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
		src: &generic::Tensor<ND<0>, BorrowGuard<'buf, DeviceBuffer>>,
	) -> Result<f64, ErrPack<TensorOpError>> {
		unsafe { (self.read_float)(self.into(), src) }
	}

	#[inline]
	pub fn load_bin<'buf>(
		&self,
		dst: &mut generic::Tensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
		src: &mut dyn std::io::Read,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { (self.load_bin)(self.into(), dst, src) }
	}

	#[inline]
	pub fn store_bin<'buf>(
		&self,
		src: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		dst: &mut dyn std::io::Write,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { (self.store_bin)(self.into(), src, dst) }
	}

	#[inline]
	pub fn mm<'buf>(
		&self,
		o: &mut generic::Tensor<ND<2>, BorrowMutGuard<'buf, DeviceBuffer>>,
		a: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		b: &generic::Tensor<ND<2>, BorrowGuard<'buf, DeviceBuffer>>,
		scale: f64,
	) -> Result<(), ErrPack<TensorOpError>> {
		unsafe { (self.mm)(self.into(), o, a, b, scale) }
	}

	#[inline]
	pub fn attention<'buf>(
		&self,
		o: &mut generic::Tensor<ND<3>, BorrowMutGuard<'buf, DeviceBuffer>>,
		q: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>,
		k: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>,
		v: &generic::Tensor<ND<3>, BorrowGuard<'buf, DeviceBuffer>>,
	) {
		unsafe { (self.attention)(self.into(), o, q, k, v) }
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
