//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;
use std::rc::Rc;

pub mod buffer;
pub mod cpu;
pub mod cuda;
pub mod dtype;
pub mod kernel;

pub use buffer::DeviceBuffer;
pub use dtype::{DType, HasDType};

use crate::ErrPack;
use crate::tensor::TensorOpError;
use crate::tensor::device::kernel::DynKernelCall;
use crate::tensor::generic::map::{ND, SizeAndStride};
use crate::util::mycell;

//--------------------------------------------------------------------------------------------------

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DevicePtr {
	ptr: *mut (),
}

impl DevicePtr {
	#[inline]
	pub fn new(ptr: *mut ()) -> Self {
		Self { ptr }
	}

	/// For a CPU device, `self.ptr` is probably just a pointer to the memory.
	/// For other devices, it could be the device pointer casted to `*mut ()`, or
	/// even some handle that can be used to get the device pointer.
	///
	/// Since it can be a casted device pointer, dereferencing it may be undefined behavior.
	///
	/// And since it can be some handle, pointer arithmetic on it may not make sense.
	///
	/// These operations should only be done by device-specific code that knows what the pointer is.
	#[inline]
	pub unsafe fn as_ptr<T>(&self) -> *mut T {
		self.ptr.cast::<T>()
	}
}

#[repr(C)]
pub struct KernelElemArg {
	pub stride_bytes: [usize; 2],
	pub buf: DevicePtr,
	pub offset_bytes: usize,
}

#[repr(C)]
pub struct KernelReduceArg {
	pub stride_bytes: [usize; 3],
	pub buf: DevicePtr,
	pub offset_bytes: usize,
}

#[repr(C)]
pub struct KernelOutput {
	pub size: [usize; 2],
	pub stride_bytes: [usize; 2],
	pub buf: DevicePtr,
	pub offset_bytes: usize,
	pub reduction_size: usize,
}

#[repr(C)]
pub struct MatMulArgs {
	pub o_row_stride: usize,
	pub o_col_stride: usize,
	pub o_rows: usize,
	pub o_cols: usize,
	pub o_offset: usize,
	pub o_buf: DevicePtr, // [o_rows, o_cols]

	pub a_row_stride: usize,
	pub a_col_stride: usize,
	// a_rows == o_rows
	pub a_cols: usize,
	pub a_offset: usize,
	pub a_buf: DevicePtr, // [o_rows, a_cols]

	pub b_row_stride: usize,
	pub b_col_stride: usize,
	// b_rows == a_cols
	// b_cols == o_cols
	pub b_offset: usize,
	pub b_buf: DevicePtr, // [a_cols, o_cols]

	pub o_dtype: DType,
	pub a_dtype: DType,
	pub b_dtype: DType,
	pub internal_dtype: DType,

	pub o_buf_elems: usize,
	pub a_buf_elems: usize,
	pub b_buf_elems: usize,
}

#[repr(C)]
pub struct AttentionArgs {
	pub q_count: usize,
	pub head_count: usize,
	pub q_width: usize,
	pub q_offset: usize,
	pub q_item_stride: usize,
	pub q_head_stride: usize,
	pub q: DevicePtr, // [q_count, head_count, q_width]

	pub k_count: usize,
	pub group_shift: usize,
	// k_width == q_width
	pub k_offset: usize,
	pub k_item_stride: usize,
	pub k_head_stride: usize,
	pub k: DevicePtr, // [kv_count, head_count >> group_shift, q_width]

	// v_count == k_count
	// v_head_count == head_count >> group_shift
	pub v_width: usize,
	pub v_offset: usize,
	pub v_item_stride: usize,
	pub v_head_stride: usize,
	pub v: DevicePtr, // [kv_count, head_count >> group_shift, v_width]

	// o_count == q_count
	// o_head_count == head_count
	// o_width == v_width
	pub o_offset: usize,
	pub o_head_stride: usize,
	pub o_item_stride: usize,
	pub o: DevicePtr, // [q_count, head_count, v_width]

	pub q_buf_elems: usize,
	pub k_buf_elems: usize,
	pub v_buf_elems: usize,
	pub o_buf_elems: usize,
	pub dtype: DType,
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NewDeviceBufferError {
	UnsupportedDType,
	AllocationFailed,
}

//--------------------------------------------------------------------------------------------------

pub trait Device {
	fn name(&self) -> &str;

	fn is_cpu(&self) -> bool {
		false
	}

	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<mycell::RefCell<DeviceBuffer>>, NewDeviceBufferError>;

	/// # Safety
	/// This should only be called from `DeviceBuffer::drop()`
	unsafe fn drop_buffer(&self, device_ptr: DevicePtr, dtype: DType, elems: usize);

	unsafe fn read_float(
		&self,
		buf: &DeviceBuffer,
		offset: usize,
	) -> Result<f64, ErrPack<TensorOpError>>;

	unsafe fn upload_data(
		&self,
		src: NonNull<u8>,
		dst: &DeviceBuffer,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>>;

	unsafe fn download_data(
		&self,
		src: &DeviceBuffer,
		dst: NonNull<u8>,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>>;

	unsafe fn mm(&self, args: &MatMulArgs, scale: f64) -> Result<(), ErrPack<TensorOpError>>;

	unsafe fn attention(&self, args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>>;

	unsafe fn run_kernel(&self, data: &DynKernelCall) -> Result<(), ErrPack<TensorOpError>>;
}
