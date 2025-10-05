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
use crate::tensor::device::dtype::DTypeId;
use crate::tensor::device::kernel::expr::{ExprToDyn, KEY_TYPE_SIZE, KernelKeyType};
use crate::tensor::device::kernel::runner::{KernelData, KernelRunner};
use crate::tensor::generic::map::{ND, SizeAndStride};
use crate::tensor::{Tensor, TensorOpError};
use crate::util::mycell;

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct KernelElemArg {
	pub stride_bytes: [usize; 2],
	pub buf: NonNull<u8>, // TODO - is this FFI safe?
	pub offset_bytes: usize,
}

#[repr(C)]
pub struct KernelReduceArg {
	pub stride_bytes: [usize; 3],
	pub buf: NonNull<u8>,
	pub offset_bytes: usize,
}

#[repr(C)]
pub struct KernelOutput {
	pub size: [usize; 2],
	pub stride_bytes: [usize; 2],
	pub buf: NonNull<u8>,
	pub offset_bytes: usize,
	pub reduction_size: usize,
}

pub struct KernelData<'a> {
	key: &'a [KernelKeyType],
	expr: &'a dyn ExprToDyn,
	output: &'a KernelOutput,
	elemwise_args: &'a [KernelElemArg],
	reduce_args: &'a [KernelReduceArg],
	scalar_args: &'a [f64],
}

impl<'a> KernelData<'a> {
	pub unsafe fn elemwise_args(&self) -> &'a [KernelElemArg] {
		self.elemwise_args
	}
	pub unsafe fn reduce_args(&self) -> &'a [KernelReduceArg] {
		self.reduce_args
	}
	pub unsafe fn scalar_args(&self) -> &'a [f64] {
		self.scalar_args
	}

	pub fn elemwise_count(&self) -> usize {
		self.elemwise_args.len()
	}
	pub fn reduce_count(&self) -> usize {
		self.reduce_args.len()
	}
	pub fn scalar_count(&self) -> usize {
		self.scalar_args.len()
	}

	pub const fn dtype_config_items(E: usize, R: usize) -> usize {
		(2 + E + R)
	}
	pub const fn dtype_config_words(E: usize, R: usize) -> usize {
		(Self::dtype_config_items(E, R) * std::mem::size_of::<DTypeId>())
			.next_multiple_of(KEY_TYPE_SIZE)
			/ KEY_TYPE_SIZE
	}
	pub fn my_dtype_config_words(&self) -> usize {
		Self::dtype_config_words(self.elemwise_count(), self.reduce_count())
	}

	pub fn new_dtype_config<const E: usize, const R: usize>(
		internal_dtype: DType,
		output: &Tensor,
		elem_args: [&Tensor; E],
		reduce_args: [&Tensor; R],
	) -> [u64; Self::dtype_config_words(E, R)] {
		let mut result = [0; Self::dtype_config_words(E, R)];
		let ptr = result.as_mut_ptr().cast::<DTypeId>();
		unsafe {
			*ptr = internal_dtype.id();
			*ptr.add(1) = output.dtype().id();
			for i in 0..E {
				*ptr.add(2 + i) = elem_args[i].dtype().id();
			}
			for i in 0..R {
				*ptr.add(2 + E + i) = reduce_args[i].dtype().id();
			}
		}
		result
	}

	pub fn internal_dtype(&self) -> DType {
		let dtype_config = self.key as *const [u64];
		let dtype_config = dtype_config.cast::<u64>().cast::<DTypeId>();
		unsafe { *dtype_config }.to_dtype()
	}
	pub fn output_dtype(&self) -> DType {
		let dtype_config = self.key as *const [u64];
		let dtype_config = dtype_config.cast::<u64>().cast::<DTypeId>();
		unsafe { *dtype_config.add(1) }.to_dtype()
	}
	pub fn elemwise_dtype(&self, i: usize) -> DType {
		assert!(i < self.elemwise_count());
		let dtype_config = self.key as *const [u64];
		let dtype_config = dtype_config.cast::<u64>().cast::<DTypeId>();
		unsafe { *dtype_config.add(2 + i) }.to_dtype()
	}
	pub fn reduce_dtype(&self, i: usize) -> DType {
		assert!(i < self.reduce_count());
		let dtype_config = self.key as *const [u64];
		let dtype_config = dtype_config.cast::<u64>().cast::<DTypeId>();
		unsafe { *dtype_config.add(2 + self.elemwise_count() + i) }.to_dtype()
	}
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

pub struct DeviceBase {
	pub kernel_runner: Rc<KernelRunner>,
	pub is_cpu: bool,
}

impl DeviceBase {
	#[inline]
	pub fn from_device(device: &dyn Device) -> &Self {
		// SAFETY: `device` implements `DerivesDeviceBase`, so it must be properly aligned and
		// `DeviceBase` must be its first field.
		let device = device as *const dyn Device;
		#[allow(clippy::cast_ptr_alignment)]
		let device = device.cast::<Self>();
		unsafe { &*device }
	}
}

/// # Safety
/// - Any type implementing this trait must have `DeviceBase` as its first field.
/// - Alignment of the type must be compatible with `DeviceBase`. So don't use `repr(packed)`.
///
/// The way to ensure this is:
/// ```rust
/// #[repr(C)]
/// struct MyDevice {
///     base: DeviceBase,
///     // other fields...
/// }
/// unsafe impl DerivesDeviceBase for MyDevice {}
/// ```
pub unsafe trait DerivesDeviceBase {}

pub trait Device: DerivesDeviceBase {
	fn name(&self) -> &str;

	fn new_buffer(
		self: Rc<Self>,
		dtype: DType,
		elems: usize,
	) -> Result<Rc<mycell::RefCell<DeviceBuffer>>, NewDeviceBufferError>;

	/// # Safety
	/// This should only be called from `DeviceBuffer::drop()`
	unsafe fn drop_buffer(&self, memory: NonNull<u8>, dtype: DType, elems: usize);

	unsafe fn read_float(
		&self,
		buf: &DeviceBuffer,
		offset: usize,
	) -> Result<f64, ErrPack<TensorOpError>>;

	unsafe fn load_from_cpu_memory(
		&self,
		cpu_src: NonNull<u8>,
		dev_dst: &DeviceBuffer,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>>;

	unsafe fn store_to_cpu_memory(
		&self,
		dev_src: &DeviceBuffer,
		cpu_dst: NonNull<u8>,
		offset_bytes: usize,
		count_bytes: usize,
	) -> Result<(), ErrPack<TensorOpError>>;

	unsafe fn mm(&self, args: &MatMulArgs, scale: f64) -> Result<(), ErrPack<TensorOpError>>;

	unsafe fn attention(&self, args: &AttentionArgs) -> Result<(), ErrPack<TensorOpError>>;

	unsafe fn run_kernel(
		&self,
		kernel_data: &KernelData,
		o: &KernelOutput,
		elemwise_args: *const KernelElemArg,
		reduce_args: *const KernelReduceArg,
		scalar_args: *const f64,
		dtype_config: *const u64,
	) -> Result<(), ErrPack<TensorOpError>>;
}
