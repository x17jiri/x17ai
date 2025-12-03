//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::alloc::AllocError;
use std::hint::{cold_path, likely};
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::new::device::{Device, DevicePtr};
use crate::tensor::{DType, HasDType, TensorOpError};
use crate::util::intrusive_rc::{self, IntrusiveRc, IntrusiveRcTrait};

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct ShapeHelper<'a> {
	dtype: DType,
	shape: &'a [usize],
	elems: usize,
}

pub enum ShapeOverflowError {
	NdimOverflow,
	ElemsOverflow,
	BitsOverflow,
}

impl<'a> ShapeHelper<'a> {
	pub fn new(dtype: DType, shape: &'a [usize]) -> Result<Self, ShapeOverflowError> {
		let Ok(_ndim) = TryInto::<u8>::try_into(shape.len()) else {
			cold_path();
			return Err(ShapeOverflowError::NdimOverflow);
		};
		let mut nonzero_elems: usize = 1;
		let mut elems: usize = 1;
		for &dim in shape {
			// When checking for overflow, we ignore size zero dimensions.
			// This is to make sure that overflow doesn't depend on the order of dimensions.
			// Example:
			// - [usize::MAX, usize::MAX, 0] // without ignoring zeros, this overflows
			// - [0, usize::MAX, usize::MAX] // without ignoring zeros, this doesn't overflow
			if likely(dim != 0) {
				let Some(t) = nonzero_elems.checked_mul(dim) else {
					cold_path();
					return Err(ShapeOverflowError::ElemsOverflow);
				};
				nonzero_elems = t;
			}
			elems *= dim;
		}
		let max_bits = usize::MAX - 7;
		let max_elems = max_bits / dtype.bits();
		if nonzero_elems > max_elems {
			cold_path();
			return Err(ShapeOverflowError::BitsOverflow);
		}
		Ok(ShapeHelper { dtype, shape, elems })
	}

	pub fn ndim(&self) -> u8 {
		// SAFETY: Checked in `new`.
		#[allow(clippy::cast_possible_truncation)]
		(self.shape.len() as u8)
	}

	pub fn shape(&self) -> &[usize] {
		self.shape
	}

	pub fn elems(&self) -> usize {
		self.elems
	}

	pub fn bits(&self) -> usize {
		self.elems * self.dtype.bits()
	}

	pub fn bytes(&self) -> usize {
		// SAFETY: We ensured in `new` that the addition cannot overflow.
		#[allow(clippy::manual_div_ceil)]
		((self.bits() + 7) / 8)
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}
}

//--------------------------------------------------------------------------------------------------

pub trait TensorLiteral {
	fn dtype(&self) -> DType;
	fn shape(&self) -> &[usize];
	fn data(&self) -> &[u8];
}

pub struct TensorLiteral1D<'a, T: HasDType> {
	shape: [usize; 1],
	data: &'a [T],
}

impl<'a, T: HasDType> TensorLiteral1D<'a, T> {
	pub fn new<const X: usize>(data: &'a [T; X]) -> Self {
		Self { shape: [X], data }
	}
}

impl<'a, T: HasDType> TensorLiteral for TensorLiteral1D<'a, T> {
	fn dtype(&self) -> DType {
		T::dtype
	}

	fn shape(&self) -> &[usize] {
		&self.shape
	}

	fn data(&self) -> &[u8] {
		unsafe {
			std::slice::from_raw_parts(
				self.data.as_ptr().cast::<u8>(),
				self.data.len() * std::mem::size_of::<T>(),
			)
		}
	}
}

pub struct TensorLiteral2D<'a, T: HasDType> {
	shape: [usize; 2],
	data: &'a [T],
}

impl<'a, T: HasDType> TensorLiteral2D<'a, T> {
	pub fn new<const Y: usize, const X: usize>(data: &'a [[T; X]; Y]) -> Self {
		let flat_data: &'a [T] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), X * Y) };
		Self { shape: [Y, X], data: flat_data }
	}
}

impl<'a, T: HasDType> TensorLiteral for TensorLiteral2D<'a, T> {
	fn dtype(&self) -> DType {
		T::dtype
	}

	fn shape(&self) -> &[usize] {
		&self.shape
	}

	fn data(&self) -> &[u8] {
		unsafe {
			std::slice::from_raw_parts(
				self.data.as_ptr().cast::<u8>(),
				self.data.len() * std::mem::size_of::<T>(),
			)
		}
	}
}

pub struct TensorLiteral3D<'a, T: HasDType> {
	shape: [usize; 3],
	data: &'a [T],
}

impl<'a, T: HasDType> TensorLiteral3D<'a, T> {
	pub fn new<const Z: usize, const Y: usize, const X: usize>(data: &'a [[[T; X]; Y]; Z]) -> Self {
		let flat_data: &'a [T] =
			unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), X * Y * Z) };
		Self { shape: [Z, Y, X], data: flat_data }
	}
}

impl<'a, T: HasDType> TensorLiteral for TensorLiteral3D<'a, T> {
	fn dtype(&self) -> DType {
		T::dtype
	}

	fn shape(&self) -> &[usize] {
		&self.shape
	}

	fn data(&self) -> &[u8] {
		unsafe {
			std::slice::from_raw_parts(
				self.data.as_ptr().cast::<u8>(),
				self.data.len() * std::mem::size_of::<T>(),
			)
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub const MIN_SHAPE_LEN: usize = 1;

#[repr(C)]
struct TensorData {
	dtype: DType,
	ndim: u8,
	_reserved: [u8; 3],

	refcount: intrusive_rc::RefCount,
	device_ptr: DevicePtr,
	elems: usize,

	// TODO
	// - could this be replaced with some sort of thin rc that stores metadata in the pointee
	//   not in the pointer itself? This would save 8 bytes per tensor.
	device: Rc<dyn Device>,

	shape: [usize; MIN_SHAPE_LEN], // flexible array member
}

impl IntrusiveRcTrait for TensorData {
	unsafe fn refcount(&self) -> &intrusive_rc::RefCount {
		&self.refcount
	}

	unsafe fn destroy(this: std::ptr::NonNull<Self>) {
		unsafe {
			let Self { device_ptr, device, .. } = this.read();
			device.drop_buffer(device_ptr);
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Tensor {
	data: IntrusiveRc<TensorData>,
}

impl Tensor {
	#[inline(never)]
	pub fn new(
		device: Rc<dyn Device>,
		literal: &dyn TensorLiteral,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let Ok(shape) = ShapeHelper::new(literal.dtype(), literal.shape()) else {
			cold_path();
			return Err(TensorOpError::ElementsOverflow.into());
		};
		let bytes = shape.bytes();
		unsafe {
			let Ok(device_ptr) = device.new_buffer(bytes) else {
				cold_path();
				return Err(TensorOpError::DevBufAllocFailed.into());
			};
			device.upload_data(NonNull::from_ref(literal.data()).cast(), device_ptr, bytes)?;
			match Self::with_buffer(device, device_ptr, shape) {
				Ok(tensor) => Ok(tensor),
				Err(device) => {
					cold_path();
					device.drop_buffer(device_ptr);
					std::mem::drop(device);
					Err(TensorOpError::DevBufAllocFailed.into())
				},
			}
		}
	}

	/// # Safety
	/// - device_ptr must be valid for device
	/// - size of buffer must be at least shape.bytes()
	///
	/// # Errors
	/// - gives back the consumed `device` if allocation fails
	#[inline(never)]
	pub unsafe fn with_buffer(
		device: Rc<dyn Device>,
		device_ptr: DevicePtr,
		shape: ShapeHelper,
	) -> Result<Self, Rc<dyn Device>> {
		const {
			assert!(MIN_SHAPE_LEN > 0);
		}
		unsafe {
			let ndim = shape.ndim();
			let struct_layout = std::alloc::Layout::new::<TensorData>();
			let shape_extra = (ndim as usize).saturating_sub(MIN_SHAPE_LEN);
			let layout = std::alloc::Layout::from_size_align_unchecked(
				struct_layout.size() + (shape_extra * std::mem::size_of::<usize>()),
				struct_layout.align(),
			);

			let Some(raw_ptr) = NonNull::new(std::alloc::alloc(layout)) else {
				cold_path();
				return Err(device);
			};

			let struct_ptr = raw_ptr.cast::<TensorData>();
			struct_ptr.write(TensorData {
				dtype: shape.dtype(),
				ndim,
				_reserved: [0; 3],
				refcount: intrusive_rc::RefCount::new(),
				device_ptr,
				elems: shape.elems(),
				device,
				shape: [0; MIN_SHAPE_LEN],
			});

			let shape_ptr = &raw mut (*struct_ptr.as_ptr()).shape[0];
			std::ptr::copy_nonoverlapping(shape.shape().as_ptr(), shape_ptr, ndim as usize);

			Ok(Self { data: IntrusiveRc::new(struct_ptr) })
		}
	}

	pub fn dtype(&self) -> DType {
		self.data.dtype
	}

	pub fn ndim(&self) -> usize {
		self.data.ndim as usize
	}

	pub fn shape(&self) -> &[usize] {
		unsafe {
			let struct_ptr = IntrusiveRc::as_ptr(&self.data);
			let shape_ptr = &raw const (*struct_ptr).shape[0];
			std::slice::from_raw_parts(shape_ptr, self.ndim())
		}
	}
}

//--------------------------------------------------------------------------------------------------
