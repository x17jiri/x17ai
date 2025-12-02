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

use crate::tensor::device::DevicePtr;
use crate::tensor::{DType, Device};
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

	unsafe fn destroy(_this: std::ptr::NonNull<Self>) {
		todo!() // TODO
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
		device_ptr: DevicePtr,
		shape: ShapeHelper,
	) -> Result<Self, AllocError> {
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
				return Err(AllocError);
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
