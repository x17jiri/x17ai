//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::alloc::AllocError;
use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::device::{DeviceBuffer, DevicePtr};
use crate::tensor::shape::Shape;
use crate::tensor::{DType, Device, HasDType};
use crate::util::intrusive_rc::{self, IntrusiveRc, IntrusiveRcTrait};

//--------------------------------------------------------------------------------------------------

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
			return Err(ShapeOverflowError::NdimOverflow);
		};
		let mut elems: usize = 1;
		let mut nonzero_elems: usize = 1;
		for &dim in shape {
			if dim != 0 {
				let Some(t) = nonzero_elems.checked_mul(dim) else {
					cold_path();
					return Err(ShapeOverflowError::ElemsOverflow);
				};
				nonzero_elems = t;
			}
			elems *= dim;
		}
		let Some(nonzero_bits) = nonzero_elems.checked_mul(dtype.bits()) else {
			cold_path();
			return Err(ShapeOverflowError::BitsOverflow);
		};
		let Some(_nonzero_bits) = nonzero_bits.checked_add(7) else {
			cold_path();
			return Err(ShapeOverflowError::BitsOverflow);
		};
		Ok(ShapeHelper { dtype, shape, elems })
	}

	pub fn ndim(&self) -> u8 {
		self.shape.len() as u8
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
		(self.bits() + 7) / 8
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}
}

//--------------------------------------------------------------------------------------------------

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
		unsafe {
			let struct_layout = std::alloc::Layout::new::<TensorData>();
			let shape_layout = std::alloc::Layout::array::<usize>(shape.ndim() as usize).unwrap();
			let (layout, shape_offset) = struct_layout.extend(shape_layout).unwrap();

			let Some(raw_ptr) = NonNull::new(std::alloc::alloc(layout)) else {
				cold_path();
				return Err(AllocError);
			};
			let struct_ptr = raw_ptr.cast::<TensorData>();

			struct_ptr.write(TensorData {
				dtype: shape.dtype(),
				ndim: shape.ndim(),
				_reserved: [0; 3],
				refcount: intrusive_rc::RefCount::new(),
				device_ptr,
				elems: shape.elems(),
				device,
			});

			let shape_ptr = raw_ptr.as_ptr().add(shape_offset).cast::<usize>();
			std::ptr::copy_nonoverlapping(shape.shape().as_ptr(), shape_ptr, shape.ndim() as usize);

			Ok(Self { data: IntrusiveRc::new(struct_ptr) })
		}
	}

	pub fn dtype(&self) -> DType {
		self.data.dtype
	}

	pub fn ndim(&self) -> usize {
		self.data.ndim as usize
	}

	#[inline(never)]
	pub fn shape(&self) -> &[usize] {
		unsafe {
			let struct_layout = std::alloc::Layout::new::<TensorData>();
			let shape_layout = std::alloc::Layout::array::<usize>(self.ndim()).unwrap();
			let (_layout, shape_offset) = struct_layout.extend(shape_layout).unwrap();

			let struct_ptr = IntrusiveRc::as_ptr(&self.data);
			let raw_ptr = struct_ptr.cast::<u8>();
			let shape_ptr = raw_ptr.add(shape_offset).cast::<usize>();
			std::slice::from_raw_parts(shape_ptr, self.ndim())
		}
	}
}

//--------------------------------------------------------------------------------------------------
