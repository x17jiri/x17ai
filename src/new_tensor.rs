//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::tensor::device::{DeviceBuffer, DevicePtr};
use crate::tensor::shape::Shape;
use crate::tensor::{DType, Device, HasDType};
use crate::util::intrusive_rc::{self, IntrusiveRc, IntrusiveRcTrait};

//--------------------------------------------------------------------------------------------------

const INLINE_DIMS: usize = 5;

pub struct ShapeHelper<'a> {
	dtype: DType,
	shape: &'a [usize],
	bytes: usize,
}

impl<'a> ShapeHelper<'a> {
	pub fn new(dtype: DType, shape: &'a [usize]) -> Result<Self, ()> {
		let Ok(_ndim) = TryInto::<u8>::try_into(shape.len()) else {
			return Err(()); // too many dimensions
		};
		let mut bits = dtype.bits();
		let mut nonzero_bits = dtype.bits();
		for &dim in shape {
			if dim != 0 {
				let Some(new) = nonzero_bits.checked_mul(dim) else {
					return Err(()); // overflow
				};
				nonzero_bits = new;
			}
			bits *= dim;
		}
		let Some(_nonzero_bits) = nonzero_bits.checked_add(7) else {
			return Err(()); // overflow
		};
		let bytes = (bits + 7) / 8;
		Ok(ShapeHelper { dtype, shape, bytes })
	}

	pub fn ndim(&self) -> u8 {
		self.shape.len() as u8
	}
}

//--------------------------------------------------------------------------------------------------

#[repr(C)]
struct TensorData {
	dtype: DType,
	ndim: u8,
	_reserved: [u8; 3],
	shape: NonNull<usize>,
	inline_shape: [usize; INLINE_DIMS],

	refcount: intrusive_rc::RefCount,
	device_ptr: DevicePtr,
	bytes: usize,

	// TODO
	// - could this be replaced with some sort of thin rc that stores metadata in the pointee
	//   not in the pointer itself? This would save 8 bytes per buffer.
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
	pub fn new(device: Rc<dyn Device>, device_ptr: DevicePtr, bytes: usize) -> Self {
		let mut instance = Box::new(TensorData {
			dtype: f32::dtype,
			ndim: 1,
			_reserved: [0; 3],
			shape: NonNull::dangling(),
			inline_shape: [0; INLINE_DIMS],

			refcount: intrusive_rc::RefCount::new(),
			device_ptr,
			bytes,

			device,
		});
		instance.as_mut().shape = NonNull::from_mut(&mut instance.inline_shape[0]);
		Tensor {
			data: unsafe { IntrusiveRc::from_box(instance) },
		}
	}

	pub fn set_shape<'a>(&mut self, dtype: DType, shape: ShapeHelper<'a>) -> Result<(), ()> {
		let Some(m) = self.data.get_mut() else {
			return Err(()); // cannot borrow
		};

		if (m.ndim as usize) > INLINE_DIMS {
			cold_path();
			unsafe {
				let layout = std::alloc::Layout::array::<usize>(m.ndim as usize).unwrap_unchecked();
				std::alloc::dealloc(m.shape.as_ptr().cast(), layout);
			}
			m.inline_shape[0] = 0;
			m.ndim = 1;
			m.shape = NonNull::from_mut(&mut m.inline_shape[0]);
		}
		debug_assert!(m.shape == NonNull::from_mut(&mut m.inline_shape[0]));

		if (ndim as usize) > INLINE_DIMS {
			cold_path();
			let shape: Option<NonNull<usize>> = unsafe {
				let layout = std::alloc::Layout::array::<usize>(shape.len()).unwrap_unchecked();
				NonNull::new(std::alloc::alloc(layout).cast())
			};
			let Some(shape) = shape else {
				return Err(());
			};
			m.shape = shape;
		}
		m.dtype = dtype;
		m.ndim = ndim;
		Ok(())
	}

	pub fn dtype(&self) -> DType {
		self.data.dtype
	}

	pub fn ndim(&self) -> usize {
		self.data.ndim as usize
	}

	pub fn shape(&self) -> &[usize] {
		unsafe { std::slice::from_raw_parts(self.data.shape.as_ptr(), self.data.ndim as usize) }
	}
}

//--------------------------------------------------------------------------------------------------
