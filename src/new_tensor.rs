//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::ptr::NonNull;

use crate::tensor::DType;
use crate::tensor::device::DeviceBuffer;
use crate::util::intrusive_rc::IntrusiveRc;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ShapeLen(u32);

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::len_without_is_empty)]
#[allow(clippy::missing_panics_doc)]
impl ShapeLen {
	pub fn new_inline(len: usize) -> Self {
		assert!(len <= u32::MAX as usize / 2);
		let len = len as u32;
		Self(len << 1)
	}

	pub fn new_heap(len: usize) -> Self {
		assert!(len <= u32::MAX as usize / 2);
		let len = len as u32;
		Self((len << 1) | 1)
	}

	pub fn is_inline(&self) -> bool {
		(self.0 & 1) == 0
	}

	pub fn is_heap(&self) -> bool {
		(self.0 & 1) != 0
	}

	pub fn get(&self) -> usize {
		(self.0 as usize) >> 1
	}
}

//--------------------------------------------------------------------------------------------------

const INLINE_DIMS: usize = 4;

#[derive(Clone, Copy)]
pub union ShapeData {
	inline: [usize; INLINE_DIMS],
	heap: NonNull<[usize]>,
}

//--------------------------------------------------------------------------------------------------

pub struct Tensor {
	shape_len: ShapeLen,
	shape_data: ShapeData,
	dtype: DType,
	buf: IntrusiveRc<DeviceBuffer>,
}

impl Tensor {
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn ndim(&self) -> usize {
		self.shape_len.get()
	}

	pub fn shape(&self) -> &[usize] {
		let len = self.shape_len.get();
		if self.shape_len.is_inline() {
			unsafe { self.shape_data.inline.get_unchecked(..len) }
		} else {
			unsafe { self.shape_data.heap.as_ref().get_unchecked(..len) }
		}
	}
}

//--------------------------------------------------------------------------------------------------
