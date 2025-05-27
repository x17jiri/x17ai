// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ptr::NonNull;
use std::rc::Rc;

use super::device::Device;
use super::dtype::DType;

//--------------------------------------------------------------------------------------------------

pub struct Buffer {
	pub device: ManuallyDrop<Rc<dyn Device>>,
	pub device_buffer: NonNull<u8>,
	pub size_bytes: usize,
}

impl Buffer {
	#[inline]
	pub fn is_on_device(&self, device: &dyn Device) -> bool {
		let my_dev = self.device.as_ref();
		let my_dev = my_dev as *const dyn Device;
		let my_dev = my_dev as *const u8;

		let dev = device as *const dyn Device;
		let dev = dev as *const u8;

		my_dev == dev
	}

	#[inline(never)]
	pub fn executor(&self) -> &dyn Device {
		self.device.as_ref()
	}
}

impl Drop for Buffer {
	fn drop(&mut self) {
		unsafe { ManuallyDrop::take(&mut self.device) }
			.drop_buffer(self.device_buffer, self.size_bytes);
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SliceSet<'a> {
	pub buffer: &'a Buffer,
	pub dtype: DType,
	pub offset: usize,

	pub len: usize,
	pub count: usize,
	pub stride: usize,
}

impl<'a> SliceSet<'a> {
	pub fn span(&self) -> std::ops::Range<usize> {
		let begin = self.offset;
		let len = if self.count > 0 { (self.count - 1) * self.stride + self.len } else { 0 };
		let end = begin + len;
		begin..end
	}
}

//--------------------------------------------------------------------------------------------------

pub struct MatrixSet<'a> {
	pub slice_set: SliceSet<'a>,

	pub rows: NonZeroUsize,
	pub cols: NonZeroUsize,
	pub row_stride: usize,
	pub col_stride: usize,
}

impl<'a> MatrixSet<'a> {
	pub fn slice_len(
		rows: NonZeroUsize, cols: NonZeroUsize, row_stride: usize, col_stride: usize,
	) -> usize {
		(rows.get() - 1) * row_stride + (cols.get() - 1) * col_stride + 1
	}
}

//--------------------------------------------------------------------------------------------------
