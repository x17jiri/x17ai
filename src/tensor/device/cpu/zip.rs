//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;
use std::hint::cold_path;

use crate::tensor::HasDType;
use crate::tensor::device::executor::SliceBatch;
use crate::tensor::generic::map::Map;
use crate::tensor::generic::{self};
use crate::util::array::map_borrowed;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct SliceMap {
	pub slice_len: usize,
	pub batch_size: usize,
	pub batch_stride: usize,
}

impl Map for SliceMap {
	fn ndim(&self) -> usize {
		2
	}

	fn elems(&self) -> usize {
		self.slice_len * self.batch_size
	}

	fn span(&self) -> std::ops::Range<usize> {
		let elems = self.elems();
		if elems == 0 {
			cold_path();
			return 0..0;
		}
		let len = (self.batch_size - 1) * self.batch_stride + self.slice_len;
		0..len
	}

	fn is_contiguous(&self) -> bool {
		let elems = self.elems();
		if elems == 0 {
			cold_path();
			return true;
		}
		let len = (self.batch_size - 1) * self.batch_stride + self.slice_len;
		len == elems
	}
}

#[derive(Clone, Copy, Debug)]
pub struct BroadcastMap {
	pub broadcast_len: usize,
	pub batch_size: usize,
	pub batch_stride: usize,
}

impl Map for BroadcastMap {
	fn ndim(&self) -> usize {
		2
	}

	fn elems(&self) -> usize {
		self.broadcast_len * self.batch_size
	}

	fn span(&self) -> std::ops::Range<usize> {
		let elems = self.elems();
		if elems == 0 {
			cold_path();
			return 0..0;
		}
		let len = (self.batch_size - 1) * self.batch_stride + 1;
		0..len
	}

	fn is_contiguous(&self) -> bool {
		false
	}
}

pub type CPUSliceTensor<'a, T> = generic::Tensor<SliceMap, &'a [Cell<T>]>;
pub type CPUBroadcastTensor<'a, T> = generic::Tensor<BroadcastMap, &'a [Cell<T>]>;

pub enum CPUInput<'a, T> {
	Slice(CPUSliceTensor<'a, T>),
	Broadcast(CPUBroadcastTensor<'a, T>),
}

impl<'a, T: HasDType> CPUInput<'a, T> {
	pub fn new_safe(input: &'a SliceBatch) -> Result<Self> {
		let input = input.view()?;
		if input.map.dims[1].stride == 0 || input.map.dims[1].size <= 1 {
			let tensor = CPUBroadcastTensor {
				map: BroadcastMap {
					broadcast_len: input.map.dims[1].size,
					batch_size: input.map.dims[0].size,
					batch_stride: input.map.dims[0].stride,
				},
				buf: input.buf,
			};
			tensor.ensure_safe()?;
			Ok(CPUInput::Broadcast(tensor))
		} else if input.map.dims[1].stride == 1 {
			let tensor = CPUSliceTensor {
				map: SliceMap {
					slice_len: input.map.dims[1].size,
					batch_size: input.map.dims[0].size,
					batch_stride: input.map.dims[0].stride,
				},
				buf: input.buf,
			};
			tensor.ensure_safe()?;
			Ok(CPUInput::Slice(tensor))
		} else {
			#[cold]
			fn err_tensor_has_stride() -> Error {
				"Tensor data is neither contiguous nor broadcasted.".into()
			}
			Err(err_tensor_has_stride())
		}
	}

	pub fn new_safe_contiguous(input: &'a SliceBatch) -> Result<CPUSliceTensor<'a, T>> {
		let input = input.view()?;
		if input.map.dims[1].stride != 0 || input.map.dims[1].size <= 1 {
			let tensor = CPUSliceTensor {
				map: SliceMap {
					slice_len: input.map.dims[1].size,
					batch_size: input.map.dims[0].size,
					batch_stride: input.map.dims[0].stride,
				},
				buf: input.buf,
			};
			tensor.ensure_safe()?;
			Ok(tensor)
		} else {
			#[cold]
			fn err_tensor_not_contiguous() -> Error {
				"Tensor has to be contiguous".into()
			}
			Err(err_tensor_not_contiguous())
		}
	}
}

pub trait Zippable {
	fn item_len(&self) -> usize;
	fn batch_size(&self) -> usize;
	fn offset(&self, b: usize, i: usize) -> usize;
}

impl Zippable for SliceMap {
	fn item_len(&self) -> usize {
		self.slice_len
	}

	fn batch_size(&self) -> usize {
		self.batch_size
	}

	fn offset(&self, b: usize, i: usize) -> usize {
		debug_assert!(b < self.batch_size);
		debug_assert!(i < self.slice_len);
		b * self.batch_stride + i
	}
}

impl Zippable for BroadcastMap {
	fn item_len(&self) -> usize {
		self.broadcast_len
	}

	fn batch_size(&self) -> usize {
		self.batch_size
	}

	fn offset(&self, b: usize, i: usize) -> usize {
		debug_assert!(b < self.batch_size);
		debug_assert!(i < self.broadcast_len);
		b * self.batch_stride
	}
}

/// # Safety
/// - safe-map - `t1` must have a safe map, i.e., every index must be mapped to an offset that is
///   within the bounds of the buffer.
pub unsafe fn zip1<T: Copy, M1: Map + Zippable>(
	t1: generic::Tensor<M1, &[Cell<T>]>, mut f: impl FnMut(&Cell<T>),
) {
	debug_assert!(t1.ensure_safe().is_ok());
	let batch_size = t1.map.batch_size();
	let item_len = t1.map.item_len();
	for b in 0..batch_size {
		for i in 0..item_len {
			let o1 = t1.map.offset(b, i);
			debug_assert!(o1 < t1.buf.len());
			let v1 = unsafe { t1.buf.get_unchecked(o1) };

			f(v1);
		}
	}
}

/// # Safety
/// - safe-map - `t1`, `t2` must have a safe maps, i.e., every index must be mapped to an offset
///   that is within the bounds of the buffer.
/// - equal-shape - `t1`, `t2` must have equal `batch_size` and `item_len`.
pub unsafe fn zip2<T: Copy, M1: Map + Zippable, M2: Map + Zippable>(
	t1: generic::Tensor<M1, &[Cell<T>]>, t2: generic::Tensor<M2, &[Cell<T>]>,
	mut f: impl FnMut(&Cell<T>, &Cell<T>),
) {
	debug_assert!(t1.ensure_safe().is_ok());
	debug_assert!(t2.ensure_safe().is_ok());
	let batch_size = t1.map.batch_size();
	debug_assert_eq!(t2.map.batch_size(), batch_size);
	let item_len = t1.map.item_len();
	debug_assert_eq!(t2.map.item_len(), item_len);
	for b in 0..batch_size {
		for i in 0..item_len {
			let o1 = t1.map.offset(b, i);
			debug_assert!(o1 < t1.buf.len());
			let v1 = unsafe { t1.buf.get_unchecked(o1) };

			let o2 = t2.map.offset(b, i);
			debug_assert!(o2 < t2.buf.len());
			let v2 = unsafe { t2.buf.get_unchecked(o2) };

			f(v1, v2);
		}
	}
}

/// # Safety
/// - safe-map - `t1`, `t2`, `t3` must have safe maps, i.e., every index must be mapped to an offset
///   that is within the bounds of the buffer.
/// - equal-shape - `t1`, `t2`, `t3` must have equal `batch_size` and `item_len`.
pub unsafe fn zip3<T: Copy, M1: Map + Zippable, M2: Map + Zippable, M3: Map + Zippable>(
	t1: generic::Tensor<M1, &[Cell<T>]>, t2: generic::Tensor<M2, &[Cell<T>]>,
	t3: generic::Tensor<M3, &[Cell<T>]>, mut f: impl FnMut(&Cell<T>, &Cell<T>, &Cell<T>),
) {
	debug_assert!(t1.ensure_safe().is_ok());
	debug_assert!(t2.ensure_safe().is_ok());
	debug_assert!(t3.ensure_safe().is_ok());
	let batch_size = t1.map.batch_size();
	debug_assert_eq!(t2.map.batch_size(), batch_size);
	debug_assert_eq!(t3.map.batch_size(), batch_size);
	let item_len = t1.map.item_len();
	debug_assert_eq!(t2.map.item_len(), item_len);
	debug_assert_eq!(t3.map.item_len(), item_len);
	for b in 0..batch_size {
		for i in 0..item_len {
			let o1 = t1.map.offset(b, i);
			debug_assert!(o1 < t1.buf.len());
			let v1 = unsafe { t1.buf.get_unchecked(o1) };

			let o2 = t2.map.offset(b, i);
			debug_assert!(o2 < t2.buf.len());
			let v2 = unsafe { t2.buf.get_unchecked(o2) };

			let o3 = t3.map.offset(b, i);
			debug_assert!(o3 < t3.buf.len());
			let v3 = unsafe { t3.buf.get_unchecked(o3) };

			f(v1, v2, v3);
		}
	}
}

pub unsafe fn zip_n<T: Copy, M: Map + Zippable, const N: usize>(
	t: [generic::Tensor<M, &[Cell<T>]>; N], mut f: impl FnMut([&Cell<T>; N]),
) {
	debug_assert!(t.iter().all(|t| t.ensure_safe().is_ok()));
	let batch_size = t.first().map_or(0, |t| t.map.batch_size());
	debug_assert!(t.iter().map(|t| t.map.batch_size()).all(|b| b == batch_size));
	let item_len = t.first().map_or(0, |t| t.map.item_len());
	debug_assert!(t.iter().map(|t| t.map.item_len()).all(|i| i == item_len));
	for b in 0..batch_size {
		for i in 0..item_len {
			f(map_borrowed(&t, |_, t| {
				let o = t.map.offset(b, i);
				debug_assert!(o < t.buf.len());
				unsafe { t.buf.get_unchecked(o) }
			}));
		}
	}
}

pub unsafe fn vec_zip_n<T: Copy, const N: usize>(
	t: [generic::Tensor<SliceMap, &[Cell<T>]>; N], mut f: impl FnMut([&[Cell<T>]; N]),
) {
	debug_assert!(t.iter().all(|t| t.ensure_safe().is_ok()));
	let batch_size = t.first().map_or(0, |t| t.map.batch_size());
	debug_assert!(t.iter().map(|t| t.map.batch_size()).all(|b| b == batch_size));
	let item_len = t.first().map_or(0, |t| t.map.item_len());
	debug_assert!(t.iter().map(|t| t.map.item_len()).all(|i| i == item_len));
	for b in 0..batch_size {
		f(map_borrowed(&t, |_, t| {
			let b = t.map.offset(b, 0);
			let e = b + item_len;
			debug_assert!(e < t.buf.len());
			unsafe { t.buf.get_unchecked(b..e) }
		}));
	}
}
