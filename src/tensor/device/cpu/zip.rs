//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::tensor::HasDType;
use crate::tensor::device::executor::SliceBatchRef;
use crate::tensor::generic::map::{Map, ND, NDShape};
use crate::tensor::generic::{self};
use crate::util::array;
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

	fn size(&self, dim: usize) -> usize {
		match dim {
			0 => self.batch_size,
			1 => self.slice_len,
			_ => {
				cold_path();
				1
			},
		}
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

	fn size(&self, dim: usize) -> usize {
		match dim {
			0 => self.batch_size,
			1 => self.broadcast_len,
			_ => {
				cold_path();
				1
			},
		}
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

pub type CPUSliceTensor<'a, T> = generic::Tensor<SliceMap, &'a [T]>;
pub type CPUBroadcastTensor<'a, T> = generic::Tensor<BroadcastMap, &'a [T]>;

pub enum CPUInput<'a, T> {
	Slice(CPUSliceTensor<'a, T>),
	Broadcast(CPUBroadcastTensor<'a, T>),
}

impl<'a, T: HasDType> CPUInput<'a, T> {
	pub fn new_safe(input: &'a SliceBatchRef) -> Result<Self> {
		let input = input.view()?;
		input.ensure_safe()?;
		if input.map.dims[1].stride == 0 || input.map.dims[1].size <= 1 {
			let tensor = CPUBroadcastTensor {
				map: BroadcastMap {
					broadcast_len: input.map.dims[1].size,
					batch_size: input.map.dims[0].size,
					batch_stride: input.map.dims[0].stride,
				},
				buf: unsafe { input.buf.get_unchecked(input.map.offset..) },
			};
			Ok(CPUInput::Broadcast(tensor))
		} else if input.map.dims[1].stride == 1 {
			let tensor = CPUSliceTensor {
				map: SliceMap {
					slice_len: input.map.dims[1].size,
					batch_size: input.map.dims[0].size,
					batch_stride: input.map.dims[0].stride,
				},
				buf: unsafe { input.buf.get_unchecked(input.map.offset..) },
			};
			Ok(CPUInput::Slice(tensor))
		} else {
			Err(err_tensor_has_stride())
		}
	}

	pub fn new_safe_contiguous(input: &'a SliceBatchRef<'a>) -> Result<CPUSliceTensor<'a, T>> {
		let input = input.view()?;
		input.ensure_safe()?;
		if input.map.dims[1].stride != 0 || input.map.dims[1].size <= 1 {
			let tensor = CPUSliceTensor {
				map: SliceMap {
					slice_len: input.map.dims[1].size,
					batch_size: input.map.dims[0].size,
					batch_stride: input.map.dims[0].stride,
				},
				buf: unsafe { input.buf.get_unchecked(input.map.offset..) },
			};
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

pub unsafe fn zip<
	T: Copy,
	const O: usize, // number of outputs
	const C: usize, // number of contiguous inputs
	const B: usize, // number of broadcasted inputs
>(
	o: [generic::Tensor<ND<2>, &mut [T]>; O], //
	c: [generic::Tensor<ND<2>, &[T]>; C],
	b: [generic::Tensor<ND<2>, &[T]>; B],
	mut f: impl FnMut([&mut T; O], [T; C], [T; B]),
) {
	let shape = if let Some(t) = o.first() {
		t.map.nd_shape().unwrap()
	} else if let Some(t) = c.first() {
		t.map.nd_shape().unwrap()
	} else if let Some(t) = b.first() {
		t.map.nd_shape().unwrap()
	} else {
		return;
	};

	debug_assert!(o.iter().all(|t| t.ensure_safe().is_ok()));
	debug_assert!(c.iter().all(|t| t.ensure_safe().is_ok()));
	debug_assert!(b.iter().all(|t| t.ensure_safe().is_ok()));

	debug_assert!(o.iter().all(|t| t.nd_shape().unwrap() == shape));
	debug_assert!(c.iter().all(|t| t.nd_shape().unwrap() == shape));
	debug_assert!(b.iter().all(|t| t.nd_shape().unwrap() == shape));

	debug_assert!(o.iter().all(|t| t.map.dims[1].is_contiguous()));
	debug_assert!(c.iter().all(|t| t.map.dims[1].is_contiguous()));
	debug_assert!(b.iter().all(|t| t.map.dims[1].is_broadcasted()));

	let o = o.map(|t| (t.buf.as_mut_ptr().add(t.map.offset), t.map.dims[0].stride));
	let c = c.map(|t| (t.buf.as_ptr().add(t.map.offset), t.map.dims[0].stride));
	let b = b.map(|t| (t.buf.as_ptr().add(t.map.offset), t.map.dims[0].stride));

	for j in 0..shape[0] {
		let o = o.map(|(ptr, stride)| (ptr.add(j * stride)));
		let c = c.map(|(ptr, stride)| (ptr.add(j * stride)));
		let b = b.map(|(ptr, stride)| (ptr.add(j * stride)));
		for i in 0..shape[1] {
			let o = o.map(|ptr| ptr.add(i).as_mut().unwrap_unchecked());
			let c = c.map(|ptr| ptr.add(i).read());
			let b = b.map(|ptr| ptr.add(i).read());
			f(o, c, b);
		}
	}
}

/// # Safety
/// - safe-map - `t1` must have a safe map, i.e., every index must be mapped to an offset that is
///   within the bounds of the buffer.
pub unsafe fn zip1<T: Copy, M1: Map + Zippable>(
	t1: generic::Tensor<M1, &[T]>, mut f: impl FnMut(&T),
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
	t1: generic::Tensor<M1, &[T]>, t2: generic::Tensor<M2, &[T]>, mut f: impl FnMut(&T, &T),
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
	t1: generic::Tensor<M1, &[T]>, t2: generic::Tensor<M2, &[T]>, t3: generic::Tensor<M3, &[T]>,
	mut f: impl FnMut(&T, &T, &T),
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
	t: [generic::Tensor<M, &[T]>; N], mut f: impl FnMut([&T; N]),
) {
	debug_assert!(t.iter().all(|t| t.ensure_safe().is_ok()));
	let batch_size = t.first().map_or(0, |t| t.map.batch_size());
	debug_assert!(t.iter().map(|t| t.map.batch_size()).all(|b| b == batch_size));
	let item_len = t.first().map_or(0, |t| t.map.item_len());
	debug_assert!(t.iter().map(|t| t.map.item_len()).all(|i| i == item_len));
	for b in 0..batch_size {
		for i in 0..item_len {
			f(array::map(&t, |_, t| {
				let o = t.map.offset(b, i);
				debug_assert!(o < t.buf.len());
				unsafe { t.buf.get_unchecked(o) }
			}));
		}
	}
}

pub unsafe fn vec_zip_n<T: Copy, const N: usize>(
	t: [generic::Tensor<SliceMap, &[T]>; N], mut f: impl FnMut([&[T]; N]),
) {
	debug_assert!(t.iter().all(|t| t.ensure_safe().is_ok()));
	let batch_size = t.first().map_or(0, |t| t.map.batch_size());
	debug_assert!(t.iter().map(|t| t.map.batch_size()).all(|b| b == batch_size));
	let item_len = t.first().map_or(0, |t| t.map.item_len());
	debug_assert!(t.iter().map(|t| t.map.item_len()).all(|i| i == item_len));
	for b in 0..batch_size {
		f(array::map(&t, |_, t| {
			let b = t.map.offset(b, 0);
			let e = b + item_len;
			debug_assert!(e <= t.buf.len());
			unsafe { t.buf.get_unchecked(b..e) }
		}));
	}
}

#[allow(clippy::many_single_char_names)]
pub unsafe fn reduce_zip_n<T: Copy, const M: usize, const N: usize>(
	r: [generic::Tensor<SliceMap, &[T]>; M], t: [generic::Tensor<SliceMap, &[T]>; N],
	mut f: impl FnMut([&T; M], [&[T]; N]),
) {
	debug_assert!(t.iter().all(|t| t.ensure_safe().is_ok()));
	debug_assert!(r.iter().all(|r| r.ensure_safe().is_ok()));
	let batch_size = t.first().map_or(0, |t| t.map.batch_size());
	debug_assert!(t.iter().all(|t| t.map.batch_size() == batch_size));
	debug_assert!(r.iter().all(|r| r.map.batch_size() == batch_size));
	let item_len = t.first().map_or(0, |t| t.map.item_len());
	debug_assert!(t.iter().all(|t| t.map.item_len() == item_len));
	debug_assert!(r.iter().all(|r| r.map.item_len() == 1));
	for b in 0..batch_size {
		f(
			array::map(&r, |_, r| {
				let o = r.map.offset(b, 0);
				debug_assert!(o < r.buf.len());
				unsafe { r.buf.get_unchecked(o) }
			}),
			array::map(&t, |_, t| {
				let b = t.map.offset(b, 0);
				let e = b + item_len;
				debug_assert!(e <= t.buf.len());
				unsafe { t.buf.get_unchecked(b..e) }
			}),
		);
	}
}
