// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

pub mod batch;
pub mod buffer;
pub mod device;
pub mod dim_merger;
pub mod dim_vec;
pub mod dtype;
pub mod io;
pub mod math;

#[cfg(test)]
mod tests;

use std::intrinsics::{cold_path, likely};
use std::rc::Rc;
use std::slice::SliceIndex;

use buffer::Buffer;
use dim_vec::DimVec;

pub use device::Device;
pub use dtype::{DType, HasDType};

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Default)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

impl SizeAndStride {
	pub fn is_contiguous(&self) -> bool {
		self.stride == 1 || self.size <= 1
	}

	pub fn is_broadcasted(&self) -> bool {
		self.stride < 1 && self.size > 1
	}
}

//--------------------------------------------------------------------------------------------------

pub trait DimIndex {
	fn resolve(self, ndim: usize) -> usize;
}

impl DimIndex for usize {
	fn resolve(self, ndim: usize) -> usize {
		if self < ndim {
			self
		} else {
			cold_path();
			panic!("dimension index out of bounds: index = {}, ndim = {}", self, ndim);
		}
	}
}

impl DimIndex for isize {
	fn resolve(self, ndim: usize) -> usize {
		let dim = if self >= 0 { self as usize } else { ndim.wrapping_add(self as usize) };
		if dim < ndim {
			dim
		} else {
			cold_path();
			panic!("dimension index out of bounds: index = {}, ndim = {}", self, ndim);
		}
	}
}

impl DimIndex for u32 {
	fn resolve(self, ndim: usize) -> usize {
		(self as usize).resolve(ndim)
	}
}

impl DimIndex for i32 {
	fn resolve(self, ndim: usize) -> usize {
		(self as isize).resolve(ndim)
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tensor {
	// TODO - I could save 8 bytes per Tensor if I stored ndim as `u32` instead of `usize`.
	// However, in order to avoid padding, I'd need to stop using `DimVec` and implement all the
	// logic directly in `Tensor`.
	dims: DimVec,
	offset: usize,
	dtype: DType,
	buffer: Rc<Buffer>,
}

impl Tensor {
	/// This function will initialize strides in a DimVec so a new tensor is contiguous in memory.
	///
	/// It returns the total number of elements in the tensor.
	fn __init_strides(dims: &mut DimVec) -> usize {
		let mut elems = 1;
		let mut nonzero_elems: usize = 1;
		for dim in dims.iter_mut().rev() {
			// Check that if we ignore zero length dimensions, the number of elements does not
			// overflow. This is done to make sure our calculations would not overflow even if we
			// had the same dimensions but in different order.
			if likely(dim.size != 0) {
				nonzero_elems = nonzero_elems.checked_mul(dim.size).expect("too many elements");
			}

			dim.stride = elems;
			elems *= dim.size;
		}
		elems
	}

	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on(shape: &[usize], dtype: DType, device: Rc<dyn Device>) -> Tensor {
		let mut dims =
			DimVec::new_from_iter(shape.iter().map(|&size| SizeAndStride { size, stride: 0 }));
		let elems = Self::__init_strides(&mut dims);
		Tensor {
			dims,
			offset: 0,
			dtype,
			buffer: device.new_buffer(dtype, elems),
		}
	}

	/// Allocate a new tensor on the same device as `self`.
	pub fn new_empty(&self, shape: &[usize], dtype: DType) -> Tensor {
		Self::new_empty_on(shape, dtype, self.device())
	}

	pub fn new_replace_tail(&self, tail_len: usize, replace_with: &[usize]) -> Tensor {
		let n_keep = self.dims.len().checked_sub(tail_len).expect("not enough dimensions");
		let ndim = n_keep + replace_with.len();
		let mut dims = DimVec::with_capacity(ndim);
		unsafe {
			dims.extend_unchecked(self.dims.get_unchecked(..n_keep).iter().copied());
			dims.extend_unchecked(
				replace_with.iter().map(|&size| SizeAndStride { size, stride: 0 }),
			);
		}
		let elems = Self::__init_strides(&mut dims);
		Tensor {
			dims,
			offset: 0,
			dtype: self.dtype,
			buffer: self.device().new_buffer(self.dtype, elems),
		}
	}

	/// Allocate a new tensor on the same device with the same shape and dtype as `self`.
	pub fn new_empty_like(&self) -> Tensor {
		let dims = self.dims.clone();
		let elems = Self::__init_strides(&mut dims.clone());
		Tensor {
			dims,
			offset: 0,
			dtype: self.dtype,
			buffer: self.device().new_buffer(self.dtype, elems),
		}
	}

	/// Typical user of this function would be `nn` layer that can either
	/// allocate a new tensor for storing the output, or overwrite the input tensor.
	///
	/// If no one else has reference to this tensor's buffer, i.e.,
	/// this tensor owns its buffer, we assume the buffer is safe to overwrite
	/// and simply return a clone of `self`.
	///
	/// If the tensor does not own its buffer, we allocate a new empty tensor
	/// with the same shape and dtype as `self`.
	pub fn reuse_or_new_like(&self) -> Tensor {
		if self.owns_buffer() { self.clone() } else { self.new_empty_like() }
	}

	#[inline]
	pub fn owns_buffer(&self) -> bool {
		let buffer = &self.buffer;
		let weak = Rc::weak_count(buffer) + 1;
		let strong = Rc::strong_count(buffer);
		(weak | strong) <= 1
	}

	/// Returns the device on which the tensor is allocated.
	pub fn device(&self) -> Rc<dyn Device> {
		let device = &*self.buffer.device;
		device.clone()
	}

	pub fn dim<D: DimIndex>(&self, dim: D) -> SizeAndStride {
		let ndim = self.ndim();
		let dim = dim.resolve(ndim);
		*unsafe { self.dims.get_unchecked(dim) }
	}

	pub fn dim_slice<I: SliceIndex<[SizeAndStride]>>(&self, index: I) -> &I::Output {
		self.dims.get(index).expect("dimension index out of range")
	}

	/// Returns the number of dimensions in the tensor.
	///
	/// This is also known as the rank of the tensor.
	pub fn ndim(&self) -> usize {
		self.dims.len()
	}

	/// Returns the total number of elements in the tensor.
	pub fn elems(&self) -> usize {
		self.dims.iter().map(|dim| dim.size).product()
	}

	/// Returns the data type of the tensor elements.
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn reshape_last_dim<const TO: usize>(mut self, to_shape: [usize; TO]) -> Tensor {
		let dims = &mut self.dims;

		let removed_dim = dims.pop().expect("not enough dimensions");

		let elems = to_shape.iter().copied().product::<usize>();
		assert!(elems == removed_dim.size, "incompatible reshape");

		let mut stride = removed_dim.stride;
		dims.extend_rev(to_shape.iter().rev().map(|&size| {
			let dim = SizeAndStride { size, stride };
			stride *= size;
			dim
		}));

		self
	}

	pub fn merge_dims<const N: usize>(mut self) -> Tensor {
		let dims = &mut self.dims;
		let ndim = dims.len();
		assert!(N <= ndim, "not enough dimensions");

		if N == 0 {
			dims.push(SizeAndStride { size: 1, stride: 0 });
		} else {
			let mut merged = *unsafe { dims.get_unchecked(ndim - 1) };
			for i in (ndim - N..ndim - 1).rev() {
				let dim = *unsafe { dims.get_unchecked(i) };
				assert!(
					dim.size <= 1 || dim.stride == merged.size * merged.stride,
					"cannot merge because of discontinuity"
				);
				merged.size *= dim.size;
			}
			dims.pop_n(N - 1);
			*unsafe { dims.get_unchecked_mut(ndim - N) } = merged;
		}

		self
	}

	pub fn merge_all_dims(mut self) -> Tensor {
		let dims = &mut self.dims;
		let ndim = dims.len();

		match ndim {
			0 => {
				dims.push(SizeAndStride { size: 1, stride: 0 });
			},
			1 => {},
			_ => {
				let mut merged = *unsafe { dims.get_unchecked(ndim - 1) };
				for i in (0..ndim - 1).rev() {
					let dim = *unsafe { dims.get_unchecked(i) };
					assert!(
						dim.size <= 1 || dim.stride == merged.size * merged.stride,
						"cannot merge because of discontinuity"
					);
					merged.size *= dim.size;
				}
				dims.pop_n(ndim - 1);
				*unsafe { dims.get_unchecked_mut(0) } = merged;
			},
		}

		self
	}

	pub fn reshape<const FROM: usize, const TO: usize>(self, to_shape: [usize; TO]) -> Tensor {
		self.merge_dims::<FROM>().reshape_last_dim(to_shape)
	}

	/*
	/// Reshapes the last `n_dims_to_reshape` dimensions of the tensor
	pub fn reshape_n(self, _n_dims_to_reshape: usize, _new_shape: &[usize]) -> Tensor {
		let dims = &mut self.dims;
		let ndim = dims.len();
		let n_dims_to_keep = ndim - n_dims_to_reshape.min(ndim);

		// Merge the dimensions that we are going to reshape
		let dims_to_reshape = &dims[n_dims_to_keep..];
		let merged = DimMerger::new([dims_to_reshape]);

		// Resize the dims array. The new dimensions will be initialized in the `for` loop below.
		unsafe { dims.set_len(n_dims_to_keep) };
		dims.reserve(new_shape.len());
		unsafe { dims.set_len(n_dims_to_keep + new_shape.len()) };
		let new_dims = &mut dims[n_dims_to_keep..];

		// Try to match the new shape with the merged dimensions
		let smallest_dim = merged.smallest_dim();
		let mut prev_stride = smallest_dim.strides[0];
		let mut target_stride = smallest_dim.size * smallest_dim.strides[0];

		let merged_dims = merged.dims_increasing_without_smallest();
		let mut merged_dims = merged_dims.iter();

		for (dims_slot, size) in new_dims.iter_mut().zip(new_shape.iter().copied()).rev() {
			// We are about to store `size` into a slot in `dims`,
			// but first we need to calculate the stride

			let mut new_stride = prev_stride * size;

			if new_stride > target_stride {
				cold_path();

				if prev_stride == target_stride {
					let Some(merged_dim) = merged_dims.next() else {
						panic!("incompatible reshape");
					};
					prev_stride = merged_dim.strides[0];
					target_stride = merged_dim.size * merged_dim.strides[0];

					new_stride = prev_stride * size;
					if new_stride > target_stride {
						panic!("incompatible reshape");
					}
				} else {
					panic!("incompatible reshape");
				}
			}

			*dims_slot = SizeAndStride { size, stride: prev_stride };
			prev_stride = new_stride;
		}

		assert!(merged_dims.is_empty());
		assert!(prev_stride == target_stride);

		self
	}

	pub fn reshape_all(self, new_shape: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		self.reshape_n(ndim, new_shape)
	}
	*/

	pub fn batch_size(&self, non_batch_dims: usize) -> usize {
		let batch_dims =
			self.dims.len().checked_sub(non_batch_dims).expect("not enough dimensions");
		self.dims[..batch_dims].iter().map(|dim| dim.size).product()
	}

	pub fn transposed<D1: DimIndex, D2: DimIndex>(mut self, dim1: D1, dim2: D2) -> Tensor {
		let ndim = self.ndim();
		let dim1 = dim1.resolve(ndim);
		let dim2 = dim2.resolve(ndim);
		self.dims.swap(dim1, dim2);
		self
	}

	pub fn slice<D: DimIndex>(mut self, dim: D, range: std::ops::Range<usize>) -> Tensor {
		let ndim = self.ndim();
		let dim = dim.resolve(ndim);
		let dim = &mut self.dims[dim];
		assert!(range.start <= range.end, "invalid range");
		assert!(range.end <= dim.size, "invalid range");
		dim.size = range.end - range.start;
		self.offset += range.start * dim.stride;
		self
	}

	pub fn read_bin(&self, reader: &mut dyn std::io::Read) -> std::io::Result<()> {
		io::read_bin(self, reader)
	}

	pub fn read_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
		io::read_file(self, path)
	}

	pub fn write_bin(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
		io::write_bin(self, writer)
	}

	pub fn write_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
		io::write_file(self, path)
	}

	#[inline(never)]
	pub fn new_debug_1d<T: HasDType>(device: Rc<dyn Device>, value: io::DebugData1D<T>) -> Tensor {
		let (x,) = value.shape();
		let tensor = Self::new_empty_on(&[x], T::dtype, device);
		tensor.fill_debug_1d(value);
		tensor
	}

	#[inline(never)]
	pub fn new_debug_2d<T: HasDType>(device: Rc<dyn Device>, value: io::DebugData2D<T>) -> Tensor {
		let (y, x) = value.shape();
		let tensor = Self::new_empty_on(&[y, x], T::dtype, device);
		tensor.fill_debug_2d(value);
		tensor
	}

	#[inline(never)]
	pub fn new_debug_3d<T: HasDType>(device: Rc<dyn Device>, value: io::DebugData3D<T>) -> Tensor {
		let (z, y, x) = value.shape();
		let tensor = Self::new_empty_on(&[z, y, x], T::dtype, device);
		tensor.fill_debug_3d(value);
		tensor
	}

	#[inline(never)]
	pub fn fill_debug_1d<T: HasDType>(&self, value: io::DebugData1D<T>) {
		let (x,) = value.shape();
		assert!(self.ndim() == 1, "fill_1d() can only be used on 1D tensors");
		assert!(self.dims[0].size == x, "invalid size");
		assert!(self.dtype == T::dtype, "invalid dtype");

		let mut reader = value.into_read();
		io::read_bin(self, &mut reader).expect("failed to fill 1D tensor");
	}

	#[inline(never)]
	pub fn fill_debug_2d<T: HasDType>(&self, value: io::DebugData2D<T>) {
		let (y, x) = value.shape();
		assert!(self.ndim() == 2, "fill_2d() can only be used on 2D tensors");
		assert!(self.dims[0].size == y, "invalid size");
		assert!(self.dims[1].size == x, "invalid size");
		assert!(self.dtype == T::dtype, "invalid dtype");

		let mut reader = value.into_read();
		io::read_bin(self, &mut reader).expect("failed to fill 2D tensor");
	}

	#[inline(never)]
	pub fn fill_debug_3d<T: HasDType>(&self, value: io::DebugData3D<T>) {
		let (z, y, x) = value.shape();
		assert!(self.ndim() == 3, "fill_3d() can only be used on 3D tensors");
		assert!(self.dims[2].size == x, "invalid size");
		assert!(self.dims[1].size == y, "invalid size");
		assert!(self.dims[0].size == z, "invalid size");

		let mut reader = value.into_read();
		io::read_bin(self, &mut reader).expect("failed to fill 3D tensor");
	}
}

//--------------------------------------------------------------------------------------------------
