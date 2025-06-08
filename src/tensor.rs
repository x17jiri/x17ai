//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;
use std::rc::Rc;

pub use device::{DType, Device, HasDType};

use crate::Result;
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::device::executor::Executor;

pub mod batch;
pub mod device;
pub mod dim_merger;
pub mod generic;
// pub mod io; TODO
pub mod math;

#[cfg(false)] // TODO: #[cfg(test)]
mod tests;

//--------------------------------------------------------------------------------------------------

impl<M: generic::map::Map> generic::Tensor<M, Rc<device::DeviceBuffer>> {
	/// Returns a "view" tensor which has a slice `&[Cell<T>]` as its buffer.
	///
	/// # Errors
	/// If the buffer's dtype does not match `T` or if the buffer is not on CPU device.
	pub fn view<T: HasDType>(&self) -> Result<generic::Tensor<M, &[Cell<T>]>> {
		let buf = CPUDevice::view(self.buf.as_ref())?;
		let map = self.map.clone();
		Ok(generic::Tensor { map, buf })
	}
}

impl<M: generic::map::Map> generic::Tensor<M, &device::DeviceBuffer> {
	/// Returns a "view" tensor which has a slice `&[Cell<T>]` as its buffer.
	///
	/// # Errors
	/// If the buffer's dtype does not match `T` or if the buffer is not on CPU device.
	pub fn view<T: HasDType>(&self) -> Result<generic::Tensor<M, &[Cell<T>]>> {
		let buf = CPUDevice::view(self.buf)?;
		let map = self.map.clone();
		Ok(generic::Tensor { map, buf })
	}
}

//--------------------------------------------------------------------------------------------------

pub type Tensor = generic::Tensor<generic::map::DynD, Rc<device::DeviceBuffer>>;

impl Tensor {
	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on(shape: &[usize], dtype: DType, device: Rc<dyn Device>) -> Tensor {
		let (map, elems) = generic::map::DynD::new(shape);
		let buf = device.new_buffer(dtype, elems);
		Tensor { map, buf }
	}

	/// Allocate a new tensor on the same device as `self`.
	pub fn new_empty(&self, shape: &[usize], dtype: DType) -> Tensor {
		Self::new_empty_on(shape, dtype, self.device())
	}

	/// Allocate a new tensor on the same device with the same shape and dtype as `self`.
	pub fn new_empty_like(&self) -> Tensor {
		let (map, elems) = self.map.new_like();
		let buf = self.device().new_buffer(self.buf.dtype, elems);
		Tensor { map, buf }
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
		let buf = &self.buf;
		let weak = Rc::weak_count(buf) + 1;
		let strong = Rc::strong_count(buf);
		(weak | strong) <= 1
	}

	pub fn new_replace_tail(&self, tail_len: usize, replace_with: &[usize]) -> Tensor {
		let (map, elems) = self.map.new_replace_tail(tail_len, replace_with);
		let buf = self.device().new_buffer(self.buf.dtype, elems);
		Tensor { map, buf }
	}

	/// Returns the device on which the tensor is allocated.
	pub fn device(&self) -> Rc<dyn Device> {
		let device = &*self.buf.device;
		device.clone()
	}

	/// Returns the data type of the tensor elements.
	pub fn dtype(&self) -> DType {
		self.buf.dtype
	}

	pub fn executor(&self) -> &dyn Executor {
		self.buf.executor()
	}
}

#[cfg(false)]
impl Tensor {
	pub fn dim<D: DimIndex>(&self, dim: D) -> SizeAndStride {
		let ndim = self.ndim();
		let dim = dim.resolve(ndim);
		*unsafe { self.dims.get_unchecked(dim) }
	}

	pub fn dim_slice<I: SliceIndex<[SizeAndStride]>>(&self, index: I) -> &I::Output {
		self.dims.get(index).expect("dimension index out of range")
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
