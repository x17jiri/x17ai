// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

pub mod batch;
pub mod buffer;
pub mod device;
pub mod dim_merger;
pub mod dim_vec;
pub mod dtype;
pub mod math;

#[cfg(test)]
mod tests;

use std::intrinsics::likely;
use std::rc::Rc;
use std::slice::SliceIndex;

use buffer::Buffer;
use dim_vec::{DimIndex, DimVec};

pub use device::Device;
pub use dtype::{DType, HasDType};

//--------------------------------------------------------------------------------------------------

pub struct Data1D<T: HasDType> {
	pub data: Vec<T>,
}

pub struct Data2D<T: HasDType> {
	pub data: Vec<Vec<T>>,
}

pub struct Data3D<T: HasDType> {
	pub data: Vec<Vec<Vec<T>>>,
}

#[macro_export]
macro_rules! data1d {
    ( $dt:ty; $( $x:expr ),* $(,)? ) => {
        $crate::tensor::Data1D::<$dt> {
			data: vec![$($x),*]
		}
    };
}

#[macro_export]
macro_rules! data2d {
    ( $dt:ty; $( [ $( $x:expr ),* ] ),* $(,)? ) => {
		$crate::tensor::Data2D::<$dt> {
        	data: vec![
				$(vec![$($x),*]),*
			]
		}
    };
}

#[macro_export]
macro_rules! data3d {
	( $dt:ty; $( [ $( [ $( $x:expr ),* ] ),* $(,)? ] ),* $(,)? ) => {
		$crate::tensor::Data3D::<$dt> {
			data: vec![
				$(
					vec![
						$(vec![$($x),*]),*
					]
				),*
			]
		}
	};
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Default)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

pub enum DimType {
	Contiguous,
	Broadcasted,
	Strided,
}

impl SizeAndStride {
	pub fn is_contiguous(&self) -> bool {
		self.stride == 1 || self.size <= 1
	}

	pub fn is_broadcasted(&self) -> bool {
		self.stride < 1 && self.size > 1
	}

	pub fn is_strided(&self) -> bool {
		self.stride > 1 && self.size > 1
	}

	pub fn dim_type(&self) -> DimType {
		if self.stride == 1 || self.size <= 1 {
			DimType::Contiguous
		} else if self.stride < 1 {
			DimType::Broadcasted
		} else {
			DimType::Strided
		}
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
	fn __new_empty_on(mut dims: DimVec, dtype: DType, device: Rc<dyn Device>) -> Tensor {
		let mut stride = 1;
		let mut nonzero_elems: usize = 1;
		for dim in dims.iter_mut().rev() {
			// Check that if we ignore zero length dimensions, the number of elements does not
			// overflow. This is done to make sure our calculations would not overflow even if we
			// had the same dimensions but in different order.
			if likely(dim.size != 0) {
				if let Some(mul) = nonzero_elems.checked_mul(dim.size) {
					nonzero_elems = mul;
				} else {
					panic!("too many elements");
				};
			}

			dim.stride = stride;
			stride *= dim.size;
		}
		Tensor {
			dims,
			offset: 0,
			dtype,
			buffer: device.new_buffer(dtype, stride),
		}
	}

	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on(shape: &[usize], dtype: DType, device: Rc<dyn Device>) -> Tensor {
		let dims =
			DimVec::new_from_iter(shape.iter().map(|&size| SizeAndStride { size, stride: 0 }));
		Self::__new_empty_on(dims, dtype, device)
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

		Self::__new_empty_on(dims, self.dtype, self.device())
	}

	/// Allocate a new tensor on the same device with the same shape and dtype as `self`.
	pub fn new_empty_like(&self) -> Tensor {
		let dims = self.dims.clone();
		Self::__new_empty_on(dims, self.dtype, self.device())
	}

	/// Returns the device on which the tensor is allocated.
	pub fn device(&self) -> Rc<dyn Device> {
		let device = &*self.buffer.device;
		device.clone()
	}

	#[inline]
	pub fn owns_buffer(&self) -> bool {
		let buffer = &self.buffer;
		let weak = Rc::weak_count(buffer) + 1;
		let strong = Rc::strong_count(buffer);
		(weak | strong) <= 1
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

	/// Reshapes the last `n_dims_to_reshape` dimensions of the tensor
	pub fn reshape_n(self, _n_dims_to_reshape: usize, _new_shape: &[usize]) -> Tensor {
		todo!("reshape_n"); // TODO
		//
		/*
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
		*/
	}

	pub fn reshape_all(self, new_shape: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		self.reshape_n(ndim, new_shape)
	}

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

	#[inline(never)]
	pub fn new_1d<T: HasDType>(device: Rc<dyn Device>, value: Data1D<T>) -> Tensor {
		let x = value.data.len();

		let tensor = Self::new_empty_on(&[x], T::dtype, device);
		tensor.fill_1d(value);
		tensor
	}

	#[inline(never)]
	pub fn new_2d<T: HasDType>(device: Rc<dyn Device>, value: Data2D<T>) -> Tensor {
		let y = value.data.len();

		let x = value.data.get(0).map_or(0, |row| row.len());
		assert!(value.data.iter().all(|row| row.len() == x), "rows have different lengths");

		let tensor = Self::new_empty_on(&[y, x], T::dtype, device);
		tensor.fill_2d(value);
		tensor
	}

	#[inline(never)]
	pub fn new_3d<T: HasDType>(device: Rc<dyn Device>, value: Data3D<T>) -> Tensor {
		let z = value.data.len();

		let y = value.data.get(0).map_or(0, |mat| mat.len());
		assert!(
			value.data.iter().all(|mat| mat.len() == y),
			"matrices have different numbers of rows"
		);

		let x = value.data.get(0).and_then(|mat| mat.get(0)).map_or(0, |row| row.len());
		assert!(
			value.data.iter().all(|mat| mat.iter().all(|row| row.len() == x)),
			"rows have different lengths"
		);

		let tensor = Self::new_empty_on(&[z, y, x], T::dtype, device);
		tensor.fill_3d(value);
		tensor
	}

	#[inline(never)]
	pub fn fill_1d<T: HasDType>(&self, value: Data1D<T>) {
		let x = value.data.len();

		assert!(self.ndim() == 1, "fill_1d() can only be used on 1D tensors");

		assert!(self.dims[0].size == x, "invalid size");
		assert!(self.dims[0].size <= 1 || self.dims[0].stride == 1, "tensor is not contiguous");

		assert!(self.dtype == T::dtype, "invalid dtype");
		let dtype = self.dtype;
		let offset = self.offset;

		let buf = value.data.as_slice();
		let buf = unsafe {
			std::slice::from_raw_parts(
				buf.as_ptr() as *const u8,
				buf.len() * std::mem::size_of::<T>(),
			)
		};

		let executor = self.buffer.executor();
		executor.load_data(self.buffer.as_ref(), dtype, offset, x, buf);
	}

	#[inline(never)]
	pub fn fill_2d<T: HasDType>(&self, value: Data2D<T>) {
		let y = value.data.len();

		let x = value.data.get(0).map_or(0, |row| row.len());
		assert!(value.data.iter().all(|row| row.len() == x), "rows have different lengths");

		assert!(self.ndim() == 2, "fill_2d() can only be used on 2D tensors");

		assert!(self.dims[1].size == x, "invalid size");
		assert!(self.dims[1].size <= 1 || self.dims[1].stride == 1, "tensor is not contiguous");

		assert!(self.dims[0].size == y, "invalid size");
		assert!(self.dims[0].size <= 1 || self.dims[0].stride >= x, "output overlap");

		assert!(self.dtype == T::dtype, "invalid dtype");
		let dtype = self.dtype;

		for (i, row) in value.data.into_iter().enumerate() {
			let offset = self.offset + i * self.dims[0].stride;

			let buf = row.as_slice();
			let buf = unsafe {
				std::slice::from_raw_parts(
					buf.as_ptr() as *const u8,
					buf.len() * std::mem::size_of::<T>(),
				)
			};

			let executor = self.buffer.executor();
			executor.load_data(self.buffer.as_ref(), dtype, offset, x, buf);
		}
	}

	#[inline(never)]
	pub fn fill_3d<T: HasDType>(&self, value: Data3D<T>) {
		let z = value.data.len();

		let y = value.data.get(0).map_or(0, |mat| mat.len());
		assert!(
			value.data.iter().all(|mat| mat.len() == y),
			"matrices have different numbers of rows"
		);

		let x = value.data.get(0).and_then(|mat| mat.get(0)).map_or(0, |row| row.len());
		assert!(
			value.data.iter().all(|mat| mat.iter().all(|row| row.len() == x)),
			"rows have different lengths"
		);

		assert!(self.ndim() == 3, "fill_3d() can only be used on 3D tensors");

		assert!(self.dims[2].size == x, "invalid size");
		assert!(self.dims[2].size <= 1 || self.dims[2].stride == 1, "tensor is not contiguous");

		assert!(self.dims[1].size == y, "invalid size");
		assert!(self.dims[1].size <= 1 || self.dims[1].stride >= x, "output overlap");

		assert!(self.dims[0].size == z, "invalid size");
		assert!(
			self.dims[0].size <= 1 || self.dims[0].stride >= y * self.dims[1].stride,
			"output overlap"
		);

		assert!(self.dtype == T::dtype, "invalid dtype");
		let dtype = self.dtype;

		for (i, mat) in value.data.into_iter().enumerate() {
			let offset = self.offset + i * self.dims[0].stride;

			for (j, row) in mat.into_iter().enumerate() {
				let row_offset = offset + j * self.dims[1].stride;

				let buf = row.as_slice();
				let buf = unsafe {
					std::slice::from_raw_parts(
						buf.as_ptr() as *const u8,
						buf.len() * std::mem::size_of::<T>(),
					)
				};

				let executor = self.buffer.executor();
				executor.load_data(self.buffer.as_ref(), dtype, row_offset, x, buf);
			}
		}
	}
}

fn fmt_0d(tensor: &Tensor, f: &mut std::fmt::Formatter, offset: usize) -> std::fmt::Result {
	let executor = tensor.buffer.executor();
	let offset = tensor.offset + offset;
	let len = 1;
	let stride = 1;
	executor.format(f, tensor.buffer.as_ref(), tensor.dtype, offset, len, stride)
}

fn fmt_1d(tensor: &Tensor, f: &mut std::fmt::Formatter, offset: usize) -> std::fmt::Result {
	let executor = tensor.buffer.executor();
	let dim = tensor.dims[tensor.ndim() - 1];
	let offset = tensor.offset + offset;
	let len = dim.size;
	let stride = dim.stride;
	write!(f, "[")?;
	executor.format(f, tensor.buffer.as_ref(), tensor.dtype, offset, len, stride)?;
	write!(f, "]")
}

fn fmt_Nd(
	tensor: &Tensor, f: &mut std::fmt::Formatter, offset: usize, d: usize,
) -> std::fmt::Result {
	let indent = "\t".repeat(d);
	writeln!(f, "{indent}[")?;
	let dim = tensor.dims[d];
	for i in 0..dim.size {
		write!(f, "{indent}\t")?;
		let offset = offset + i * dim.stride;
		if d + 1 < tensor.ndim() - 1 {
			fmt_Nd(tensor, f, offset, d + 1)?;
		} else {
			fmt_1d(tensor, f, offset)?;
		}
		writeln!(f, ",")?;
	}
	write!(f, "{indent}]")
}

impl std::fmt::Display for Tensor {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Tensor(")?;
		match self.ndim() {
			0 => fmt_0d(self, f, 0)?,
			1 => fmt_1d(self, f, 0)?,
			_ => fmt_Nd(self, f, 0, 0)?,
		};
		write!(f, ")")
	}
}

//--------------------------------------------------------------------------------------------------
