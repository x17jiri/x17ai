// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

pub mod batch;
pub mod buffer;
pub mod device;
pub mod dim_merger;
pub mod dim_vec;
pub mod dtype;
pub mod math;

use core::panic;
use smallvec::{SmallVec, smallvec};
use std::alloc::Layout;
use std::intrinsics::{cold_path, likely, unlikely};
use std::iter::ExactSizeIterator;
use std::mem::MaybeUninit;
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::rc::{Rc, Weak};
use std::{array, fmt};

use buffer::{Buffer, BufferBase};
use dim_vec::{DimVec, SizeAndStride};

pub use device::Device;
pub use dtype::DType;

//--------------------------------------------------------------------------------------------------

pub type TensorSize = u32;
pub type NonZeroTensorSize = NonZeroU32;

pub fn tensor_size_to_usize(size: TensorSize) -> usize {
	usize::try_from(size).unwrap()
}

//--------------------------------------------------------------------------------------------------

pub struct ShapeView<'a> {
	dims: &'a [SizeAndStride],
}

impl<'a> ShapeView<'a> {
	pub fn len(&self) -> usize {
		self.dims.len()
	}
}

impl Index<isize> for ShapeView<'_> {
	type Output = TensorSize;

	fn index(&self, index: isize) -> &Self::Output {
		let i = if index < 0 { self.dims.len() as isize + index } else { index };
		&self.dims[i as usize].size
	}
}

impl Index<usize> for ShapeView<'_> {
	type Output = TensorSize;

	fn index(&self, index: usize) -> &Self::Output {
		&self.dims[index].size
	}
}

/*
impl std::cmp::PartialEq<ShapeView<'_>> for ShapeView<'_> {
	fn eq(&self, other: &ShapeView) -> bool {
		if self.tensor.dims.len() != other.tensor.dims.len() {
			return false;
		}
		for (a, b) in self.tensor.dims.iter().zip(other.tensor.dims.iter()) {
			if a.size != b.size {
				return false;
			}
		}
		true
	}
}
*/

impl<'a> IntoIterator for ShapeView<'a> {
	type Item = &'a TensorSize;
	type IntoIter =
		std::iter::Map<std::slice::Iter<'a, SizeAndStride>, fn(&SizeAndStride) -> &TensorSize>;

	fn into_iter(self) -> Self::IntoIter {
		self.dims.iter().map(|x| &x.size)
	}
}

//--------------------------------------------------------------------------------------------------

/// This iterator is used in the `new_replace_tail()` function.
/// I would use `std::iter::chain()` but it doesn't implement `ExactSizeIterator`.
struct ReplaceTailIter<'a> {
	a: &'a [SizeAndStride],
	b: &'a [TensorSize],
}

impl<'a> ReplaceTailIter<'a> {
	fn new(a: &'a [SizeAndStride], b: &'a [TensorSize]) -> ReplaceTailIter<'a> {
		ReplaceTailIter { a, b }
	}
}

impl<'a> ExactSizeIterator for ReplaceTailIter<'a> {
	fn len(&self) -> usize {
		self.a.len() + self.b.len()
	}
}

impl<'a> Iterator for ReplaceTailIter<'a> {
	type Item = &'a TensorSize;

	fn next(&mut self) -> Option<Self::Item> {
		if self.a.is_empty() {
			if self.b.is_empty() {
				None
			} else {
				let item = &self.b[0];
				self.b = &self.b[1..];
				Some(item)
			}
		} else {
			let item = &self.a[0].size;
			self.a = &self.a[1..];
			Some(item)
		}
	}
}

impl<'a> DoubleEndedIterator for ReplaceTailIter<'a> {
	fn next_back(&mut self) -> Option<Self::Item> {
		if self.b.is_empty() {
			if self.a.is_empty() {
				None
			} else {
				let item = &self.a[self.a.len() - 1].size;
				self.a = &self.a[..self.a.len() - 1];
				Some(item)
			}
		} else {
			let item = &self.b[self.b.len() - 1];
			self.b = &self.b[..self.b.len() - 1];
			Some(item)
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tensor {
	// dims in reverse order
	pub(crate) dims: DimVec,
	pub(crate) offset: TensorSize,
	pub(crate) dtype: DType,
	pub(crate) elems: TensorSize,
	pub(crate) buffer: Rc<dyn Buffer>,
}

impl Tensor {
	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on<'a, Shape>(shape: Shape, dtype: DType, device: Rc<dyn Device>) -> Tensor
	where
		Shape: IntoIterator<Item = &'a TensorSize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a TensorSize> + ExactSizeIterator,
	{
		let shape = shape.into_iter().copied();
		let ndim = shape.len();
		let mut dims = DimVec::with_capacity(ndim);
		let mut elems = 1;
		let mut nonzero_elems: TensorSize = 1;

		let mut ptr = unsafe { dims.vec.as_mut_ptr().add(ndim) };
		for size in shape.rev() {
			// Check that if we ignore zero length dimensions, the number of elements does not
			// overflow. This is done to make sure our calculations would not overflow even if we
			// had the same dimensions but in different order.
			if likely(size != 0) {
				if let Some(mul) = nonzero_elems.checked_mul(size) {
					nonzero_elems = mul;
				} else {
					panic!("too many elements");
				};
			}

			let stride = elems;
			elems *= size;

			unsafe {
				ptr = ptr.sub(1);
				ptr.write(SizeAndStride { size, stride })
			};
		}

		unsafe { dims.vec.set_len(ndim) };

		Tensor {
			dims,
			offset: 0,
			dtype,
			elems,
			buffer: device.new_buffer(dtype, elems),
		}
	}

	/// Allocate a new tensor on the same device as `self`.
	pub fn new_empty<'a, Shape>(&'a self, shape: Shape, dtype: DType) -> Tensor
	where
		Shape: IntoIterator<Item = &'a TensorSize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a TensorSize> + ExactSizeIterator,
	{
		Self::new_empty_on(shape, dtype, self.device())
	}

	pub fn new_replace_tail(&self, tail_len: usize, replace_with: &[TensorSize]) -> Tensor {
		let ndim = self.dims.len();
		assert!(tail_len <= ndim, "not enough dimensions");
		let keep_len = ndim - tail_len;
		let keep = &self.dims[..keep_len];

		let new_shape = ReplaceTailIter::new(keep, replace_with);

		Self::new_empty_on(new_shape, self.dtype, self.device())
	}

	/// Allocate a new tensor on the same device with the same shape and dtype
	/// as `self`.
	pub fn new_empty_like(&self) -> Tensor {
		Self::new_empty_on(self.shape(), self.dtype, self.device())
	}

	/// Returns the device on which the tensor is allocated.
	pub fn device(&self) -> Rc<dyn Device> {
		BufferBase::from_dyn_buf(self.buffer.as_ref()).device.clone()
	}

	#[inline]
	pub fn owns_buffer(&self) -> bool {
		let buffer = &self.buffer;
		let weak = Rc::weak_count(buffer) + 1;
		let strong = Rc::strong_count(buffer);
		(weak | strong) <= 1
	}

	pub fn shape(&self) -> ShapeView {
		ShapeView { dims: &self.dims }
	}

	/// `dim` should be in the range `0..<ndim`.
	pub fn dim_from_start(&self, dim: usize) -> SizeAndStride {
		let ndim = self.dims.len();
		if dim < ndim { self.dims[dim] } else { SizeAndStride { size: 1, stride: 1 } }
	}

	/// `dim` should be in the range `1..=ndim`.
	pub fn dim_from_end(&self, dim: usize) -> SizeAndStride {
		let ndim = self.dims.len();
		let dim = ndim.wrapping_sub(dim);
		if dim < ndim { self.dims[dim] } else { SizeAndStride { size: 1, stride: 0 } }
	}

	pub fn dim(&self, dim: isize) -> SizeAndStride {
		if dim >= 0 {
			self.dim_from_start(dim.unsigned_abs())
		} else {
			self.dim_from_end(dim.unsigned_abs())
		}
	}

	/// Returns the number of dimensions in the tensor.
	///
	/// This is also known as the rank of the tensor.
	pub fn ndim(&self) -> usize {
		self.dims.len()
	}

	/// Returns the total number of elements in the tensor.
	pub fn elems(&self) -> TensorSize {
		self.elems
	}

	/// Returns the data type of the tensor elements.
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn reshape_last_dim<const TO: usize>(mut self, to_shape: [TensorSize; TO]) -> Tensor {
		let dims = &mut self.dims;

		let removed_dim = dims.pop().expect("not enough dimensions");

		let elems = to_shape.iter().copied().product::<TensorSize>();
		assert!(elems == removed_dim.size, "incompatible reshape");

		let mut stride = removed_dim.stride;
		let mut new_dims: [MaybeUninit<SizeAndStride>; TO] = MaybeUninit::uninit_array();
		for i in (0..TO).rev() {
			let size = to_shape[i];
			new_dims[i].write(SizeAndStride { size, stride });
			stride *= size;
		}
		let new_dims = unsafe { MaybeUninit::array_assume_init(new_dims) };

		dims.extend_with_array(&new_dims);

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

	pub fn reshape<const FROM: usize, const TO: usize>(self, to_shape: [TensorSize; TO]) -> Tensor {
		self.merge_dims::<FROM>().reshape_last_dim(to_shape)
	}

	/// Reshapes the last `n_dims_to_reshape` dimensions of the tensor
	pub fn reshape_n(mut self, n_dims_to_reshape: usize, new_shape: &[TensorSize]) -> Tensor {
		todo!("reshape_n");
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

	pub fn reshape_all(self, new_shape: &[TensorSize]) -> Tensor {
		let ndim = self.dims.len();
		self.reshape_n(ndim, new_shape)
	}

	pub fn batch_size(&self, non_batch_dims: usize) -> TensorSize {
		assert!(non_batch_dims <= self.dims.len(), "not enough dimensions");
		let batch_dims = self.dims.len() - non_batch_dims;

		let mut batch_size = 1;
		for i in 0..batch_dims {
			batch_size *= self.dims[i].size;
		}
		batch_size
	}

	pub fn dim_to_positive(&self, dim: isize) -> usize {
		let ndim = self.dims.len();
		let dim = if dim >= 0 { dim as usize } else { ndim.wrapping_add(dim as usize) };
		if likely(dim < ndim) {
			dim
		} else {
			panic!("dimension out of range");
		}
	}

	pub fn transposed(mut self, dim1: isize, dim2: isize) -> Tensor {
		let dim1 = self.dim_to_positive(dim1);
		let dim2 = self.dim_to_positive(dim2);

		self.dims.swap(dim1, dim2);
		self
	}

	pub fn permuted(mut self, perm: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		assert!(perm.len() == ndim, "number of dimensions does not match");

		let mut sum = 0;
		let mut new_dims = DimVec::with_capacity(ndim);
		new_dims.extend(perm.iter().map(|p| {
			sum += *p;
			&self.dims[*p]
		}));

		let expected_sum = ndim * (ndim - 1) / 2;
		assert!(sum == expected_sum, "invalid permutation"); // TODO - does this work?

		self.dims = new_dims;
		self
	}

	#[inline(never)] // TODO - remove
	pub fn slice(mut self, dim: isize, range: std::ops::Range<TensorSize>) -> Tensor {
		let dim = self.dim_to_positive(dim as isize);
		let dim = &mut self.dims[dim];
		assert!(range.start <= range.end, "invalid range");
		assert!(range.end <= dim.size, "invalid range");
		dim.size = range.end - range.start;
		self.offset += range.start * dim.stride;
		self
	}
}

fn fmt_0d(tensor: &Tensor, f: &mut fmt::Formatter, offset: TensorSize) -> fmt::Result {
	let offset = tensor.offset + offset;
	tensor.buffer.format(f, tensor.dtype, offset, 1, 1)
}

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, offset: TensorSize) -> fmt::Result {
	let offset = tensor.offset + offset;
	let dim = tensor.dims[tensor.ndim() - 1];
	write!(f, "[")?;
	tensor.buffer.format(f, tensor.dtype, offset, dim.size, dim.stride)?;
	write!(f, "]")
}

fn fmt_2d(tensor: &Tensor, f: &mut fmt::Formatter, offset: TensorSize) -> fmt::Result {
	writeln!(f, "[")?;
	let dim = tensor.dims[tensor.ndim() - 2];
	for i in 0..dim.size {
		write!(f, "\t")?;
		fmt_1d(tensor, f, offset + i * dim.stride)?;
		writeln!(f, ",")?;
	}
	write!(f, "]")
}

impl fmt::Display for Tensor {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Tensor(")?;
		match self.ndim() {
			0 => fmt_0d(self, f, 0)?,
			1 => fmt_1d(self, f, 0)?,
			2 => fmt_2d(self, f, 0)?,
			_ => {
				todo!("Tensor with {} dimensions", self.ndim());
			},
		};
		write!(f, ")")
	}
}

//--------------------------------------------------------------------------------------------------
