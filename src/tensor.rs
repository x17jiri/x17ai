//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

pub use device::{DType, Device, HasDType};

use crate::tensor::device::NewDeviceBufferError;
use crate::tensor::device::buffer::{
	BorrowError, BorrowMutError, DeviceBufferRef, DeviceBufferRefMut,
};
use crate::tensor::device::cpu::{CPUDevice, ViewError};
use crate::tensor::device::executor::{Executor, ExecutorError};
use crate::tensor::dim_merger::DimMergerError;
use crate::tensor::generic::map::dd::ReplaceTailError;
use crate::tensor::generic::map::{
	DD, ElementsOverflowError, IncompatibleStridesError, MergeDimsError, NotEnoughDimensionsError,
	ReshapeLastDimError, SelectError,
};
use crate::tensor::math::EvaluatesToTensor;
use crate::{ErrExtra, ErrPack};

pub mod device;
pub mod dim_merger;
pub mod generic;
pub mod io;
pub mod math;

#[cfg(false)] // TODO: #[cfg(test)]
mod tests;

//--------------------------------------------------------------------------------------------------

impl<M: generic::map::Map> generic::Tensor<M, Rc<device::DeviceBuffer>> {
	/// # Errors
	/// `BorrowError` if there is a mutable borrow preventing a shared borrow.
	pub fn borrow(
		&self,
	) -> std::result::Result<generic::Tensor<&M::Deref, DeviceBufferRef<'_>>, BorrowError> {
		let map = self.map().as_ref();
		let buf = self.buf().try_borrow()?;
		// SAFETY: We only change the type of buffer reference.
		// So if the map was safe before, it is still safe.
		Ok(unsafe { generic::Tensor::new_unchecked(map, buf) })
	}

	/// # Errors
	/// `BorrowMutError` if there already is any other borrow of the buffer.
	pub fn borrow_mut(
		&self,
	) -> std::result::Result<generic::Tensor<&M::Deref, DeviceBufferRefMut<'_>>, BorrowMutError> {
		let map = self.map().as_ref();
		let buf = self.buf().try_borrow_mut()?;
		// SAFETY: We only change the type of buffer reference.
		// So if the map was safe before, it is still safe.
		Ok(unsafe { generic::Tensor::new_unchecked(map, buf) })
	}
}

impl<'buf, M: generic::map::Map> generic::Tensor<M, DeviceBufferRef<'buf>> {
	/// Returns a "view" tensor which has a slice `&[T]` as its buffer.
	pub fn view<T: HasDType>(&self) -> Result<generic::Tensor<&M::Deref, &[T]>, ViewError> {
		let map = self.map().as_ref();
		let buf = self.buf();
		CPUDevice::ensure_can_view::<T>(buf.device_buffer())?;
		let data = buf.device_data;
		let elems = buf.elems;
		let slice = unsafe { std::slice::from_raw_parts(data.cast(), elems) };

		// SAFETY: We only change the type of buffer reference.
		// So if the map was safe before, it is still safe.
		Ok(unsafe { generic::Tensor::new_unchecked(map, slice) })
	}
}

impl<'buf, M: generic::map::Map> generic::Tensor<M, DeviceBufferRefMut<'buf>> {
	/// Returns a "view" tensor which has a slice `&mut [T]` as its buffer.
	pub fn view_mut<T: HasDType>(
		&mut self,
	) -> Result<generic::Tensor<&M::Deref, &mut [T]>, ViewError> {
		let map = self.map().as_ref();
		let buf = self.buf();
		CPUDevice::ensure_can_view::<T>(buf.device_buffer())?;
		let data = buf.device_data;
		let elems = buf.elems;
		let slice = unsafe { std::slice::from_raw_parts_mut(data.cast(), elems) };

		// SAFETY: We only change the type of buffer reference.
		// So if the map was safe before, it is still safe.
		Ok(unsafe { generic::Tensor::new_unchecked(map, slice) })
	}
}

//--------------------------------------------------------------------------------------------------

pub type Tensor = generic::Tensor<DD, Rc<device::DeviceBuffer>>;

impl Tensor {
	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on(
		shape: &[usize],
		dtype: DType,
		device: Rc<dyn Device>,
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let (map, elems) = DD::new(shape)?;
		let buf = device.new_buffer(dtype, elems)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Tensor::new_unchecked(map, buf) })
	}

	/// Allocate a new tensor on the same device as `self`.
	pub fn new_empty(
		&self,
		shape: &[usize],
		dtype: DType,
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		Self::new_empty_on(shape, dtype, self.device())
	}

	/// Allocate a new tensor on the same device with the same shape and dtype as `self`.
	pub fn new_empty_like(&self) -> Result<Tensor, ErrPack<TensorOpError>> {
		let (map, elems) = self.map().new_like();
		let buf = self.device().new_buffer(self.buf().dtype, elems)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Tensor::new_unchecked(map, buf) })
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
	pub fn reuse_or_new_like(&self) -> Result<Tensor, ErrPack<TensorOpError>> {
		if self.owns_buffer() { Ok(self.clone()) } else { self.new_empty_like() }
	}

	#[inline]
	pub fn owns_buffer(&self) -> bool {
		let buf = self.buf();
		let weak = Rc::weak_count(buf) + 1;
		let strong = Rc::strong_count(buf);
		(weak | strong) <= 1
	}

	pub fn new_replace_tail(
		&self,
		tail_len: usize,
		replace_with: &[usize],
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let (map, elems) = self.map().new_replace_tail(tail_len, replace_with)?;
		let buf = self.device().new_buffer(self.buf().dtype, elems)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Tensor::new_unchecked(map, buf) })
	}

	/// Returns the device on which the tensor is allocated.
	pub fn device(&self) -> Rc<dyn Device> {
		let device = &*self.buf().device;
		device.clone()
	}

	/// Returns the data type of the tensor elements.
	pub fn dtype(&self) -> DType {
		self.buf().dtype
	}

	pub fn executor(&self) -> &dyn Executor {
		self.buf().executor()
	}

	pub fn assign<Expr: EvaluatesToTensor>(
		&self,
		expr: Expr,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.eval_to_tensor(self)
	}

	/// I use this function because Rust doesn't allow specifying only some generic parameters.
	///
	/// If I created `Tensor::new_2d<T, const Y: usize, const X: usize>(...)`,
	/// the caller would have to specify either all `T`, `Y`, `X`, or none of them.
	///
	/// With `literal_factory`, it is possible to only specify `T` and have `Y` and `X` inferred:
	///
	///     Tensor::literal_factory<f32>::new_2d(...)
	pub fn literal_factory<T: HasDType>(device: Rc<dyn Device>) -> TensorLiteralFactory<T> {
		TensorLiteralFactory {
			device,
			phantom: std::marker::PhantomData,
		}
	}
}

pub struct TensorLiteralFactory<T: HasDType> {
	pub device: Rc<dyn Device>,
	pub phantom: std::marker::PhantomData<T>,
}

impl<T: HasDType> TensorLiteralFactory<T> {
	#[inline(never)]
	pub fn new_1d<const X: usize>(&self, value: &[T; X]) -> Result<Tensor, ErrPack<TensorOpError>> {
		let tensor = Tensor::new_empty_on(&[X], T::dtype, self.device.clone())?;

		let val_ptr = value.as_ptr() as *const u8;
		let val_len = X * std::mem::size_of::<T>();
		let val = unsafe { std::slice::from_raw_parts(val_ptr, val_len) };
		let mut reader = std::io::Cursor::new(val);
		io::read_bin(&tensor, &mut reader)?;

		Ok(tensor)
	}

	#[inline(never)]
	pub fn new_2d<const Y: usize, const X: usize>(
		&self,
		value: &[[T; X]; Y],
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let tensor = Tensor::new_empty_on(&[Y, X], T::dtype, self.device.clone())?;

		let val_ptr = value.as_ptr() as *const u8;
		let val_len = Y * X * std::mem::size_of::<T>();
		let val = unsafe { std::slice::from_raw_parts(val_ptr, val_len) };
		let mut reader = std::io::Cursor::new(val);
		io::read_bin(&tensor, &mut reader)?;

		Ok(tensor)
	}

	#[inline(never)]
	pub fn new_3d<const Z: usize, const Y: usize, const X: usize>(
		&self,
		value: &[[[T; X]; Y]; Z],
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let tensor = Tensor::new_empty_on(&[Z, Y, X], T::dtype, self.device.clone())?;

		let val_ptr = value.as_ptr() as *const u8;
		let val_len = Z * Y * X * std::mem::size_of::<T>();
		let val = unsafe { std::slice::from_raw_parts(val_ptr, val_len) };
		let mut reader = std::io::Cursor::new(val);
		io::read_bin(&tensor, &mut reader)?;

		Ok(tensor)
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
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorOpError {
	DimsDontMatch,
	TooManyMergedDimensions,
	CannotBorrow,
	CannotBorrowMut,
	MissingVecDimension,
	ExecutorError,
	DimIndexOutOfBounds,
	IndexOutOfBounds,
	ElementsOverflow,
	NotEnoughDimensions,
	NewBufUnsupportedDType,
	NewBufAllocationFailed,
	IncompatibleStridesForMerge,
}

impl TensorOpError {
	#[cold]
	#[inline(never)]
	pub fn missing_vec_dimension() -> ErrPack<Self> {
		let message = "At least one dimension is required for vector-wise operations".into();
		ErrPack {
			code: Self::MissingVecDimension,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}
}

impl From<DimMergerError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: DimMergerError) -> Self {
		match err {
			DimMergerError::DimsDontMatch => Self::DimsDontMatch,
			DimMergerError::TooManyMergedDimensions => Self::TooManyMergedDimensions,
		}
	}
}

impl From<DimMergerError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: DimMergerError) -> Self {
		Self { code: err.into(), extra: None }
	}
}

impl From<BorrowError> for TensorOpError {
	fn from(_: BorrowError) -> Self {
		Self::CannotBorrow
	}
}

impl From<BorrowError> for ErrPack<TensorOpError> {
	fn from(_: BorrowError) -> Self {
		Self {
			code: TensorOpError::CannotBorrow,
			extra: None,
		}
	}
}

impl From<BorrowMutError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(_: BorrowMutError) -> Self {
		Self::CannotBorrowMut
	}
}

impl From<BorrowMutError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(_: BorrowMutError) -> Self {
		Self {
			code: TensorOpError::CannotBorrowMut,
			extra: None,
		}
	}
}

impl From<ErrPack<ExecutorError>> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: ErrPack<ExecutorError>) -> Self {
		Self {
			code: TensorOpError::ExecutorError,
			extra: Some(Box::new(ErrExtra {
				message: String::new(),
				nested: Some(err.into()),
			})),
		}
	}
}

impl From<NewDeviceBufferError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: NewDeviceBufferError) -> Self {
		match err {
			NewDeviceBufferError::AllocationFailed => Self::NewBufAllocationFailed,
			NewDeviceBufferError::UnsupportedDType => Self::NewBufUnsupportedDType,
		}
	}
}

impl From<NewDeviceBufferError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: NewDeviceBufferError) -> Self {
		Self { code: err.into(), extra: None }
	}
}

impl From<NotEnoughDimensionsError> for TensorOpError {
	fn from(_: NotEnoughDimensionsError) -> Self {
		Self::NotEnoughDimensions
	}
}

impl From<NotEnoughDimensionsError> for ErrPack<TensorOpError> {
	fn from(_: NotEnoughDimensionsError) -> Self {
		Self {
			code: TensorOpError::NotEnoughDimensions,
			extra: None,
		}
	}
}

impl From<MergeDimsError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: MergeDimsError) -> Self {
		match err {
			MergeDimsError::NotEnoughDimensions => Self::NotEnoughDimensions,
			MergeDimsError::IncompatibleStrides => Self::IncompatibleStridesForMerge,
		}
	}
}

impl From<MergeDimsError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: MergeDimsError) -> Self {
		Self { code: err.into(), extra: None }
	}
}

impl From<IncompatibleStridesError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(_: IncompatibleStridesError) -> Self {
		Self::IncompatibleStridesForMerge
	}
}

impl From<IncompatibleStridesError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: IncompatibleStridesError) -> Self {
		Self { code: err.into(), extra: None }
	}
}

impl From<ReshapeLastDimError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: ReshapeLastDimError) -> Self {
		match err {
			ReshapeLastDimError::NotEnoughDimensions => Self::NotEnoughDimensions,
			ReshapeLastDimError::InvalidNumElements => Self::ElementsOverflow,
		}
	}
}

impl From<ReshapeLastDimError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: ReshapeLastDimError) -> Self {
		Self { code: err.into(), extra: None }
	}
}

impl From<ElementsOverflowError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(_: ElementsOverflowError) -> Self {
		Self::ElementsOverflow
	}
}

impl From<ElementsOverflowError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(_: ElementsOverflowError) -> Self {
		Self {
			code: TensorOpError::ElementsOverflow,
			extra: None,
		}
	}
}

impl From<ReplaceTailError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: ReplaceTailError) -> Self {
		match err {
			ReplaceTailError::ElementsOverflow => Self::ElementsOverflow,
			ReplaceTailError::NotEnoughDimensions => Self::NotEnoughDimensions,
		}
	}
}

impl From<ReplaceTailError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: ReplaceTailError) -> Self {
		Self { code: err.into(), extra: None }
	}
}

impl From<SelectError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(err: SelectError) -> Self {
		match err {
			SelectError::DimIndexOutOfBounds => Self::DimIndexOutOfBounds,
			SelectError::IndexOutOfBounds => Self::IndexOutOfBounds,
		}
	}
}

impl From<SelectError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: SelectError) -> Self {
		Self { code: err.into(), extra: None }
	}
}

//--------------------------------------------------------------------------------------------------
