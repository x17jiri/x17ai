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

#[cfg(test)]
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

	pub unsafe fn buf_ptr<T: HasDType>(&self) -> *const T {
		let buf = self.buf();
		debug_assert!(CPUDevice::ensure_can_view::<T>(buf.device_buffer()).is_ok());
		buf.device_data.cast()
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

	pub unsafe fn buf_ptr_mut<T: HasDType>(&mut self) -> *mut T {
		let buf = self.buf();
		debug_assert!(CPUDevice::ensure_can_view::<T>(buf.device_buffer()).is_ok());
		buf.device_data.cast()
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
	) -> Result<Self, ErrPack<TensorOpError>> {
		let (map, elems) = DD::new(shape)?;
		let buf = device.new_buffer(dtype, elems)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Self::new_unchecked(map, buf) })
	}

	/// Allocate a new tensor on the same device as `self`.
	pub fn new_empty(&self, shape: &[usize], dtype: DType) -> Result<Self, ErrPack<TensorOpError>> {
		Self::new_empty_on(shape, dtype, self.device())
	}

	/// Allocate a new tensor on the same device with the same shape and dtype as `self`.
	pub fn new_empty_like(&self) -> Result<Self, ErrPack<TensorOpError>> {
		let (map, elems) = self.map().new_like();
		let buf = self.device().new_buffer(self.buf().dtype, elems)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Self::new_unchecked(map, buf) })
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
	pub fn reuse_or_new_like(&self) -> Result<Self, ErrPack<TensorOpError>> {
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
	) -> Result<Self, ErrPack<TensorOpError>> {
		let (map, elems) = self.map().new_replace_tail(tail_len, replace_with)?;
		let buf = self.device().new_buffer(self.buf().dtype, elems)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Self::new_unchecked(map, buf) })
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

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorOpError {
	DimsDontMatch,
	TooManyMergedDimensions,
	CannotBorrow,
	CannotBorrowMut,
	MissingReduceDimension,
	ExecutorError,
	DimIndexOutOfBounds,
	IndexOutOfBounds,
	ElementsOverflow,
	NotEnoughDimensions,
	NewBufUnsupportedDType,
	NewBufAllocationFailed,
	IncompatibleStridesForMerge,
	InvalidValue,
}

impl TensorOpError {
	#[cold]
	#[inline(never)]
	pub fn missing_reduce_dimension() -> ErrPack<Self> {
		let message = "At least one dimension is required for reducing operations".into();
		ErrPack {
			code: Self::MissingReduceDimension,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}

	#[cold]
	#[inline(never)]
	pub fn not_contiguous() -> ErrPack<Self> {
		ExecutorError::not_contiguous().into()
	}

	#[cold]
	#[inline(never)]
	pub fn not_contiguous_or_broadcasted() -> ErrPack<Self> {
		ExecutorError::not_contiguous_or_broadcasted().into()
	}

	#[cold]
	#[inline(never)]
	pub fn invalid_shape() -> ErrPack<Self> {
		let err = ErrPack {
			code: ExecutorError::InvalidShape,
			extra: None,
		};
		err.into()
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

impl From<ViewError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(err: ViewError) -> Self {
		Self {
			code: TensorOpError::ExecutorError,
			extra: Some(Box::new(ErrExtra {
				message: String::new(),
				nested: Some(Box::new(ErrPack::<ExecutorError>::from(err))),
			})),
		}
	}
}

//--------------------------------------------------------------------------------------------------
