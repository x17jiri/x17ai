//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

pub use device::{DType, Device, HasDType};

use crate::rng::Rng;
use crate::tensor::device::dtype::DTypeMismatch;
use crate::tensor::device::kernel::expr::EvaluatesToTensor;
use crate::tensor::device::{DeviceBase, DeviceBuffer, NewDeviceBufferError};
use crate::tensor::dim_merger::{DimMergerError, DimsDontMatchError, TooManyMergedDimensionsError};
use crate::tensor::generic::dim_index::DimIndexOutOfBoundsError;
use crate::tensor::generic::map::dd::ReplaceTailError;
use crate::tensor::generic::map::{
	DD, ElementsOverflowError, IncompatibleStridesError, IndexOutOfBoundsError, MergeDimsError, ND,
	NotEnoughDimensionsError, ReshapeLastDimError, SelectError,
};
use crate::tensor::generic::{GenericTensor, TensorUnsafeError};
use crate::tensor::io::merge_dims;
use crate::util::LossyFrom;
use crate::util::mycell::{self, BorrowError, BorrowGuard, BorrowMutError, BorrowMutGuard};
use crate::{ErrExtra, ErrPack};

pub mod device;
pub mod dim_merger;
pub mod generic;
pub mod io;
pub mod math;

#[cfg(test)]
mod tests;

//--------------------------------------------------------------------------------------------------

impl<M: generic::map::Map> GenericTensor<M, Rc<mycell::RefCell<DeviceBuffer>>> {
	/// # Errors
	/// `BorrowError` if there is a mutable borrow preventing a shared borrow.
	pub fn borrow<'buf>(
		&'buf self,
	) -> std::result::Result<
		GenericTensor<&'buf M::Deref, BorrowGuard<'buf, DeviceBuffer>>,
		BorrowError,
	> {
		let map = self.map().as_ref();
		let buf = self.buf().try_borrow()?;
		// SAFETY: We only change the type of buffer reference.
		// So if the map was safe before, it is still safe.
		Ok(unsafe { GenericTensor::new_unchecked(map, buf) })
	}

	/// # Errors
	/// `BorrowMutError` if there already is any other borrow of the buffer.
	pub fn borrow_mut<'buf>(
		&'buf self,
	) -> std::result::Result<
		GenericTensor<&'buf M::Deref, BorrowMutGuard<'buf, DeviceBuffer>>,
		BorrowMutError,
	> {
		let map = self.map().as_ref();
		let buf = self.buf().try_borrow_mut()?;
		// SAFETY: We only change the type of buffer reference.
		// So if the map was safe before, it is still safe.
		Ok(unsafe { GenericTensor::new_unchecked(map, buf) })
	}
}

//--------------------------------------------------------------------------------------------------

pub type Tensor = GenericTensor<DD, Rc<mycell::RefCell<DeviceBuffer>>>;

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
		Self::new_empty_on(shape, dtype, self.rc_device())
	}

	/// Allocate a new tensor on the same device and with the same shape as `self`.
	pub fn new_empty_like(&self, dtype: DType) -> Result<Self, ErrPack<TensorOpError>> {
		let (map, elems) = self.map().new_like();
		let buf = self.rc_device().new_buffer(dtype, elems)?;

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
		if self.owns_buffer() { Ok(self.clone()) } else { self.new_empty_like(self.dtype()) }
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
		dtype: DType,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let (map, elems) = self.map().new_replace_tail(tail_len, replace_with)?;
		let buf = self.rc_device().new_buffer(dtype, elems)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Self::new_unchecked(map, buf) })
	}

	/// Returns float at index [0, 0, ..., 0].
	pub fn scalar(&self) -> Result<f64, ErrPack<TensorOpError>> {
		if self.elems() == 0 {
			cold_path();
			return Err(IndexOutOfBoundsError.into());
		}
		self.ensure_safe()?;

		let (mut map, _elems) = ND::new(&[])?;
		map.offset = self.map().offset;
		let buf = self.buf().try_borrow()?;
		unsafe { self.device().read_float((map, &*buf)) }
	}

	/// Sometimes we want to calculate the mean of the last dimension,
	/// but our custom kernels only support `sum()`.
	///
	/// This function returns the factor by which we need to multiply
	/// the sum to get the mean.
	///
	/// I.e., it returns `1.0 / self.size(-1)`.
	///
	/// If the tensor has no dimensions, the return value is 1.0.
	pub fn sum_to_mean(&self) -> f64 {
		let n = self.size(-1).unwrap_or(1);
		1.0 / f64::lossy_from(n)
	}

	pub fn device_base(&self) -> &DeviceBase {
		self.buf().device_base()
	}

	pub fn device(&self) -> &dyn Device {
		unsafe { self.device_base().device() }
	}

	/// Returns the device on which the tensor is allocated.
	pub fn rc_device(&self) -> Rc<dyn Device> {
		unsafe { self.device_base().rc_device() }
	}

	/// Returns the data type of the tensor elements.
	pub fn dtype(&self) -> DType {
		self.buf().dtype()
	}

	pub fn assign<Expr: EvaluatesToTensor>(
		&self,
		expr: Expr,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.eval_to_tensor(self)
	}

	#[inline(never)]
	pub fn randn_(&self, rng: &mut Rng) -> Result<(), ErrPack<TensorOpError>> {
		#[allow(clippy::single_match_else)]
		match self.dtype() {
			f32::dtype => {
				let mut array = vec![0.0_f32; self.elems()];
				rng.randn(&mut array);
				let bytes = unsafe {
					std::slice::from_raw_parts(
						array.as_ptr().cast::<u8>(),
						std::mem::size_of_val(array.as_slice()),
					)
				};
				self.load_from_cpu_memory(bytes)?;
				Ok(())
			},
			_ => {
				cold_path();
				todo!("Implement tensor.randn_() for other dtypes");
			},
		}
	}

	pub fn store_to_cpu_memory(&self, dst: &mut [u8]) -> Result<(), ErrPack<TensorOpError>> {
		let nd = merge_dims::<1>(self)?;
		if !nd.dims[0].is_contiguous() {
			cold_path();
			return Err(TensorOpError::not_contiguous());
		}
		let count = nd.dims[0].size;
		if dst.len() != unsafe { self.dtype().array_bytes_unchecked(count) } {
			cold_path();
			return Err(TensorOpError::invalid_buffer_size());
		}
		let borrow = self.buf().try_borrow()?;
		let src = (ND::<0> { offset: nd.offset, dims: [] }, &*borrow);
		let dst = NonNull::from_ref(dst).cast::<u8>();
		unsafe { self.device().store_to_cpu_memory(src, dst, count) }
	}

	pub fn load_from_cpu_memory(&self, src: &[u8]) -> Result<(), ErrPack<TensorOpError>> {
		let nd = merge_dims::<1>(self)?;
		if !nd.dims[0].is_contiguous() {
			cold_path();
			return Err(TensorOpError::not_contiguous());
		}
		let count = nd.dims[0].size;
		if src.len() != unsafe { self.dtype().array_bytes_unchecked(count) } {
			cold_path();
			return Err(TensorOpError::invalid_buffer_size());
		}
		let borrow = self.buf().try_borrow_mut()?;
		let dst = (ND::<0> { offset: nd.offset, dims: [] }, &*borrow);
		let src = NonNull::from_ref(src).cast::<u8>();
		unsafe { self.device().load_from_cpu_memory(src, dst, count) }
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

		let val_ptr = value.as_ptr().cast::<u8>();
		let val_len = X * std::mem::size_of::<T>();
		let val = unsafe { std::slice::from_raw_parts(val_ptr, val_len) };

		tensor.load_from_cpu_memory(val)?;
		Ok(tensor)
	}

	#[inline(never)]
	pub fn new_2d<const Y: usize, const X: usize>(
		&self,
		value: &[[T; X]; Y],
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let tensor = Tensor::new_empty_on(&[Y, X], T::dtype, self.device.clone())?;

		let val_ptr = value.as_ptr().cast::<u8>();
		let val_len = Y * X * std::mem::size_of::<T>();
		let val = unsafe { std::slice::from_raw_parts(val_ptr, val_len) };

		tensor.load_from_cpu_memory(val)?;
		Ok(tensor)
	}

	#[inline(never)]
	pub fn new_3d<const Z: usize, const Y: usize, const X: usize>(
		&self,
		value: &[[[T; X]; Y]; Z],
	) -> Result<Tensor, ErrPack<TensorOpError>> {
		let tensor = Tensor::new_empty_on(&[Z, Y, X], T::dtype, self.device.clone())?;

		let val_ptr = value.as_ptr().cast::<u8>();
		let val_len = Z * Y * X * std::mem::size_of::<T>();
		let val = unsafe { std::slice::from_raw_parts(val_ptr, val_len) };

		tensor.load_from_cpu_memory(val)?;
		Ok(tensor)
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnsupportedDTypeError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorOpError {
	DimsDontMatch,
	TooManyMergedDimensions,
	CannotBorrow,
	CannotBorrowMut,
	MissingReduceDimension,
	DimIndexOutOfBounds,
	IndexOutOfBounds,
	ElementsOverflow,
	NotEnoughDimensions,
	NewBufUnsupportedDType,
	UnsupportedDType,
	NewBufAllocationFailed,
	IncompatibleStridesForMerge,
	InvalidValue,
	CannotBroadcastOutput,
	ShapeMismatch,
	DTypeMismatch,
	UnsafeTensor,
	NotContiguous,
	NotContiguousOrBroadcasted,
	InvalidShape,
	InvalidDType,
	InvalidDevice,
	InvalidBufferSize,
	IOError,
	DeviceError,
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
		let message = "Expected the tensor to have contiguous dimension -1, but it does not".into();
		ErrPack {
			code: Self::NotContiguous,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}

	#[cold]
	#[inline(never)]
	pub fn not_contiguous_or_broadcasted() -> ErrPack<Self> {
		let message =
			"Expected the tensor to have contiguous or broadcasted dimension -1, but it does not"
				.into();
		ErrPack {
			code: Self::NotContiguousOrBroadcasted,
			extra: Some(Box::new(ErrExtra { message, nested: None })),
		}
	}

	pub fn invalid_shape() -> ErrPack<Self> {
		ErrPack { code: Self::InvalidShape, extra: None }
	}

	pub fn invalid_buffer_size() -> ErrPack<Self> {
		ErrPack {
			code: Self::InvalidBufferSize,
			extra: None,
		}
	}

	pub fn cannot_broadcast_output() -> ErrPack<Self> {
		ErrPack {
			code: Self::CannotBroadcastOutput,
			extra: None,
		}
	}

	#[cold]
	#[inline(never)]
	pub fn io_error(err: std::io::Error) -> ErrPack<Self> {
		ErrPack {
			code: Self::IOError,
			extra: Some(Box::new(ErrExtra {
				message: String::new(),
				nested: Some(Box::new(err)),
			})),
		}
	}

	pub fn shape_mismatch() -> ErrPack<Self> {
		ErrPack { code: Self::ShapeMismatch, extra: None }
	}

	pub fn dtype_mismatch() -> ErrPack<Self> {
		ErrPack { code: Self::DTypeMismatch, extra: None }
	}

	#[cold]
	#[inline(never)]
	pub fn device_error(message: &str) -> ErrPack<Self> {
		ErrPack {
			code: Self::DeviceError,
			extra: Some(Box::new(ErrExtra {
				message: message.to_string(),
				nested: None,
			})),
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

impl From<DimsDontMatchError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(_: DimsDontMatchError) -> Self {
		Self::DimsDontMatch
	}
}

impl From<DimsDontMatchError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(_: DimsDontMatchError) -> Self {
		Self {
			code: TensorOpError::DimsDontMatch,
			extra: None,
		}
	}
}

impl From<TooManyMergedDimensionsError> for TensorOpError {
	#[cold]
	#[inline(never)]
	fn from(_: TooManyMergedDimensionsError) -> Self {
		Self::TooManyMergedDimensions
	}
}

impl From<TooManyMergedDimensionsError> for ErrPack<TensorOpError> {
	#[cold]
	#[inline(never)]
	fn from(_: TooManyMergedDimensionsError) -> Self {
		Self {
			code: TensorOpError::TooManyMergedDimensions,
			extra: None,
		}
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

impl From<IndexOutOfBoundsError> for ErrPack<TensorOpError> {
	fn from(_: IndexOutOfBoundsError) -> Self {
		Self {
			code: TensorOpError::IndexOutOfBounds,
			extra: None,
		}
	}
}

impl From<TensorUnsafeError> for TensorOpError {
	fn from(_: TensorUnsafeError) -> Self {
		Self::UnsafeTensor
	}
}

impl From<ErrPack<TensorUnsafeError>> for ErrPack<TensorOpError> {
	fn from(err: ErrPack<TensorUnsafeError>) -> Self {
		Self { code: err.code.into(), extra: err.extra }
	}
}

impl From<std::io::Error> for ErrPack<TensorOpError> {
	fn from(err: std::io::Error) -> Self {
		TensorOpError::io_error(err)
	}
}

impl From<DimIndexOutOfBoundsError> for ErrPack<TensorOpError> {
	fn from(_: DimIndexOutOfBoundsError) -> Self {
		Self {
			code: TensorOpError::DimIndexOutOfBounds,
			extra: None,
		}
	}
}

impl From<DTypeMismatch> for ErrPack<TensorOpError> {
	fn from(_: DTypeMismatch) -> Self {
		Self {
			code: TensorOpError::DTypeMismatch,
			extra: None,
		}
	}
}

impl From<UnsupportedDTypeError> for ErrPack<TensorOpError> {
	fn from(_: UnsupportedDTypeError) -> Self {
		Self {
			code: TensorOpError::UnsupportedDType,
			extra: None,
		}
	}
}

//--------------------------------------------------------------------------------------------------
