//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ErrPack;
use crate::rng::Rng;
use crate::tensor::device::cpu::cpu_float_methods::FromToF64;
use crate::tensor::dim_merger::ReshapeError;
use crate::tensor::error::{InvalidBufferSizeError, UnsupportedDTypeError};
use crate::tensor::map::SelectError;
use crate::util::LossyFrom;
use crate::util::mycell::{self};
use crate::util::universal_range::UniversalRange;

use self::device::DeviceBuffer;
use self::device::kernel::EvaluatesToTensor;
use self::dim_index::{DimIndex, DimIndexOutOfBoundsError};
use self::error::NotContiguousError;
use self::map::{IndexOutOfBoundsError, Map, SizeAndStride};
use self::shape::Shape;

pub use self::device::{DType, Device, HasDType};
pub use self::error::TensorOpError;

pub mod device;
pub mod dim_merger;
//pub mod generic;
pub mod dim_index;
pub mod error;
pub mod io;
pub mod map;
pub mod math;
pub mod shape;

#[cfg(test)]
mod tests;

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tensor {
	map: Map,
	buf: Rc<mycell::RefCell<DeviceBuffer>>,
}

impl Tensor {
	/// # Safety
	/// The map must be safe, i.e., the span of the map must be within the bounds of the buffer.
	pub unsafe fn new_unchecked(map: Map, buf: Rc<mycell::RefCell<DeviceBuffer>>) -> Self {
		let map_span = map.byte_span();
		let buf_len = buf.byte_len();
		let safe = map_span.start <= map_span.end && map_span.end <= buf_len;
		debug_assert!(safe);
		Self { map, buf }
	}

	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on(
		shape: impl Shape,
		dtype: DType,
		device: Rc<dyn Device>,
	) -> Result<Self, ErrPack<TensorOpError>> {
		let (map, bytes) = shape.to_map(dtype)?;
		let buf = device.new_buffer(bytes)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Self::new_unchecked(map, buf) })
	}

	/// Allocate a new tensor on the same device as `self`.
	pub fn new_empty(
		&self,
		shape: impl Shape,
		dtype: DType,
	) -> Result<Self, ErrPack<TensorOpError>> {
		Self::new_empty_on(shape, dtype, self.rc_device())
	}

	/// Allocate a new tensor on the same device and with the same shape as `self`.
	pub fn new_empty_like(&self, dtype: DType) -> Result<Self, ErrPack<TensorOpError>> {
		let (map, bytes) = self.map().new_like(dtype)?;
		let buf = self.rc_device().new_buffer(bytes)?;

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
		// TODO - should I test weak? Also why do I add 1 to weak count?
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
		let (map, bytes) = self.map().new_replace_tail(tail_len, replace_with, dtype)?;
		let buf = self.rc_device().new_buffer(bytes)?;

		// SAFETY: We created the buffer to be as big as the mapping.
		Ok(unsafe { Self::new_unchecked(map, buf) })
	}

	pub fn map(&self) -> &Map {
		&self.map
	}

	pub fn buf(&self) -> &Rc<mycell::RefCell<DeviceBuffer>> {
		&self.buf
	}

	pub fn ndim(&self) -> usize {
		self.map.ndim()
	}

	pub fn dim<D: DimIndex>(&self, dim: D) -> Result<SizeAndStride, DimIndexOutOfBoundsError> {
		let dim = dim.resolve_index(self.ndim())?;
		Ok(self.map.dim(dim))
	}

	pub fn size<D: DimIndex>(&self, dim: D) -> Result<usize, DimIndexOutOfBoundsError> {
		let dim = dim.resolve_index(self.ndim())?;
		Ok(self.map.dim(dim).size)
	}

	pub fn elems(&self) -> usize {
		self.map.elems()
	}

	/// Returns float at index [0, 0, ..., 0].
	pub fn scalar(&self) -> Result<f64, ErrPack<TensorOpError>> {
		debug_assert!(self.is_map_safe());
		if self.elems() == 0 {
			cold_path();
			return Err(IndexOutOfBoundsError.into());
		}
		let buf = self.buf().try_borrow()?;
		match self.dtype() {
			f32::dtype => unsafe {
				let mut value: f32 = 0.0;
				self.device().download_data(
					&buf,
					NonNull::from_mut(&mut value).cast(),
					self.map().offset_bytes(),
					std::mem::size_of::<f32>(),
				)?;
				Ok(value.to_f64())
			},
			f64::dtype => unsafe {
				let mut value: f64 = 0.0;
				self.device().download_data(
					&buf,
					NonNull::from_mut(&mut value).cast(),
					self.map().offset_bytes(),
					std::mem::size_of::<f64>(),
				)?;
				Ok(value)
			},
			_ => {
				cold_path();
				Err(UnsupportedDTypeError.into())
			},
		}
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

	pub fn device(&self) -> &dyn Device {
		self.buf().device()
	}

	/// Returns the device on which the tensor is allocated.
	pub fn rc_device(&self) -> Rc<dyn Device> {
		self.buf().rc_device()
	}

	/// Returns the data type of the tensor elements.
	pub fn dtype(&self) -> DType {
		self.map().dtype()
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
				self.upload_data(bytes)?;
				Ok(())
			},
			_ => {
				cold_path();
				todo!("Implement tensor.randn_() for other dtypes");
			},
		}
	}

	pub fn download_data(&self, dst: &mut [u8]) -> Result<(), ErrPack<TensorOpError>> {
		let (merged, rest) = dim_merger::merge_dims(self.map.dims());
		if !merged.is_contiguous() || !rest.is_empty() {
			cold_path();
			return Err(NotContiguousError.into());
		}
		let offset_bytes = unsafe { self.dtype().array_bytes_unchecked(self.map.offset()) };
		let size_bytes = unsafe { self.dtype().array_bytes_unchecked(merged.size) };
		if dst.len() != size_bytes {
			cold_path();
			return Err(InvalidBufferSizeError.into());
		}
		let borrow = self.buf().try_borrow()?;
		unsafe {
			self.device().download_data(
				&borrow,
				NonNull::from_ref(dst).cast(),
				offset_bytes,
				size_bytes,
			)
		}
	}

	pub fn upload_data(&self, src: &[u8]) -> Result<(), ErrPack<TensorOpError>> {
		let (merged, rest) = dim_merger::merge_dims(self.map.dims());
		if !merged.is_contiguous() || !rest.is_empty() {
			cold_path();
			return Err(NotContiguousError.into());
		}
		let offset_bytes = unsafe { self.dtype().array_bytes_unchecked(self.map.offset()) };
		let size_bytes = unsafe { self.dtype().array_bytes_unchecked(merged.size) };
		if src.len() != size_bytes {
			cold_path();
			return Err(InvalidBufferSizeError.into());
		}
		let borrow = self.buf().try_borrow_mut()?;
		unsafe {
			self.device().upload_data(
				NonNull::from_ref(src).cast(),
				&borrow,
				offset_bytes,
				size_bytes,
			)
		}
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

	pub fn to_device(&self, device: Rc<dyn Device>) -> Result<Self, ErrPack<TensorOpError>> {
		let dtype = self.dtype();

		let (merged, rest) = dim_merger::merge_dims(self.map.dims());
		if !merged.is_contiguous() || !rest.is_empty() {
			cold_path();
			return Err(NotContiguousError.into());
		}
		let offset_bytes = unsafe { dtype.array_bytes_unchecked(self.map.offset()) };
		let size_bytes = unsafe { dtype.array_bytes_unchecked(merged.size) };
		let borrow = self.buf().try_borrow()?;

		let tensor = Self::new_empty_on(self.map(), dtype, device)?;
		let output_offset_bytes = unsafe { dtype.array_bytes_unchecked(tensor.map.offset()) };

		if self.device().is_cpu() {
			unsafe {
				let src =
					NonNull::new_unchecked(borrow.device_ptr().as_ptr::<u8>()).add(offset_bytes);
				tensor.device().upload_data(src, tensor.buf(), output_offset_bytes, size_bytes)?;
			}
		} else if tensor.device().is_cpu() {
			unsafe {
				let dst = NonNull::new_unchecked(tensor.buf().device_ptr().as_ptr::<u8>())
					.add(output_offset_bytes);
				self.device().download_data(&borrow, dst, offset_bytes, size_bytes)?;
			}
		} else {
			todo!("Implement tensor.to_device() between two non-CPU devices");
		}
		Ok(tensor)
	}

	/// # Errors
	/// - If the map is not safe, i.e., if some index may be mapped to an out-of-bounds offset.
	pub fn is_map_safe(&self) -> bool {
		let span = self.map.byte_span();
		let buf_len = self.buf.byte_len();
		span.start <= span.end && span.end <= buf_len
	}

	pub fn merge_dims(&self, n: usize) -> Result<Self, ErrPack<TensorOpError>> {
		let new_map = self.map.merge_dims(n)?;
		Ok(unsafe { Self::new_unchecked(new_map, self.buf.clone()) })
	}

	pub fn merge_all_dims(&self) -> Result<Self, ErrPack<TensorOpError>> {
		let new_map = self.map.merge_all_dims()?;
		Ok(unsafe { Self::new_unchecked(new_map, self.buf.clone()) })
	}

	pub fn reshape_last_dim(self, to_shape: &[usize]) -> Result<Self, ErrPack<TensorOpError>> {
		let new_map = self.map.reshape_last_dim(to_shape)?;
		Ok(unsafe { Self::new_unchecked(new_map, self.buf) })
	}

	pub fn reshape_dims(&self, n: usize, to_shape: &[usize]) -> Result<Self, ReshapeError> {
		let new_map = self.map.reshape_dims(n, to_shape)?;
		Ok(unsafe { Self::new_unchecked(new_map, self.buf.clone()) })
	}

	pub fn reshape_all_dims(&self, to_shape: &[usize]) -> Result<Self, ReshapeError> {
		let new_map = self.map.reshape_all_dims(to_shape)?;
		Ok(unsafe { Self::new_unchecked(new_map, self.buf.clone()) })
	}

	pub fn select<D: DimIndex>(&self, dim: D, index: usize) -> Result<Self, SelectError> {
		let dim = dim.resolve_index(self.ndim())?;
		let new_map = self.map.select(dim, index)?;
		Ok(unsafe { Self::new_unchecked(new_map, self.buf.clone()) })
	}

	pub fn narrow<D: DimIndex, R: Into<UniversalRange>>(
		self,
		dim: D,
		range: R,
	) -> Result<Self, SelectError> {
		let dim = dim.resolve_index(self.ndim())?;
		let range = range.into();
		let new_map = self.map.narrow(dim, range)?;
		Ok(unsafe { Self::new_unchecked(new_map, self.buf) })
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

		tensor.upload_data(val)?;
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

		tensor.upload_data(val)?;
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

		tensor.upload_data(val)?;
		Ok(tensor)
	}
}

//--------------------------------------------------------------------------------------------------
