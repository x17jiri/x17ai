use crate::dtype::{DType, HasDType};

pub trait TensorLiteral {
	fn dtype(&self) -> DType;
	fn shape(&self) -> &[usize];
	fn data(&self) -> &[u8];
}

pub struct TensorLiteral1D<'a, T: HasDType> {
	shape: [usize; 1],
	data: &'a [T],
}

impl<'a, T: HasDType> TensorLiteral1D<'a, T> {
	pub fn new<const X: usize>(data: &'a [T; X]) -> Self {
		Self { shape: [X], data }
	}
}

impl<'a, T: HasDType> TensorLiteral for TensorLiteral1D<'a, T> {
	fn dtype(&self) -> DType {
		T::dtype
	}

	fn shape(&self) -> &[usize] {
		&self.shape
	}

	fn data(&self) -> &[u8] {
		unsafe {
			std::slice::from_raw_parts(
				self.data.as_ptr().cast::<u8>(),
				std::mem::size_of_val(self.data),
			)
		}
	}
}

pub struct TensorLiteral2D<'a, T: HasDType> {
	shape: [usize; 2],
	data: &'a [T],
}

impl<'a, T: HasDType> TensorLiteral2D<'a, T> {
	pub fn new<const Y: usize, const X: usize>(data: &'a [[T; X]; Y]) -> Self {
		let flat_data: &'a [T] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), X * Y) };
		Self { shape: [Y, X], data: flat_data }
	}
}

impl<'a, T: HasDType> TensorLiteral for TensorLiteral2D<'a, T> {
	fn dtype(&self) -> DType {
		T::dtype
	}

	fn shape(&self) -> &[usize] {
		&self.shape
	}

	fn data(&self) -> &[u8] {
		unsafe {
			std::slice::from_raw_parts(
				self.data.as_ptr().cast::<u8>(),
				std::mem::size_of_val(self.data),
			)
		}
	}
}

pub struct TensorLiteral3D<'a, T: HasDType> {
	shape: [usize; 3],
	data: &'a [T],
}

impl<'a, T: HasDType> TensorLiteral3D<'a, T> {
	pub fn new<const Z: usize, const Y: usize, const X: usize>(data: &'a [[[T; X]; Y]; Z]) -> Self {
		let flat_data: &'a [T] =
			unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), X * Y * Z) };
		Self { shape: [Z, Y, X], data: flat_data }
	}
}

impl<'a, T: HasDType> TensorLiteral for TensorLiteral3D<'a, T> {
	fn dtype(&self) -> DType {
		T::dtype
	}

	fn shape(&self) -> &[usize] {
		&self.shape
	}

	fn data(&self) -> &[u8] {
		unsafe {
			std::slice::from_raw_parts(
				self.data.as_ptr().cast::<u8>(),
				std::mem::size_of_val(self.data),
			)
		}
	}
}
