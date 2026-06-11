use std::hint::{cold_path, likely};

use crate::ShapeOverflowError;
use crate::dtype::DType;

#[derive(Clone, Copy)]
pub struct ShapeHelper<'a> {
	dtype: DType,
	shape: &'a [usize],
	elems: usize,
}

impl<'a> ShapeHelper<'a> {
	pub fn new(dtype: DType, shape: &'a [usize]) -> Result<Self, ShapeOverflowError> {
		let Ok(_ndim) = TryInto::<u8>::try_into(shape.len()) else {
			cold_path();
			return Err(ShapeOverflowError);
		};
		let mut nonzero_elems: usize = 1;
		let mut elems: usize = 1;
		for &dim in shape {
			// When checking for overflow, we ignore size zero dimensions.
			// This is to make sure that overflow doesn't depend on the order of dimensions.
			// Example:
			// - [usize::MAX, usize::MAX, 0] // without ignoring zeros, this overflows
			// - [0, usize::MAX, usize::MAX] // without ignoring zeros, this doesn't overflow
			if likely(dim != 0) {
				let Some(t) = nonzero_elems.checked_mul(dim) else {
					cold_path();
					return Err(ShapeOverflowError);
				};
				nonzero_elems = t;
			}
			elems *= dim;
		}
		let max_bits = usize::MAX - 7;
		let max_elems = max_bits / dtype.bits();
		if nonzero_elems > max_elems {
			cold_path();
			return Err(ShapeOverflowError);
		}
		Ok(ShapeHelper { dtype, shape, elems })
	}

	pub fn ndim(&self) -> u8 {
		// SAFETY: Checked in `new`.
		#[allow(clippy::cast_possible_truncation)]
		(self.shape.len() as u8)
	}

	pub fn shape(&self) -> &[usize] {
		self.shape
	}

	pub fn elems(&self) -> usize {
		self.elems
	}

	pub fn bits(&self) -> usize {
		self.elems * self.dtype.bits()
	}

	pub fn bytes(&self) -> usize {
		// SAFETY: We ensured in `new` that the addition cannot overflow.
		#[allow(clippy::manual_div_ceil)]
		((self.bits() + 7) / 8)
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}
}
