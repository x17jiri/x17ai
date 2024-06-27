// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::{cold_path, Error};
use smallvec::SmallVec;
use std::intrinsics::{likely, unlikely};

// If the smaller dim has abs(stride) == 1, we can have MAX_LOCAL_DIMS dims.
// If the smaller dim has abs(stride) > 1, we can have MAX_LOCAL_DIMS - 1 dims.
pub const MAX_LOCAL_DIMS: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct Dim {
	pub len: usize,
	pub stride: isize, // Negative stride means the dimension is reversed
}

#[derive(Clone, Debug)]
pub struct Shape {
	__off: usize,
	__ndim: u8,
	__dims: [Dim; MAX_LOCAL_DIMS],

	// Indices of dimensions sorted from the smallest to the largest stride
	__perm: [u8; MAX_LOCAL_DIMS],
}

impl Shape {
	// Creates a new shape with the given dimension sizes.
	// Strides are initialized so that the shape is contiguous,
	// the first dimension has the largest stride, and the last
	// dimension has stride 1.
	// For example:
	//     shape = Shape::new(&[2, 3]);
	// The memory layout of the shape is:
	//     [[0, 1, 2],
	//      [3, 4, 5]]
	pub fn new(dims: &[usize]) -> Result<Shape, Error> {
		// Having more than MAX_LOCAL_DIMS dimensions is not implemented yet
		let ndim = dims.len();
		if ndim > MAX_LOCAL_DIMS {
			cold_path();
			return Err(Error::TooManyDims);
		}

		// Initialize array with dimensions
		let dim = Dim { len: 1, stride: 1 };
		let mut dim_vec = [dim; MAX_LOCAL_DIMS];

		// Total number of elements in the dimensions processed so far
		let mut elems = 1;

		// Total number of elements in the dimensions processed so far,
		// ignoring zero length dimensions.
		let mut nonzero_elems: isize = 1;

		for i in (0..ndim).rev() {
			// dimension len must be a positive number that fits into isize
			let len: Option<isize> = dims[i].try_into().ok();
			let Some(len) = len else {
				cold_path();
				return Err(Error::TooManyElems);
			};

			// Check that if we ignore zero length dimensions, the number of elements
			// does not overflow. This is done to make sure our calculations would not overflow
			// even if we had the same dimensions but in different order.
			if likely(len > 0) {
				let Some(mul) = nonzero_elems.checked_mul(len) else {
					cold_path();
					return Err(Error::TooManyElems);
				};
				nonzero_elems = mul;
			}

			// Initialize the dimension
			dim_vec[i].len = len as usize;
			dim_vec[i].stride = elems;
			elems *= len;
		}

		debug_assert!(ndim <= u8::MAX as usize);
		Ok(Shape {
			__off: 0,
			__ndim: ndim as u8,
			__dims: dim_vec,
			__perm: [3, 2, 1, 0],
		})
	}

	// Returns the permutation of dimensions that sorts them
	// from the smallest to the largest stride.
	// Length of the returned slice is equal to the number of dimensions.
	pub fn perm(&self) -> &[u8] {
		unsafe {
			let n = self.__ndim as usize;
			let ptr = self.__perm.as_ptr().add(MAX_LOCAL_DIMS - n);
			std::slice::from_raw_parts(ptr, n)
		}
	}

	pub fn ndim(&self) -> usize {
		self.__ndim as usize
	}

	pub fn dims(&self) -> &[Dim] {
		unsafe {
			let n = self.__ndim as usize;
			let ptr = self.__dims.as_ptr();
			std::slice::from_raw_parts(ptr, n)
		}
	}

	pub fn off(&self) -> usize {
		self.__off
	}
}

pub fn prep_op<const N: usize>(inputs: [&Shape; N]) -> Option<(Shape, Traversal<N>, usize)> {
	assert!(N > 0);
	let mut out_shape = inputs[0].clone();
	let ndim = out_shape.ndim();
	let perm = out_shape.perm();
	let mut traversal = Traversal::new(inputs);

	// Check if the number of dimensions is the same for all inputs
	for n in 1..N {
		if unlikely(inputs[n].ndim() != ndim) {
			return None;
		}
	}

	// If input shapes are 0-dimensional, i.e., scalar,
	// return a 1-dimensional shape with 1 element
	if unlikely(ndim == 0) {
		traversal.push_dim(1, [1; N]);
		return Some((out_shape, traversal, 1));
	}

	let mut elems = 1;

	for perm_index in 0..ndim {
		let dim = perm[perm_index] as usize;
		let len = out_shape.dims()[dim].len;

		// Set the stride of the output shape so that it is contiguous
		if likely(out_shape.dims()[dim].stride >= 0) {
			out_shape.dims()[dim].stride = elems;
		} else {
			out_shape.dims()[dim].stride = -elems;
			out_shape.__off += ((len as isize) - 1) * elems;
		}
		elems *= len;

		// Collect the strides of the input shapes
		let mut strides = [0; N];
		for n in 0..N {
			if unlikely(inputs[n].dims()[dim].len != len) {
				return None;
			}
			strides[n] = inputs[n].dims()[dim].stride;
		}

		// Push the dimension to the traversal
		traversal.push_or_merge(len, strides);
	}

	Some((out_shape, traversal, elems as usize))
}

pub struct LenAndStrides<const N: usize> {
	len: usize,
	strides: [isize; N],
}

pub struct Traversal<const N: usize> {
	off: [usize; N],
	dims: SmallVec<[LenAndStrides; MAX_LOCAL_DIMS]>,
}

impl<const N: usize> Traversal<N> {
	pub fn new(shapes: [&Shape; N]) -> Traversal<N> {
		let mut off = [0; N];
		for n in 0..N {
			off[n] = shapes[n].__off;
		}
		Traversal {
			off,
			dims: SmallVec::new(),
		}
	}

	pub fn push_dim(&mut self, len: usize, strides: [isize; N]) {
		self.dims.push(LenAndStrides { len, strides });
	}

	pub fn dim_to_merge(&self, len: usize, strides: [isize; N]) -> Option<&LenAndStrides<N>> {
		if self.dims.is_empty() {
			return None;
		}

		let prev = self.dims.last().unwrap();

		let mut contiguous = true;
		let mut same_sign = true;
		for i in 0..N {
			// Check if dim and prev are contiguous
			let real_stride = strides[i].unsigned_abs();
			let expected_stride = (prev.strides[i] * prev.len as isize).unsigned_abs();
			contiguous &= real_stride == expected_stride;

			// Check if all strides for dim have the same sign
			same_sign &= prev.strides[0] ^ prev.strides[i] >= 0;
			same_sign &= strides[0] ^ strides[i] >= 0;
		}

		if contiguous & same_sign {
			Some(prev)
		} else {
			None
		}
	}

	pub fn push_or_merge(&self, len: usize, strides: [isize; N]) {
		if let Some(prev_dim) = self.dim_to_merge(len, strides) {
			if prev_dim.strides[0] < 0 {
				for i in 0..N {
					self.off[i] += (prev_dim.len as isize - 1) * prev_dim.strides[i];
				}
			}
			if strides[0] < 0 {
				for i in 0..N {
					self.off[i] += (len as isize - 1) * strides[i];
				}
			}

			prev_dim.len *= len;
		} else {
			self.push_dim(len, strides);
		}
	}
}
