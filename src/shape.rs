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
	pub stride: usize,
}

#[derive(Clone)]
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
			let len = dims[i];

			dim_vec[i].len = len;
			dim_vec[i].stride = elems;
			elems *= len;

			// Check that if we ignore zero length dimensions, the number of elements
			// does not overflow. This is done to make sure our calculations would not overflow
			// even if we had the same dimensions but in different order.
			if likely(len > 0) {
				let len: Option<isize> = len.try_into().ok();
				let mul = len.and_then(|len| nonzero_elems.checked_mul(len));
				match mul {
					Some(mul) => {
						nonzero_elems = mul;
					},
					None => {
						cold_path();
						return Err(Error::TooManyElems);
					},
				};
			}
		}

		debug_assert!(ndim <= u8::MAX as usize);
		Ok(Shape {
			__off: 0,
			__ndim: ndim as u8,
			__dims: dim_vec,
			__perm: [3, 2, 1, 0],
		})
	}

	// Creates a new shape with the same dimensions as the given shape, but contiguous.
	// The ordering of strides is the same as in the original shape.
	// The second return value is the total number of elements in the shape.
	// The third return value is true if the shape was already contiguous.
	pub fn to_contiguous(&self) -> (Shape, usize, bool) {
		let mut new_shape = self.clone();
		new_shape.__off = 0;

		let perm = self.perm();

		let mut elems = 1;
		let mut contiguous = true;
		for i in perm.iter() {
			let dim = &mut new_shape.__dims[*i as usize];

			// Note: We don't need `contiguous = contiguous && (dim.stride == elems);`
			contiguous = (dim.stride == elems);
			dim.stride = elems;
			elems *= dim.len;
		}

		(new_shape, elems as usize, contiguous)
	}

	pub fn merge_dims<const N: usize>(shapes: &[&Shape; N]) -> SmallVec<[[Dim; N]; MAX_LOCAL_DIMS]> {
		let perm: [&[u8]; N];
		let dims: [&[Dim]; N];

		for n in 0..N {
			perm[n] = shapes[n].perm();
			dims[n] = shapes[n].dims();
		}

		let mut result = SmallVec::new();
		if N == 0 || unlikely(perm[0].len() == 0) {
			return result;
		}

		let inner_dim = dims[perm[0] as usize];

		result.push(inner_dim);
		let mut prev_dim = result.last_mut().unwrap();

		for i in perm[1..].iter() {
			let mut dim = dims[*i as usize];

			if dim.stride == prev_dim.len * prev_dim.stride {
				// If possible, merge the new dimension with the previous one
				prev_dim.len *= dim.len;
			} else {
				// Otherwise, add the new dimension to the result
				result.push(dim);
				prev_dim = result.last_mut().unwrap();
			}
		}
		result
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
