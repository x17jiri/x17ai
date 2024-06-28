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
	__off: isize,
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

	pub fn perm_mut(&mut self) -> &mut [u8] {
		unsafe {
			let n = self.__ndim as usize;
			let ptr = self.__perm.as_mut_ptr().add(MAX_LOCAL_DIMS - n);
			std::slice::from_raw_parts_mut(ptr, n)
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

	pub fn dims_mut(&mut self) -> &mut [Dim] {
		unsafe {
			let n = self.__ndim as usize;
			let ptr = self.__dims.as_mut_ptr();
			std::slice::from_raw_parts_mut(ptr, n)
		}
	}

	pub fn off(&self) -> isize {
		self.__off
	}

	// Transposes the shape by swapping dimensions dim1 and dim2.
	// Complexity: O(ndim)
	pub fn transpose(&mut self, dim1: usize, dim2: usize) {
		let ndim = self.ndim();
		debug_assert!(dim1 < ndim);
		debug_assert!(dim2 < ndim);

		self.dims_mut().swap(dim1, dim2);

		let perm = self.perm_mut();
		let mut p1 = 0;
		let mut p2 = 0;
		for i in 0..ndim {
			if perm[i] as usize == dim1 {
				p1 = i;
			}
			if perm[i] as usize == dim2 {
				p2 = i;
			}
		}
		perm.swap(p1, p2);
	}
}

pub fn prep_op<const N: usize>(inputs: [&Shape; N]) -> Option<Traversal<N>> {
	assert!(N > 0);
	let ndim = inputs[0].ndim();
	let perm = inputs[0].perm();
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
		traversal.push_dim(1, 1, [1; N]);
		return Some(traversal);
	}

	for perm_index in 0..ndim {
		let i = perm[perm_index] as usize;
		let dim = inputs[0].dims()[i];

		// Set the stride of the output shape so that it is contiguous
		let out_stride = if likely(dim.stride >= 0) {
			traversal.elems as isize
		} else {
			traversal.out_off += ((dim.len as isize) - 1) * (traversal.elems as isize);
			-(traversal.elems as isize)
		};

		// Collect the strides of the input shapes
		let mut strides = [0; N];
		for n in 0..N {
			if unlikely(inputs[n].dims()[i].len != dim.len) {
				return None;
			}
			strides[n] = inputs[n].dims()[i].stride;
		}

		// Push the dimension to the traversal
		traversal.push_or_merge(dim.len, strides);
	}

	Some(traversal)
}

#[derive(Clone, Copy, Debug)]
pub struct LenAndStrides<const N: usize> {
	pub len: usize,
	pub strides: [isize; N],
}

pub struct Traversal<const N: usize> {
	pub off: [isize; N],
	pub elems: usize,
	pub dims: SmallVec<[LenAndStrides<N>; MAX_LOCAL_DIMS]>,
}

impl<const N: usize> Traversal<N> {
	pub fn new(shapes: [&Shape; N]) -> Traversal<N> {
		let mut off = [0; N];
		for n in 0..N {
			off[n] = shapes[n].__off;
		}
		Traversal {
			off,
			elems: 1,
			dims: SmallVec::new(),
		}
	}

	pub fn can_merge(&self, prev: LenAndStrides<N>, next: LenAndStrides<N>) -> bool {
		let mut can = true;
		for i in 0..N {
			// Check if dim and prev are contiguous
			let real_stride = next.strides[i].unsigned_abs();
			let expected_stride = (prev.strides[i] * prev.len as isize).unsigned_abs();
			let contiguous = real_stride == expected_stride;

			// Check if all strides for dim have the same sign
			let prev_same_sign = prev.strides[0] ^ prev.strides[i] >= 0;
			let next_same_sign = next.strides[0] ^ next.strides[i] >= 0;

			can &= contiguous & prev_same_sign & next_same_sign;
		}
		can
	}

	pub fn push_or_merge(&mut self, len: usize, strides: [isize; N]) {
		if !self.dims.is_empty() {
			let prev = *self.dims.last().unwrap();
			let next = LenAndStrides { len, strides };
			if self.can_merge(prev, next) {
				if prev.strides[0] < 0 {
					for i in 0..N {
						self.off[i] += (prev.len as isize - 1) * prev.strides[i];
					}
				}
				if next.strides[0] < 0 {
					for i in 0..N {
						self.off[i] += (len as isize - 1) * strides[i];
					}
				}

				let mut merged = prev;
				merged.len *= len;
				*self.dims.last_mut().unwrap() = merged;
				return;
			}
		}

		// can't merge, push a new dimension
		self.push_dim(len, strides);
	}

	pub fn push_dim(&mut self, len: usize, strides: [isize; N]) {
		self.elems *= len;
		self.dims.push(LenAndStrides { len, out_stride, strides });
	}

}
