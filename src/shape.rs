// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use smallvec::SmallVec;

// If the smaller dim has abs(stride) == 1, we can have MAX_LOCAL_DIMS dims.
// If the smaller dim has abs(stride) > 1, we can have MAX_LOCAL_DIMS - 1 dims.
pub const MAX_LOCAL_DIMS: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct Dim {
	__len: usize,
	__stride: isize, // Note: stride can be negative, so it is signed.
}

#[derive(Clone)]
pub struct Shape {
	__off: usize,
	__ndim: u8,
	__dim: [Dim; MAX_LOCAL_DIMS],

	// Indices of dimensions sorted from the smallest to the largest stride.
	__perm: [u8; MAX_LOCAL_DIMS],
}

impl Shape {
	pub fn new(dims: &[usize]) -> Result<Shape, Error> {
		if dims.len() > MAX_LOCAL_DIMS {
			cold_path();
			return Err(Error::TooManyDims);
		}

		let t = [Dim { __len: 1, __stride: 1 }; MAX_LOCAL_DIMS];
		let mut elems: usize = 1;
		let mut nonzero_elems: usize = 1;
		for i in (0..dims.len()).rev() {
			t[i].__len = dims[i];
			t[i].__stride = elems as isize;
			elems *= dims[i];

			// Check that if we ignore zero length dimensions, the number of elements
			// does not overflow.
			let d = dims[i].max(1);
			let Some(n) = nonzero_elems.checked_mul(d).and_then(|n| n.try_into().ok()) else {
				cold_path();
				return Err(Error::TooManyElems);
			}
			nonzero_elems = n;
		}

		Ok(Shape {
			__off: 0,
			__ndim: dims.len() as u8,
			__dim: t,
			__perm: [3, 2, 1, 0],
		})
	}

	pub fn merge_dims(&self) -> SmallVec<[Dim; MAX_LOCAL_DIMS]> {
		let dims = self.dims();
		let perm = self.perm();
		let mut result = SmallVec::with_capacity(self.ndims());
		for i in perm.iter() {
			
			
		}
	}

	// Returns the permutation of dimensions that sorts them
	// from the smallest to the largest stride.
	// Length of the returned slice is equal to the number of dimensions.
	pub fn perm(&self) -> &[u8] unsafe {
		let n = self.__ndim as usize;
		let ptr = self.__perm.as_ptr().add(MAX_LOCAL_DIMS - n);
		std::slice::from_raw_parts(ptr, n)
	}

	pub fn ndim(&self) -> usize {
		self.__ndim as usize
	}

	pub fn dims(&self) -> &[Dim] unsafe {
		let n = self.__ndim as usize;
		let ptr = self.__dim.as_ptr();
		std::slice::from_raw_parts(ptr, n)
	}
}
