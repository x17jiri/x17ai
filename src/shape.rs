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
	pub fn new_scalar() -> Shape {
		Shape {
			__off: 0,
			__ndim: 0,
			__dims: [Dim { len: 1, stride: 1 }; MAX_LOCAL_DIMS],
			__perm: [3, 2, 1, 0],
		}
	}

	// Creates a new shape with the given dimension sizes.
	// Strides are initialized so that the shape is contiguous,
	// the first dimension has the largest stride, and the last
	// dimension has stride 1.
	// For example:
	//     shape = Shape::new(&[2, 3]);
	// The memory layout of the shape is:
	//     [[0, 1, 2],
	//      [3, 4, 5]]
	pub fn new(dims: &[usize]) -> Result<(Shape, usize), Error> {
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
		Ok((
			Shape {
				__off: 0,
				__ndim: ndim as u8,
				__dims: dim_vec,
				__perm: [3, 2, 1, 0],
			},
			elems as usize,
		))
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

	pub fn slice<R: std::ops::RangeBounds<usize>>(&self, dim: usize, range: R) -> Shape {
		let b = match range.start_bound() {
			std::ops::Bound::Included(&start) => start,
			std::ops::Bound::Excluded(&start) => start + 1,
			std::ops::Bound::Unbounded => 0,
		};
		let e = match range.end_bound() {
			std::ops::Bound::Included(&end) => end + 1,
			std::ops::Bound::Excluded(&end) => end,
			std::ops::Bound::Unbounded => self.dims()[dim].len,
		};
		let mut result = self.clone();
		let dims = result.dims_mut();
		let dim = &mut dims[dim];
		if e <= b {
			dim.len = 0;
		} else {
			dim.len = e - b;
			result.__off += (b as isize) * dim.stride;
		}
		result
	}
}

pub fn prep_op<const N: usize>(inputs: [&Shape; N]) -> Option<(Traversal<N>, Shape)> {
	assert!(N > 0);
	let ndim = inputs[0].ndim();
	let perm = inputs[0].perm();

	// Check if the number of dimensions is the same for all inputs
	for n in 1..N {
		if unlikely(inputs[n].ndim() != ndim) {
			return None;
		}
	}

	let mut traversal = Traversal::new(inputs);
	let mut out_shape = inputs[0].clone();

	for perm_index in 0..ndim {
		let i = perm[perm_index] as usize;
		let dim = inputs[0].dims()[i];

		// Set the stride of the output shape so that it is contiguous
		out_shape.dims_mut()[i].stride = traversal.elems as isize;
		if unlikely(dim.stride < 0) {
			out_shape.dims_mut()[i].stride *= -1;
		}

		// Collect the strides of the input shapes
		let mut strides = [0; N];
		for n in 0..N {
			if unlikely(inputs[n].dims()[i].len != dim.len) {
				return None;
			}
			strides[n] = inputs[n].dims()[i].stride;
		}

		// Push the dimension to the traversal
		traversal.push_dim(dim.len, strides);
	}

	traversal.finalize();
	out_shape.__off = traversal.out_off;
	Some((traversal, out_shape))
}

pub fn prep_op_1(input: &Shape) -> (Traversal<1>, Shape) {
	unsafe {
		// SAFETY: prep_op() will fail if inputs are not compatible
		// and we are passing only one input
		prep_op([&input]).unwrap_unchecked()
	}
}

//--------------------------------------------------------------------------------------------------

// We always make sure that:
//     out_stride >= 0
//     in_strides[0] >= 0
// even if it means flipping all in_strides[1..N].
#[derive(Clone, Copy, Debug)]
pub struct TraversalDim<const N: usize> {
	pub len: usize,
	pub out_stride: isize,
	pub in_strides: [isize; N],
}

pub struct Traversal<const N: usize> {
	pub out_off: isize,
	pub in_off: [isize; N],
	pub elems: usize,
	dims: SmallVec<[TraversalDim<N>; MAX_LOCAL_DIMS]>,
}

impl<const N: usize> Traversal<N> {
	pub fn new(shapes: [&Shape; N]) -> Traversal<N> {
		let mut in_off = [0; N];
		for n in 0..N {
			in_off[n] = shapes[n].__off;
		}
		Traversal {
			out_off: 0,
			in_off,
			elems: 1,
			dims: SmallVec::new(),
		}
	}

	pub fn dims(&self) -> &[TraversalDim<N>] {
		self.dims.as_slice()
	}

	fn can_merge(&self, prev: TraversalDim<N>, next: TraversalDim<N>) -> bool {
		let mut can = true;

		// Check if prev and next are contiguous
		for i in 0..N {
			let real_stride = next.in_strides[i].unsigned_abs();
			let expected_stride = (prev.in_strides[i] * prev.len as isize).unsigned_abs();
			let contiguous = real_stride == expected_stride;

			can &= contiguous;
		}

		// Check if all strides are positive
		for i in 1..N {
			let both_positive = (prev.in_strides[i] | next.in_strides[i]) >= 0;
			can &= both_positive;
		}

		can
	}

	pub fn push_dim(&mut self, len: usize, mut in_strides: [isize; N]) {
		if len == 1 {
			return;
		}

		// Prepare the new dimension
		// Make sure that out_stride >= 0 and in_strides[0] >= 0
		let out_stride = self.elems as isize;
		if unlikely(in_strides[0] < 0) {
			self.out_off += (len as isize - 1) * out_stride;

			// flip the dimension for all inputs
			for i in 0..N {
				self.in_off[i] += (len as isize - 1) * in_strides[i];
				in_strides[i] = -in_strides[i];
			}
		}
		self.elems *= len;
		let next = TraversalDim { len, out_stride, in_strides };

		// If there already are dimensions, try to merge the new dimension with the last one
		if !self.dims.is_empty() {
			let prev = *self.dims.last().unwrap();

			if self.can_merge(prev, next) {
				let mut merged = prev;
				merged.len *= next.len;
				*self.dims.last_mut().unwrap() = merged;
				return;
			}
		}

		// Can't merge, push the new dimension
		self.dims.push(next);
	}

	pub fn finalize(&mut self) {
		// If we have no dimensions, add a dummy dimension
		if self.dims.is_empty() {
			self.dims.push(TraversalDim {
				len: 1,
				out_stride: 1,
				in_strides: [1; N],
			});
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct MatMul {
	pub a_rows: usize,
	pub a_cols: usize,
	pub b_cols: usize,

	pub a_off: isize,
	pub b_off: isize,

	pub a_row_stride: isize,
	pub a_col_stride: isize,

	pub b_row_stride: isize,
	pub b_col_stride: isize,
}

impl MatMul {
	pub fn new(a: &Shape, b: &Shape) -> MatMul {
		if a.ndim() != 2 || b.ndim() != 2 {
			panic!("MatMul requires 2D tensors");
		}

		let a_row_dim = a.dims()[0];
		let a_col_dim = a.dims()[1];
		let b_row_dim = b.dims()[0];
		let b_col_dim = b.dims()[1];

		if a_col_dim.len != b_row_dim.len {
			panic!("MatMul shapes do not match");
		}

		MatMul {
			a_rows: a_row_dim.len,
			a_cols: a_col_dim.len,
			b_cols: b_col_dim.len,

			a_off: a.__off,
			b_off: b.__off,

			a_row_stride: a_row_dim.stride,
			a_col_stride: a_col_dim.stride,

			b_row_stride: b_row_dim.stride,
			b_col_stride: b_col_dim.stride,
		}
	}
}
