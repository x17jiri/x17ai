// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

// If the smaller dim has abs(stride) == 1, we can have MAX_LOCAL_DIMS dims.
// If the smaller dim has abs(stride) > 1, we can have MAX_LOCAL_DIMS - 1 dims.
pub const MAX_LOCAL_DIMS: usize = 4;

#[derive(Debug, Clone, Copy)]
struct Dim {
	__len: usize,
	__stride: isize, // Note: stride can be negative, so it is signed.
}

#[derive(Clone)]
pub struct Shape {
	__off: usize,
	__ndim: u8,
	__dim: [Dim; MAX_LOCAL_DIMS],

	// Indices of dimensions sorted from the largest to the smallest stride.
	__perm: [u8; MAX_LOCAL_DIMS],
}

impl Shape {
	pub fn new(dims: &[usize]) -> Result<Shape, Error> {
		if dims.len() > MAX_LOCAL_DIMS {
			cold_path();
			return Err(Error::TooManyDims);
		}

		let mut arr = [MaybeUninit::uninit(); MAX_LOCAL_DIMS];
		let mut stride: usize = 1;
		for i in (0..MAX_LOCAL_DIMS).rev() {
			if i >= dims.len() {
				arr.__dim[i].write(Dim {
					__len: 1,
					__stride: 1,
				});
			} else {
				arr.__dim[i].write(Dim {
					__len: dims[i],
					__stride: stride,
				});
				let Some(next_stride) = stride.checked_mul(dims[i]) else {
					cold_path();
					return Err(Error::TooManyElems);
				};
				stride = next_stride;
			}
		}

		Ok(Shape {
			__off: 0,
			__ndim: dims.len() as u8,
			__dim: unsafe { std::mem::transmute(arr) },
			__perm: [0, 1, 2, 3],
		})
	}

	pub fn merge_dims(&mut self) -> Shape {
		// assume length of __dim is always MAX_LOCAL_DIMS

		// preprocess dimensions
		for i in 1..MAX_LOCAL_DIMS {
			// If we have a dimension with length 0, the whole tensor has 0 elements.
			if unlikely(self.__dim[i].__len == 0) {
				self.__dim[0].__len = 0;
				self.__dim[0].__stride = 1;
				self.__ndim = 1;
				return;
			}
			// make strides positive
			if unlikely(self.__dim[i].__stride < 0) {
				self.__dim[i].__stride = -self.__dim[i].__stride;
				self.__off -= (self.__dim[i].__len - 1) * self.__dim[i].__stride as usize;
			}
			// TODO - stride == 0
		}

		// sort dimensions from the largest to the smallest stride
		// use insertion sort because the number of dimensions is small

		// TODO - don't sort - use __perm

		for i in 1..MAX_LOCAL_DIMS {
			for j in (0..i).rev() {
				if self.__dim[j].__stride < self.__dim[j + 1].__stride {
					self.__dim.swap(j, j + 1);
				} else {
					break;
				}
			}
		}

		// merge dimensions
		let mut w = 0;
		for r in 1..MAX_LOCAL_DIMS {
			if self.__dim[r].__len * self.__dim[r].__stride == self.__dim[w].__stride {
				let dim_r = self.__dim[r];

				self.__dim[r].__len = 1;
				self.__dim[r].__stride = 1;

				self.__dim[w].__len *= dim_r.__len;
				self.__dim[w].__stride = dim_r.__stride;
			} else {
				w += 1;
				self.__dim[w] = self.__dim[r];
			}
		}
		self.__ndim = (w + 1) as u8;
	}

	pub fn ndim(&self) -> usize {
		self.__ndim as usize
	}

	pub fn dims(&self) -> &[Dim] {
		debug_assert!(self.__ndim as usize <= MAX_LOCAL_DIMS);
		unsafe { std::slice::from_raw_parts(self.__dim.as_ptr(), self.__ndim as usize) }
	}
}
