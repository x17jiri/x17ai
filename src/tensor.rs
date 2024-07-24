// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::panic;
use smallvec::{smallvec, SmallVec};
use std::fmt;
use std::intrinsics::{likely, unlikely};
use std::mem::MaybeUninit;
use std::ops::Index;
use std::rc::{Rc, Weak};
use thin_vec::ThinVec;

pub const INLINE_DIMS: usize = 5;

#[derive(Clone, Copy)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

pub struct ShapeView<'a> {
	tensor: &'a Tensor,
}

impl<'a> ShapeView<'a> {
	pub fn len(&self) -> usize {
		self.tensor.dims.len()
	}

	pub fn to_vec(&self) -> SmallVec<[usize; INLINE_DIMS]> {
		self.tensor.dims.iter().map(|x| x.size).collect()
	}
}

impl Index<isize> for ShapeView<'_> {
	type Output = usize;

	fn index(&self, index: isize) -> &Self::Output {
		let index = self.tensor.__dim_to_usize(index);
		unsafe { &self.tensor.dims.get_unchecked(index).size }
	}
}

// This struct is used mainly to ensure that the total number of elements does not overflow
struct DimsConstructor {
	// dimensions in reverse order
	dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,

	remaining_dims: usize,

	// Total number of elements in the dimensions processed so far
	elems: usize,

	// Total number of elements in the dimensions processed so far,
	// ignoring zero length dimensions.
	nonzero_elems: usize,
}

impl DimsConstructor {
	fn from_shape(shape: &[usize]) -> DimsConstructor {
		let mut t = DimsConstructor::new(shape.len());
		for dim in shape.iter().copied().rev() {
			t.push(dim);
		}
		t.final_check();
		t
	}

	fn new(ndim: usize) -> DimsConstructor {
		let mut dims = SmallVec::with_capacity(ndim);
		unsafe { dims.set_len(ndim) };

		DimsConstructor {
			dims,
			remaining_dims: ndim,
			elems: 1,
			nonzero_elems: 1,
		}
	}

	fn push(&mut self, size: usize) {
		debug_assert!(self.remaining_dims > 0);

		// Check that if we ignore zero length dimensions, the number of elements
		// does not overflow. This is done to make sure our calculations would not overflow
		// even if we had the same dimensions but in different order.
		if likely(size != 0) {
			if let Some(mul) = self.nonzero_elems.checked_mul(size) {
				self.nonzero_elems = mul;
			} else {
				panic!("too many elements");
			};
		}

		unsafe {
			self.remaining_dims -= 1;
			*self.dims.get_unchecked_mut(self.remaining_dims) =
				SizeAndStride { size, stride: self.elems };
		}
		self.elems *= size;
	}

	fn final_check(&self) {
		debug_assert!(self.remaining_dims == 0);

		// Check that the total number of elements does not overflow isize
		let check: Option<isize> = self.elems.try_into().ok();
		if check.is_none() {
			panic!("too many elements");
		}
	}
}

#[derive(Clone)]
pub struct Tensor {
	// dims in reverse order
	dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,
	byte_offset: isize,
	dtype: DType,
	elems: usize,
	buffer: Rc<dyn Buffer>,
}

impl Tensor {
	pub fn new(shape: &[usize], dtype: DType, device: Rc<dyn Device>) -> Tensor {
		let t = DimsConstructor::from_shape(shape);
		Tensor {
			dims: t.dims,
			byte_offset: 0,
			dtype,
			elems: t.elems,
			buffer: device.new_buffer(dtype.array_bytes(t.elems).unwrap()),
		}
	}

	// Allocate a new tensor on the same device
	pub fn new_tensor(&self, shape: &[usize], dtype: DType) -> Tensor {
		let t = DimsConstructor::from_shape(shape);
		self.__new(t.elems, t.dims, dtype)
	}

	fn __new(
		&self,
		elems: usize,
		dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,
		dtype: DType,
	) -> Tensor {
		Tensor {
			dims,
			byte_offset: 0,
			dtype,
			elems,
			buffer: self.buffer.new_buffer(dtype.array_bytes(elems).unwrap()),
		}
	}

	pub fn zeros_(&self) {
		self.buffer.zeros_(self);
	}

	pub fn randn_(&self) {
		self.buffer.randn_(self);
	}

	pub fn shape(&self) -> ShapeView {
		ShapeView { tensor: self }
	}

	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn reshape_last_n(mut self, n: usize, new_shape: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		if n > ndim {
			panic!("cannot reshape more dimensions than the tensor has");
		}

		// will use `Traversal` to merge the last `n` dimensions
		let last_n = unsafe { self.dims.get_unchecked(ndim - n..) };
		let mut merged = Traversal::new(n);
		for dim in last_n.iter().rev() {
			merged.push_dim(dim.size, [dim.stride]);
		}
		let mut merged = merged.rev_dims.iter();

		let merged_dim = merged.next().unwrap();
		let mut prev_stride = merged_dim.in_strides[0];
		let mut target_stride = merged_dim.size * merged_dim.in_strides[0];

		unsafe { self.dims.set_len(ndim - n) };
		self.dims.reserve(new_shape.len());
		let result = unsafe { self.dims.get_unchecked_mut(ndim - n..) };

		for (o, size) in result.iter_mut().zip(new_shape.iter().copied()).rev() {
			if prev_stride * size > target_stride {
				if prev_stride == target_stride {
					let merged_dim = merged.next().unwrap();
					prev_stride = merged_dim.in_strides[0];
					target_stride = merged_dim.size * merged_dim.in_strides[0];

					if prev_stride * size > target_stride {
						panic!("incompatible reshape");
					}
				} else {
					panic!("incompatible reshape");
				}
			}

			prev_stride *= size;

			*o = SizeAndStride { size, stride: prev_stride };
		}

		assert!(merged.next().is_none());
		assert!(prev_stride == target_stride);

		self
	}

	fn __dim_to_internal(&self, dim: isize) -> usize {
		let ndim = self.dims.len();
		let dim = if dim >= 0 { dim as usize } else { ndim - ((-dim) as usize) };
		if likely(dim < ndim as usize) {
			dim
		} else {
			panic!("dimension out of range");
		}
	}

	pub fn transposed(mut self, dim1: isize, dim2: isize) -> Tensor {
		let dim1 = self.__dim_to_internal(dim1);
		let dim2 = self.__dim_to_internal(dim2);

		self.dims.swap(dim1, dim2);
		self
	}

	pub fn t(self) -> Tensor {
		self.transposed(-2, -1)
	}

	// Reinterprets the last dimension as a row matrix
	pub fn as_row_matrix(mut self) -> Tensor {
		self.dims.push(SizeAndStride { size: 1, stride: 1 });
		self
	}

	// Reinterprets the last dimension as a column matrix
	pub fn as_col_matrix(mut self) -> Tensor {
		let last_dim = self.dims.last_mut().unwrap();
		let last_dim_copy = *last_dim;
		*last_dim = SizeAndStride { size: 1, stride: 1 };

		self.dims.push(last_dim_copy);
		self
	}
}

pub fn assert_types_compatible(a: &Tensor, b: &Tensor) {
	if a.dtype != b.dtype {
		panic!("incompatible dtypes");
	}
}

pub fn assert_devices_compatible(a: &Tensor, b: &Tensor) {
	if !are_bufs_on_the_same_device(a.buffer.as_ref(), b.buffer.as_ref()) {
		panic!("incompatible devices");
	}
}

// result = m1 * m2
pub fn matmul(mat1: &Tensor, mat2: &Tensor, scale: f64) -> Tensor {
	assert_types_compatible(mat1, mat2);
	assert_devices_compatible(mat1, mat2);

	assert!(mat1.dims.len() >= 2);
	assert!(mat2.dims.len() >= 2);

	let mat1_batch_dims = &mat1.dims[..mat1.dims.len() - 2];
	let mat2_batch_dims = &mat2.dims[..mat2.dims.len() - 2];

	let mat1_rows = mat1.dims[mat1.dims.len() - 2];
	let mat1_cols = mat1.dims[mat1.dims.len() - 1];
	let mat1_dims = [mat1_rows, mat1_cols];

	let mat2_rows = mat2.dims[mat2.dims.len() - 2];
	let mat2_cols = mat2.dims[mat2.dims.len() - 1];
	let mat2_dims = [mat2_rows, mat2_cols];

	if mat1_cols.size != mat2_rows.size {
		panic!("incompatible dimensions");
	}

	let result_rows = mat1_rows.size;
	let result_cols = mat2_cols.size;
	let result_nonbatch_dims = [result_rows, result_cols];

	let (batch, result_dims, result_elems) =
		prep_batch_traversal([mat1_batch_dims, mat2_batch_dims], &result_nonbatch_dims);
	let batch = &batch.rev_dims;

	let result = mat1.__new(result_elems, result_dims, mat1.dtype);

	unsafe {
		mat1.buffer.matmul(mat1, mat1_dims, mat2, mat2_dims, scale, &result, batch);
	}

	result
}

// We multiply a row vector by a matrix:
//
//                      [[ 1 2 3 ]
// result = [ 1 2 3 ] *  [ 4 5 6 ]
//                       [ 7 8 9 ]]
//
// We can do the following optimization:
// - if there is a batch of the row vectors
// - and the matrix stays the same for all the row vectors in the batch
// - then we can turn the entire batch into a single matrix by matrix multiplication
//
//                   [[ row1 ]    [[ 1 2 3 ]
// result_of_batch =  [ row2 ]  *  [ 4 5 6 ]
//                    [ row2 ]]    [ 7 8 9 ]]
//
pub fn matmul_row_mat(row: &Tensor, mat: &Tensor, scale: f64) -> Tensor {
	assert_types_compatible(row, mat);
	assert_devices_compatible(row, mat);

	let row_batch_dims = &row.dims[..row.dims.len() - 1];
	let mat_batch_dims = &mat.dims[..mat.dims.len() - 2];

	let row_cols = row.dims[row.dims.len() - 1].size;
	let mat_rows = mat.dims[mat.dims.len() - 2].size;
	let mat_cols = mat.dims[mat.dims.len() - 1].size;

	if row_cols != mat_rows {
		panic!("incompatible dimensions");
	}
}

pub fn matmul_mat_col(mat: &Tensor, col: &Tensor, scale: f64) -> Tensor {
	assert_types_compatible(mat, col);
	assert_devices_compatible(mat, col);

	// TODO
}

pub fn matmul_row_col(row: &Tensor, col: &Tensor, scale: f64) -> Tensor {
	assert_types_compatible(row, col);
	assert_devices_compatible(row, col);

	// TODO
}

pub fn matmul_col_row(col: &Tensor, row: &Tensor, scale: f64) -> Tensor {
	assert_types_compatible(col, row);
	assert_devices_compatible(col, row);

	// TODO
}

// acc += (m1 * m2) * scale
pub fn scaled_matmul_acc(m1: &Tensor, m2: &Tensor, scale: f64, acc: &Tensor) {
	// TODO
}

// t = v reinterpretted as a column matrix
// result = (m * t) * scale
pub fn scaled_mat_vec_mul(m: &Tensor, v: &Tensor, scale: f64) -> Tensor {
	if m.dtype() != v.dtype() {
		panic!("incompatible dtypes");
	}
	if !are_bufs_on_the_same_device(m.buffer.as_ref(), v.buffer.as_ref()) {
		panic!("incompatible devices");
	}

	let m_batch_dims = &m.dims[..m.dims.len() - 2];
	let v_batch_dims = &v.dims[..v.dims.len() - 1];
	let nonbatch_out_dim = m.dims[m.dims.len() - 1].size;
	let (traversal, out_dims, out_elems) =
		prep_batch_traversal([&m_batch_dims, &v_batch_dims], &[nonbatch_out_dim]);

	let out = m.__new(out_elems, out_dims, m.dtype);

	unsafe {
		m.buffer.mat_vec_mul(m, v, scale, &out, traversal);
	}

	out
}

pub fn rms_norm(a: &Tensor, out: &Tensor) {
	// TODO
}

fn fmt_0d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	let byte_offset = tensor.byte_offset + off * tensor.dtype.bytes();
	tensor.buffer.format(byte_offset, tensor.dtype, f, 1)
}

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	let byte_offset = tensor.byte_offset + off * tensor.dtype.bytes();
	write!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 1];
	tensor.buffer.format(byte_offset, tensor.dtype, f, len)?;
	write!(f, "]")
}

fn fmt_2d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	writeln!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 2];
	let stride = tensor.shape.dims()[ndim - 1];
	for i in 0..len {
		write!(f, "\t")?;
		fmt_1d(tensor, f, off + i * stride)?;
		writeln!(f, ",")?;
	}
	write!(f, "]")
}

impl fmt::Display for Tensor {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Tensor(")?;
		let ndim = self.shape.ndim();
		match ndim {
			0 => fmt_0d(self, f, 0)?,
			1 => fmt_1d(self, f, 0)?,
			2 => fmt_2d(self, f, 0)?,
			_ => {
				todo!("Tensor with {} dimensions", ndim);
			},
		};
		write!(f, ")")
	}
}

//--------------------------------------------------------------------------------------------------

pub fn prep_batch_traversal<const N: usize>(
	inputs: [&[SizeAndStride]; N],
	nonbatch_dims: &[usize],
) -> (Traversal<N>, SmallVec<[SizeAndStride; INLINE_DIMS]>, usize) {
	assert!(N > 0);
	let ndim = inputs.iter().map(|x| x.len()).max().unwrap();

	let mut traversal = Traversal::new(ndim);

	let mut out_shape = DimsConstructor::new(ndim + nonbatch_dims.len());
	for dim_size in nonbatch_dims.iter().rev().copied() {
		out_shape.push(dim_size);
	}

	for d in (0..ndim).rev() {
		// Get sizes and strides for the current dimension from all inputs
		let mut in_sizes = [0; N];
		let mut in_strides = [0; N];
		for i in 0..N {
			if d < inputs[i].len() {
				in_sizes[i] = inputs[i][d].size;
				in_strides[i] = inputs[i][d].stride;
			} else {
				in_sizes[i] = 1;
				in_strides[i] = 0;
			}
		}

		// The dim_size should be the same for all inputs except for broadcasted inputs
		// So max() gets the dim_size of the non-broadcasted inputs.
		let dim_size = *in_sizes.iter().max().unwrap();

		out_shape.push(dim_size);

		// Set the strides of the broadcasted inputs to 0
		for i in 0..N {
			if in_sizes[i] != dim_size {
				if in_sizes[i] == 1 {
					in_strides[i] = 0;
				} else {
					// cannot broadcast
					cold_path();
					panic!("incompatible dimensions");
				}
			}
		}

		traversal.push_dim(dim_size, in_strides);
	}

	out_shape.final_check();
	(traversal, out_shape.dims, out_shape.elems)
}

#[derive(Clone)]
pub struct TraversalDim<const N: usize> {
	pub size: usize,
	pub out_stride: usize,
	pub in_strides: [usize; N],
}

#[derive(Clone)]
pub struct Traversal<const N: usize> {
	// dims in traversal are in reverse order
	// in other words, from smallest to largest stride
	pub rev_dims: SmallVec<[TraversalDim<N>; INLINE_DIMS]>,
	pub elems: usize,
}

impl<const N: usize> Traversal<N> {
	pub fn new(ndim: usize) -> Traversal<N> {
		if ndim == 0 {
			Traversal {
				rev_dims: smallvec![TraversalDim {
					size: 1,
					out_stride: 1,
					in_strides: [1; N],
				}],
				elems: 1,
			}
		} else {
			Traversal {
				rev_dims: SmallVec::with_capacity(ndim),
				elems: 1,
			}
		}
	}

	// dimensions should be pushed in the order of increasing strides
	pub fn push_dim(&mut self, size: usize, in_strides: [usize; N]) {
		if size == 1 {
			return;
		}

		let out_stride = self.elems;
		self.elems *= size;

		// If there already are dimensions, try to merge the new dimension with the last one
		if !self.rev_dims.is_empty() {
			let prev = self.rev_dims.last_mut().unwrap();

			let mut can_merge = true;
			#[allow(unused_parens)]
			for i in 0..N {
				can_merge &= (in_strides[i] == prev.in_strides[i] * prev.size);
			}

			if can_merge {
				prev.size *= size;
				return;
			}
		}

		// Can't merge, push the new dimension
		self.rev_dims.push(TraversalDim { size, out_stride, in_strides });
	}
}
