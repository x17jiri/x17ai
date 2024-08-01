// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::panic;
use smallvec::{smallvec, SmallVec};
use std::fmt;
use std::intrinsics::{likely, unlikely};
use std::iter::ExactSizeIterator;
use std::mem::MaybeUninit;
use std::ops::Index;
use std::rc::{Rc, Weak};
use thin_vec::ThinVec;

pub const INLINE_DIMS: usize = 5;

#[derive(Clone, Copy, PartialEq)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

pub struct ShapeView<'a> {
	dims: &'a [SizeAndStride],
}

impl<'a> ShapeView<'a> {
	pub fn len(&self) -> usize {
		self.dims.len()
	}
}

impl Index<isize> for ShapeView<'_> {
	type Output = usize;

	fn index(&self, index: isize) -> &Self::Output {
		let i = if index < 0 { self.dims.len() as isize + index } else { index };
		&self.dims[i as usize].size
	}
}

impl Index<usize> for ShapeView<'_> {
	type Output = usize;

	fn index(&self, index: usize) -> &Self::Output {
		&self.dims[index].size
	}
}

/*
impl std::cmp::PartialEq<ShapeView<'_>> for ShapeView<'_> {
	fn eq(&self, other: &ShapeView) -> bool {
		if self.tensor.dims.len() != other.tensor.dims.len() {
			return false;
		}
		for (a, b) in self.tensor.dims.iter().zip(other.tensor.dims.iter()) {
			if a.size != b.size {
				return false;
			}
		}
		true
	}
}
*/

impl<'a> IntoIterator for ShapeView<'a> {
	type Item = &'a usize;
	type IntoIter =
		std::iter::Map<std::slice::Iter<'a, SizeAndStride>, fn(&SizeAndStride) -> &usize>;

	fn into_iter(self) -> Self::IntoIter {
		self.tensor.dims.iter().map(|x| &x.size)
	}
}

// This struct is used mainly to ensure that the total number of elements does not overflow
struct DimsCtor {
	// dimensions in reverse order
	dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,

	remaining_dims: usize,

	// Total number of elements in the dimensions processed so far
	elems: usize,

	// Total number of elements in the dimensions processed so far,
	// ignoring zero length dimensions.
	nonzero_elems: usize,
}

impl DimsCtor {
	fn from_shape<'a, Shape>(shape: Shape) -> DimsCtor
	where
		Shape: IntoIterator<Item = &'a usize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a usize> + ExactSizeIterator,
	{
		let shape = shape.into_iter().copied();
		let mut t = DimsCtor::new(shape.len());
		for dim in shape.rev() {
			t.push(dim);
		}
		t.final_check();
		t
	}

	fn new(ndim: usize) -> DimsCtor {
		let mut dims = SmallVec::with_capacity(ndim);
		unsafe { dims.set_len(ndim) };

		DimsCtor {
			dims,
			remaining_dims: ndim,
			elems: 1,
			nonzero_elems: 1,
		}
	}

	// Push a new dimension with the given size
	// Returns the stride of the new dimension
	fn push(&mut self, size: usize) -> usize {
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

		let stride = self.elems;
		self.elems *= size;

		self.remaining_dims -= 1;
		unsafe {
			*self.dims.get_unchecked_mut(self.remaining_dims) = SizeAndStride { size, stride };
		}

		stride
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
	pub(crate) offset: usize,
	dtype: DType,
	elems: usize,
	pub(crate) buffer: Rc<dyn Buffer>,
}

impl Tensor {
	fn __new_empty_on(dims_ctor: DimsCtor, dtype: DType, device: Rc<dyn Device>) -> Tensor {
		Tensor {
			dims: dims_ctor.dims,
			offset: 0,
			dtype,
			elems: dims_ctor.elems,
			buffer: device.new_buffer(dtype.array_bytes(dims_ctor.elems).unwrap()),
		}
	}

	pub fn new_empty_on<'a, Shape>(shape: Shape, dtype: DType, device: Rc<dyn Device>) -> Tensor
	where
		Shape: IntoIterator<Item = &'a usize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a usize> + ExactSizeIterator,
	{
		Self::__new_empty_on(DimsCtor::from_shape(shape), dtype, device)
	}

	// Allocate a new tensor on the same device
	pub fn new_empty<'a, Shape>(&'a self, shape: Shape, dtype: DType) -> Tensor
	where
		Shape: IntoIterator<Item = &'a usize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a usize> + ExactSizeIterator,
	{
		Self::new_empty_on(shape, dtype, self.device())
	}

	// Allocate a new tensor on the same device with the same shape and dtype
	pub fn new_empty_like(&self) -> Tensor {
		Self::new_empty_on(self.shape(), self.dtype, self.device())
	}

	pub fn device(&self) -> Rc<dyn Device> {
		buf_to_base(self.buffer.as_ref()).device.clone()
	}

	pub fn zeros_(&self) {
		self.buffer.zeros_(self);
	}

	pub fn randn_(&self) {
		self.buffer.randn_(self);
	}

	/// Accumulate:
	/// ```
	///    self = alpha * self + beta * b
	/// ```
	/// This function doesn't broadcast
	/// self.shape must be equal to b.shape
	pub fn acc_(&self, alpha: f64, beta: f64, b: &Tensor) {
		assert_compatible_types(self, other);
		assert_compatible_devices(self, other);
		// TODO
	}

	/// Accumulate the result of element-wise multiplication:
	/// ```
	///     self = alpha * self + beta * (b * c)
	/// ```
	/// This function may broadcast b and c to match the shape of self.
	pub fn acc_mul_(&self, alpha: f64, beta: f64, b: &Tensor, c: &Tensor) {
		assert_compatible_types(self, a);
		assert_compatible_types(self, b);
		assert_compatible_devices(self, a);
		assert_compatible_devices(self, b);
		// TODO
	}

	/// Accumulate the result of sum:
	/// ```
	///    self = alpha * self + beta * b.sum(keepdim)
	/// ```
	/// This function doesn't broadcast.
	/// The shape of b after the sum must be equal to the shape of self.
	pub fn acc_sum_(&self, alpha: f64, beta: f64, b: &Tensor, keepdim: bool) {
		assert_compatible_types(self, other);
		assert_compatible_devices(self, other);
		// TODO
	}

	/// Accumulate the result of mean:
	/// ```
	///   self = alpha * self + beta * b.mean(keepdim)
	/// ```
	/// This function doesn't broadcast.
	/// The shape of b after the sum must be equal to the shape of self.
	pub fn acc_mean_(&self, alpha: f64, beta: f64, b: &Tensor, keepdim: bool) {
		assert_compatible_types(self, other);
		assert_compatible_devices(self, other);
		// TODO
	}

	/// Calculate `x^2`:
	/// ```
	///    result = self * self
	/// ```
	pub fn square(&self) -> Tensor {
		let mut out = self.new_empty_like();
		self.buffer.square(&mut out, self);
		out
	}

	/// reciprocal of the square root:
	/// ```
	///     result = 1.0 / (self.sqrt() + eps)
	/// ```
	pub fn rsqrt(&self, eps: f64) -> Tensor {
		let mut out = self.new_empty_like();
		self.buffer.rsqrt(&mut out, self, eps);
		out
	}

	pub fn rsqrt_into(&self, eps: f64, into: &Tensor) {
		assert_compatible_types(self, into);
		assert_compatible_devices(self, into);
		//self.buffer.rsqrt(into, self, eps);
	}

	/// Calculate the mean of the last dimension.
	///
	/// If keepdim is true, the last dimension is kept in the result with size 1.
	///
	/// If keepdim is false, the last dimension is removed from the result.
	pub fn mean(&self, keepdim: bool) -> Tensor {
		let mut out = self.new_empty_like();
		//self.buffer.mean(&mut out, self, keepdim);
		//out
	}

	pub fn mean_into(&self, keepdim: bool, into: &Tensor) {
		assert_compatible_types(self, into);
		assert_compatible_devices(self, into);
		//self.buffer.mean(into, self, keepdim);
	}

	pub fn shape(&self) -> ShapeView {
		ShapeView { tensor: self }
	}

	/// Returns the size of dimension `dim`.
	pub fn size(&self, dim: isize) -> usize {
		self.dims[self.__dim_to_internal(dim)].size
	}

	/// Returns the stride of dimension `dim`.
	pub fn stride(&self, dim: isize) -> usize {
		self.dims[self.__dim_to_internal(dim)].stride
	}

	/// Returns the number of dimensions in the tensor.
	///
	/// This is also known as the rank of the tensor.
	pub fn ndim(&self) -> usize {
		self.dims.len()
	}

	/// Returns the total number of elements in the tensor.
	pub fn elems(&self) -> usize {
		self.elems
	}

	/// Returns the data type of the tensor elements.
	pub fn dtype(&self) -> DType {
		self.dtype
	}

	pub fn reshape_last_n(mut self, n: usize, new_shape: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		if n > ndim {
			panic!("cannot reshape more dimensions than the tensor has");
		}

		// use `Batch` to merge the last `n` dimensions
		let last_n = unsafe { self.dims.get_unchecked(ndim - n..) };
		let mut merged = Batch::new_empty(n);
		let mut elems = 1;
		for dim in last_n.iter().rev() {
			merged.push_dim(dim.size, elems, [dim.stride]);
			elems *= dim.size;
		}
		merged.ensure_nonzero_ndim();
		let mut merged = merged.rev_dims.iter();

		let merged_dim = merged.next().unwrap();
		let mut prev_stride = merged_dim.in_strides[0];
		let mut target_stride = merged_dim.size * merged_dim.in_strides[0];

		unsafe { self.dims.set_len(ndim - n) };
		self.dims.reserve(new_shape.len());
		unsafe { self.dims.set_len(ndim - n + new_shape.len()) };
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

	pub fn reshape(mut self, new_shape: &[usize]) -> Tensor {
		self.reshape_last_n(self.ndim(), new_shape)
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

	pub fn permuted(mut self, perm: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		assert!(perm.len() == ndim, "number of dimensions does not match");

		let mut new_dims = SmallVec::with_capacity(ndim);
		unsafe { new_dims.set_len(ndim) };

		let mut sum = 0;
		for (new_dim, p) in new_dims.iter_mut().zip(perm.iter()) {
			*new_dim = self.dims[*p];
			sum += *p;
		}

		let expected_sum = ndim * (ndim - 1) / 2;
		assert!(sum == expected_sum, "invalid permutation");

		self.dims = new_dims;
		self
	}

	#[allow(non_snake_case)]
	pub fn T<'a>(&'a self) -> Matrix<'a> {
		self.as_matrix().T()
	}

	pub fn as_row<'a>(&'a self) -> Matrix<'a> {
		let ndim = self.ndim();
		assert!(ndim >= 1);
		Matrix {
			tensor: self,
			batch_dims: &self.dims[..ndim - 1],
			rows: SizeAndStride { size: 1, stride: 1 },
			cols: self.dims[ndim - 1],
		}
	}

	pub fn as_col<'a>(&'a self) -> Matrix<'a> {
		let ndim = self.ndim();
		assert!(ndim >= 1);
		Matrix {
			tensor: self,
			batch_dims: &self.dims[..ndim - 1],
			rows: self.dims[ndim - 1],
			cols: SizeAndStride { size: 1, stride: 1 },
		}
	}
}

pub trait TensorRef {
	fn tensor_ref(&self) -> &Tensor;
}

impl TensorRef for &Tensor {
	fn tensor_ref(&self) -> &Tensor {
		*self
	}
}

impl TensorRef for Tensor {
	fn tensor_ref(&self) -> &Tensor {
		self
	}
}

impl<B: TensorRef> std::ops::Mul<B> for Tensor {
	type Output = Tensor;

	fn mul(self, other: B) -> Tensor {
		// TODO
	}
}

impl<B: TensorRef> std::ops::Mul<B> for &Tensor {
	type Output = Tensor;

	fn mul(self, other: B) -> Tensor {
		// TODO
	}
}

impl std::ops::SubAssign<&Tensor> for Tensor {
	fn sub_assign(&mut self, other: &Tensor) {
		// TODO
	}
}

impl std::ops::SubAssign<Tensor> for Tensor {
	fn sub_assign(&mut self, other: Tensor) {
		// TODO
	}
}

pub fn assert_compatible_types(a: &Tensor, b: &Tensor) {
	assert!(a.dtype == b.dtype, "incompatible dtypes");
}

pub fn assert_compatible_devices(a: &Tensor, b: &Tensor) {
	assert!(
		are_bufs_on_the_same_device(a.buffer.as_ref(), b.buffer.as_ref()),
		"incompatible devices"
	);
}

// c = a * b * alpha
fn __gemm<'a>(a: Matrix<'a>, b: Matrix<'a>, alpha: f64) -> Tensor {
	assert_compatible_types(a.tensor, b.tensor);
	assert_compatible_devices(a.tensor, b.tensor);

	assert!(a.cols.size == b.rows.size, "incompatible dimensions");
	let c_rows = SizeAndStride { size: a.rows.size, stride: b.cols.size };
	let c_cols = SizeAndStride { size: b.cols.size, stride: 1 };

	let dtype = a.tensor.dtype;
	let batch_ndim = a.batch_dims.len().max(b.batch_dims.len());
	let mut c_dims_ctor = DimsCtor::new(batch_ndim + 2);
	// Note: dims need to be pushed to ctor in reverse order
	c_dims_ctor.push(c_cols.size);
	c_dims_ctor.push(c_rows.size);

	let batch = Batch::new(batch_ndim, [a.batch_dims, b.batch_dims], &mut c_dims_ctor);

	c_dims_ctor.final_check();
	let c = a.tensor.__new(c_dims_ctor.elems, c_dims_ctor.dims, dtype);

	let a_rows_contiguous = a.cols.stride == 1;
	let a_cols_contiguous = a.rows.stride == 1;
	let transa = !a_rows_contiguous;
	let lda = if a_rows_contiguous { a.rows.stride } else { a.cols.stride };
	assert!(
		a_rows_contiguous || a_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	let b_rows_contiguous = b.cols.stride == 1;
	let b_cols_contiguous = b.rows.stride == 1;
	let transb = !b_rows_contiguous;
	let ldb = if b_rows_contiguous { b.rows.stride } else { b.cols.stride };
	assert!(
		b_rows_contiguous || b_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	let m = c_rows.size;
	let n = c_cols.size;
	let k = a.cols.size;

	let c_rows_contiguous = c_cols.stride == 1;
	let c_cols_contiguous = c_rows.stride == 1;
	let transc = !c_rows_contiguous;
	let ldc = if c_rows_contiguous { c_rows.stride } else { c_cols.stride };
	assert!(
		c_rows_contiguous || c_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	let mut transa = transa;
	let mut transb = transb;
	let mut m = m;
	let mut n = n;
	let mut a = a;
	let mut b = b;
	let mut lda = lda;
	let mut ldb = ldb;
	if transc {
		// C^T = B^T * A^T
		(transa, transb) = (!transb, !transa);
		(m, n) = (n, m);
		(a, b) = (b, a);
		(lda, ldb) = (ldb, lda);
	}
	let transa = transa;
	let transb = transb;
	let m = m;
	let n = n;
	let a = a;
	let b = b;
	let lda = lda;
	let ldb = ldb;

	let a_buf = buf_to_base(a.tensor.buffer.as_ref());
	let b_buf = buf_to_base(b.tensor.buffer.as_ref());
	#[rustfmt::skip]
	batch.run(
		c.offset, [a.tensor.offset, b.tensor.offset],
		&|c_offset, [a_offset, b_offset], batch| {
			unsafe {
				c.buffer.gemm(
					dtype, c_offset, ldc, batch.out_stride,
					m, n, k,
					a_buf, a_offset, lda, transa, batch.in_strides[0],
					b_buf, b_offset, ldb, transb, batch.in_strides[1],
					alpha, 0.0,
					batch.size,
				);
			}
		},
	);

	c
}

#[derive(Clone, Copy)]
pub struct Matrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],
	pub rows: SizeAndStride,
	pub cols: SizeAndStride,
}

impl<'a> Matrix<'a> {
	#[allow(non_snake_case)]
	pub fn T(&self) -> Matrix<'a> {
		Matrix {
			tensor: self.tensor,
			batch_dims: self.batch_dims,
			rows: self.cols,
			cols: self.rows,
		}
	}
}

pub trait MatrixLike<'a> {
	fn as_matrix(self) -> Matrix<'a>;
}

impl<'a> MatrixLike<'a> for &'a Tensor {
	fn as_matrix(self) -> Matrix<'a> {
		let ndim = self.ndim();
		assert!(ndim >= 2);
		let batch_dims = &self.dims[..ndim - 2];
		let rows = self.dims[ndim - 2];
		let cols = self.dims[ndim - 1];
		Matrix { tensor: self, batch_dims, rows, cols }
	}
}

impl<'a> MatrixLike<'a> for Matrix<'a> {
	fn as_matrix(self) -> Matrix<'a> {
		self
	}
}

// Multiply two matrices
// result = a * b * alpha
pub fn mm<'a, A: MatrixLike<'a>, B: MatrixLike<'a>>(a: A, b: B) -> MatMul<'a> {
	let a = a.as_matrix();
	let b = b.as_matrix();
	MatMul::new(a, b)
}

pub struct MatMul<'a> {
	pub a: Matrix<'a>,
	pub b: Matrix<'a>,
}

impl<'a> MatMul<'a> {
	pub fn new(a: Matrix<'a>, b: Matrix<'a>) -> MatMul<'a> {
		assert!(a.cols.size == b.rows.size);
		MatMul { a, b }
	}

	// Calculate the default scale for the MatMul based on the `k` dimension.
	// A = [m, k]
	// B = [k, n]
	// Each cell of the result is the sum of k products.
	// The default scale is `1 / sqrt(k)` in order to preserve the variance of the input.
	pub fn scale_for_k(k: usize) -> f64 {
		1.0 / (k as f64).sqrt()
	}

	pub fn default_scale(&self) -> f64 {
		Self::scale_for_k(self.a.cols.size)
	}

	// Calculate the MatMul and multiply the result by `scale`.
	pub fn eval(&self, scale: f64) -> Tensor {
		__gemm(self.a, self.b, scale)
	}

	// Calculate the MatMul and multiply the result by the default scale.
	pub fn eval_default_scale(&self) -> Tensor {
		self.eval(self.default_scale())
	}

	// Calculate the MatMul and multiply the result by `scale`.
	//
	// In debug builds, this function checks that `scale` is equal to the default scale.
	// In release builds, no check is performed.
	//
	// This is useful when we want to use the default scale, but we don't want to
	// recalculate it every time.
	// We can calculate it once using `scale_for_k()` and then use this function to make sure
	// we didn't make a mistake and are in fact using the correct value.
	pub fn eval_check_scale(&self, scale: f64) -> Tensor {
		debug_assert!(scale == self.default_scale());
		self.eval(scale)
	}

	pub fn backward<M: MatrixLike<'a>>(&self, dy: M) -> MatMulBackward<'a> {
		let dy = dy.as_matrix();
		assert!(dy.rows.size == self.a.rows.size);
		assert!(dy.cols.size == self.b.cols.size);
		MatMulBackward { a: Some(self.a), b: Some(self.b), dy }
	}
}

pub struct MatMulBackward<'a> {
	pub a: Option<Matrix<'a>>,
	pub b: Option<Matrix<'a>>,
	pub dy: Matrix<'a>,
}

// Use this function only if either `a` or `b` can be `None`.
// If both are `Some`, it is cleaner to use `mm(a, b).backward(dy)`.
//
// `da` can only be calculated if we have `b`,
// and `db` if we have `a`.
pub fn mm_backward<'a>(
	a: Option<Matrix<'a>>,
	b: Option<Matrix<'a>>,
	dy: Matrix<'a>,
) -> MatMulBackward<'a> {
	MatMulBackward { a, b, dy }
}

impl<'a> MatMulBackward<'a> {
	pub fn da(&self) -> MatMul {
		MatMul::new(self.dy, self.b.unwrap().T())
	}

	pub fn db(&self) -> MatMul {
		MatMul::new(self.a.unwrap().T(), self.dy)
	}
}

pub fn rms_norm(a: &Tensor) -> Tensor {
	assert!(a.ndim() > 0);
	let ndim = a.ndim();
	let batch_ndim = ndim - 1;
	let dim_size = a.dims[ndim - 1].size;

	let mut out_dims_ctor = DimsCtor::new(ndim);
	out_dims_ctor.push(dim_size);

	let batch = Batch::new(batch_ndim, [&a.dims[..batch_ndim]], &mut out_dims_ctor);

	out_dims_ctor.final_check();
	let out = a.__new(out_dims_ctor.elems, out_dims_ctor.dims, a.dtype);

	let out_buf = out.buffer.as_ref();
	let a_buf = buf_to_base(a.buffer.as_ref());
	#[rustfmt::skip]
	batch.run(
		out.offset, [a.offset],
		&|out_offset, [a_offset], batch| {
			unsafe {
				out_buf.rms_norm(
					a.dtype, out_offset, batch.out_stride,
					a_buf, a_offset, batch.in_strides[0],
					dim_size, 1e-6,
					batch.size
				);
			}
		}
	);

	out
}

pub fn softmax(a: &Tensor) -> Tensor {
	assert!(a.ndim() > 0);
	let ndim = a.ndim();
	let batch_ndim = ndim - 1;
	let dim_size = a.dims[ndim - 1].size;

	let mut out_dims_ctor = DimsCtor::new(ndim);
	out_dims_ctor.push(dim_size);

	let batch = Batch::new(batch_ndim, [&a.dims[..batch_ndim]], &mut out_dims_ctor);

	out_dims_ctor.final_check();
	let out = a.__new(out_dims_ctor.elems, out_dims_ctor.dims, a.dtype);

	let out_buf = out.buffer.as_ref();
	let a_buf = buf_to_base(a.buffer.as_ref());
	#[rustfmt::skip]
	batch.run(
		out.offset, [a.offset],
		&|out_offset, [a_offset], batch| {
			unsafe {
				out_buf.softmax(
					a.dtype, out_offset, batch.out_stride,
					a_buf, a_offset, batch.in_strides[0],
					dim_size,
					batch.size
				);
			}
		}
	);

	out
}

fn fmt_0d(tensor: &Tensor, f: &mut fmt::Formatter, offset: usize) -> fmt::Result {
	let offset = tensor.offset + offset;
	unsafe { tensor.buffer.format(f, tensor.dtype, offset, 1, 1) }
}

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, offset: usize) -> fmt::Result {
	let offset = tensor.offset + offset;
	let dim = tensor.dims[tensor.ndim() - 1];
	write!(f, "[")?;
	unsafe { tensor.buffer.format(f, tensor.dtype, offset, dim.size, dim.stride)? };
	write!(f, "]")
}

fn fmt_2d(tensor: &Tensor, f: &mut fmt::Formatter, offset: usize) -> fmt::Result {
	writeln!(f, "[")?;
	let dim = tensor.dims[tensor.ndim() - 2];
	for i in 0..dim.size {
		write!(f, "\t")?;
		fmt_1d(tensor, f, offset + i * dim.stride)?;
		writeln!(f, ",")?;
	}
	write!(f, "]")
}

impl fmt::Display for Tensor {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Tensor(")?;
		match self.ndim() {
			0 => fmt_0d(self, f, 0)?,
			1 => fmt_1d(self, f, 0)?,
			2 => fmt_2d(self, f, 0)?,
			_ => {
				todo!("Tensor with {} dimensions", self.ndim());
			},
		};
		write!(f, ")")
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct BatchDim<const N: usize> {
	pub size: usize,
	pub out_stride: usize,
	pub in_strides: [usize; N],
}

#[derive(Clone)]
pub struct Batch<const N: usize> {
	// dims in batch are in reverse order
	// in other words, from smallest to largest stride
	pub rev_dims: SmallVec<[BatchDim<N>; INLINE_DIMS]>,
}

impl<const N: usize> Batch<N> {
	pub fn new(ndim: usize, inputs: [&[SizeAndStride]; N], output: &mut DimsCtor) -> Batch<N> {
		assert!(N > 0);

		let mut batch = Batch::new_empty(ndim);

		for d in (0..ndim).rev() {
			// Get sizes and strides for the current dimension from all inputs
			let mut in_sizes = [0; N];
			let mut in_strides = [0; N];
			for i in 0..N {
				// Does the input have enough dimensions,
				// or do we need to extend it with broadcasted dimensions?
				if d < inputs[i].len() {
					in_sizes[i] = inputs[i][d].size;
					in_strides[i] = inputs[i][d].stride;
				} else {
					in_sizes[i] = 1;
					in_strides[i] = 0;
				}
			}

			// TODO - what happens when one of the dimensions has size 0?

			// The dim_size should be the same for all inputs except for broadcasted inputs
			// So max() gets the dim_size of the non-broadcasted inputs.
			let dim_size = in_sizes.iter().copied().max().unwrap();

			let out_stride = output.push(dim_size);

			// Find inputs that need broadcasting and set their strides to 0
			for i in 0..N {
				if in_sizes[i] != dim_size {
					assert!(in_sizes[i] == 1, "cannot broadcast: incompatible dimensions");
					in_strides[i] = 0;
				}
			}

			batch.push_dim(dim_size, out_stride, in_strides);
		}

		batch
	}

	pub fn new_empty(ndim: usize) -> Batch<N> {
		Batch { rev_dims: SmallVec::with_capacity(ndim) }
	}

	// dimensions should be pushed in the order of increasing strides
	pub fn push_dim(&mut self, size: usize, out_stride: usize, in_strides: [usize; N]) {
		if size == 1 {
			return;
		}

		// If there already are dimensions, try to merge the new dimension with the last one
		if !self.rev_dims.is_empty() {
			let prev = self.rev_dims.last_mut().unwrap();

			let mut can_merge;
			#[allow(unused_parens)]
			{
				can_merge = (out_stride == prev.out_stride * prev.size);
				for i in 0..N {
					can_merge &= (in_strides[i] == prev.in_strides[i] * prev.size);
				}
			}

			if can_merge {
				prev.size *= size;
				return;
			}
		}

		// Can't merge, push the new dimension
		self.rev_dims.push(BatchDim { size, out_stride, in_strides });
	}

	pub fn ensure_nonzero_ndim(&mut self) {
		if unlikely(self.rev_dims.is_empty()) {
			self.rev_dims.push(BatchDim {
				size: 1,
				out_stride: 1,
				in_strides: [1; N],
			});
		}
	}

	fn __run<F: Fn(usize, [usize; N], BatchDim<N>)>(
		&self,
		out_offset: usize,
		in_offsets: [usize; N],
		f: &F,
		batch: &[BatchDim<N>],
	) {
		debug_assert!(!batch.is_empty());
		let batch_dim = unsafe { batch.get_unchecked(batch.len() - 1) };
		let batch = &batch[..batch.len() - 1];
		if batch.is_empty() {
			f(out_offset, in_offsets, *batch_dim);
		} else {
			for i in 0..batch_dim.size {
				let out_offset = out_offset + i * batch_dim.out_stride;
				let mut in_offsets = in_offsets;
				for j in 0..N {
					in_offsets[j] += i * batch_dim.in_strides[j];
				}
				self.__run(out_offset, in_offsets, f, batch);
			}
		}
	}

	// f = fn(out_offset: usize, in_offsets: [usize; N], batch_dim: BatchDim<N>)
	pub fn run<F: Fn(usize, [usize; N], BatchDim<N>)>(
		&self,
		out_offset: usize,
		in_offsets: [usize; N],
		f: &F,
	) {
		if self.rev_dims.is_empty() {
			f(
				out_offset,
				in_offsets,
				BatchDim {
					size: 1,
					out_stride: 0,
					in_strides: [0; N],
				},
			);
		} else {
			self.__run(out_offset, in_offsets, f, &self.rev_dims);
		}
	}
}
