// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::panic;
use smallvec::{SmallVec, smallvec};
use std::alloc::Layout;
use std::fmt;
use std::intrinsics::{cold_path, likely, unlikely};
use std::iter::ExactSizeIterator;
use std::mem::MaybeUninit;
use std::ops::Index;
use std::rc::{Rc, Weak};
use thin_vec::ThinVec;

pub const INLINE_DIMS: usize = 5;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Default)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

impl SizeAndStride {
	pub fn is_contiguous(&self) -> bool {
		self.stride == 1 || self.size <= 1
	}
}

//--------------------------------------------------------------------------------------------------

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
		self.dims.iter().map(|x| &x.size)
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tensor {
	// dims in reverse order
	pub(crate) dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,
	pub(crate) offset: usize,
	pub(crate) dtype: DType,
	pub(crate) elems: usize,
	pub(crate) buffer: Rc<dyn Buffer>,
}

impl Tensor {
	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on<'a, Shape>(shape: Shape, dtype: DType, device: Rc<dyn Device>) -> Tensor
	where
		Shape: IntoIterator<Item = &'a usize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a usize> + ExactSizeIterator,
	{
		let shape = shape.into_iter().copied();
		let builder = NewOutputHandler::new(dtype, device);
		let mut builder = builder.init(shape.len());
		for dim in shape.rev() {
			builder.prepend_dim(dim);
		}
		builder.value()
	}

	/// Allocate a new tensor on the same device as `self`.
	pub fn new_empty<'a, Shape>(&'a self, shape: Shape, dtype: DType) -> Tensor
	where
		Shape: IntoIterator<Item = &'a usize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a usize> + ExactSizeIterator,
	{
		Self::new_empty_on(shape, dtype, self.device())
	}

	/// Allocate a new tensor on the same device with the same shape and dtype
	/// as `self`.
	pub fn new_empty_like(&self) -> Tensor {
		Self::new_empty_on(self.shape(), self.dtype, self.device())
	}

	/// Returns the device on which the tensor is allocated.
	pub fn device(&self) -> Rc<dyn Device> {
		BufferBase::from_dyn_buf(self.buffer.as_ref()).device.clone()
	}

	pub fn shape(&self) -> ShapeView {
		ShapeView { dims: &self.dims }
	}

	/// `dim` should be in the range `0..<ndim`.
	pub fn dim_from_start(&self, dim: usize) -> SizeAndStride {
		let ndim = self.dims.len();
		if dim < ndim { self.dims[dim] } else { SizeAndStride { size: 1, stride: 1 } }
	}

	/// `dim` should be in the range `1..=ndim`.
	pub fn dim_from_end(&self, dim: usize) -> SizeAndStride {
		let ndim = self.dims.len();
		let dim = ndim.wrapping_sub(dim);
		if dim < ndim { self.dims[dim] } else { SizeAndStride { size: 1, stride: 0 } }
	}

	pub fn dim(&self, dim: isize) -> SizeAndStride {
		if dim >= 0 {
			self.dim_from_start(dim.unsigned_abs())
		} else {
			self.dim_from_end(dim.unsigned_abs())
		}
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

	// Reshapes the last `n_dims_to_reshape` dimensions of the tensor
	pub fn reshape(mut self, n_dims_to_reshape: usize, new_shape: &[usize]) -> Tensor {
		let dims = &mut self.dims;
		let ndim = dims.len();
		let n_dims_to_keep = ndim - n_dims_to_reshape.min(ndim);

		// Merge the dimensions that we are going to reshape
		let dims_to_reshape = &dims[n_dims_to_keep..];
		let merged = DimMerger::new([dims_to_reshape]);

		// Resize the dims array. The new dimensions will be initialized in the `for` loop below.
		unsafe { dims.set_len(n_dims_to_keep) };
		dims.reserve(new_shape.len());
		unsafe { dims.set_len(n_dims_to_keep + new_shape.len()) };
		let new_dims = &mut dims[n_dims_to_keep..];

		// Try to match the new shape with the merged dimensions
		let smallest_dim = merged.smallest_dim();
		let mut prev_stride = smallest_dim.strides[0];
		let mut target_stride = smallest_dim.size * smallest_dim.strides[0];

		let merged_dims = merged.dims_increasing_without_smallest();
		let mut merged_dims = merged_dims.iter();

		for (dims_slot, size) in new_dims.iter_mut().zip(new_shape.iter().copied()).rev() {
			// We are about to store `size` into a slot in `dims`,
			// but first we need to calculate the stride

			let mut new_stride = prev_stride * size;

			if new_stride > target_stride {
				cold_path();

				if prev_stride == target_stride {
					let Some(merged_dim) = merged_dims.next() else {
						panic!("incompatible reshape");
					};
					prev_stride = merged_dim.strides[0];
					target_stride = merged_dim.size * merged_dim.strides[0];

					new_stride = prev_stride * size;
					if new_stride > target_stride {
						panic!("incompatible reshape");
					}
				} else {
					panic!("incompatible reshape");
				}
			}

			*dims_slot = SizeAndStride { size, stride: new_stride };
			prev_stride = new_stride;
		}

		assert!(merged_dims.is_empty());
		assert!(prev_stride == target_stride);

		self
	}

	pub fn reshape_all(self, new_shape: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		self.reshape(ndim, new_shape)
	}

	pub fn dim_to_positive(&self, dim: isize) -> usize {
		let ndim = self.dims.len();
		let dim = if dim >= 0 { dim as usize } else { ndim.wrapping_add(dim as usize) };
		if likely(dim < ndim) {
			dim
		} else {
			panic!("dimension out of range");
		}
	}

	pub fn transposed(mut self, dim1: isize, dim2: isize) -> Tensor {
		let dim1 = self.dim_to_positive(dim1);
		let dim2 = self.dim_to_positive(dim2);

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
}

/*
//--------------------------------------------------------------------------------------------------

// c = a * b * alpha + c * beta
fn __gemm<'a, O: OutputHandler>(a: Matrix<'a>, b: Matrix<'a>, alpha: f64, beta: f64, c: &mut O) {
	assert_compatible_types(a.tensor, b.tensor);
	assert_compatible_devices(a.tensor, b.tensor);

	assert!(a.cols.size == b.rows.size, "incompatible dimensions");
	let c_rows = SizeAndStride { size: a.rows.size, stride: b.cols.size };
	let c_cols = SizeAndStride { size: b.cols.size, stride: 1 };

	let dtype = a.tensor.dtype;
	let batch_ndim = a.batch_dims.len().max(b.batch_dims.len());
	c.init(batch_ndim + 2, dtype);

	c.prepend_dim(c_cols.size, false);
	c.prepend_dim(c_rows.size, false);

	let batch = Batch::new(batch_ndim, [a.batch_dims, b.batch_dims], c);

	let a_rows_contiguous = a.cols.is_contiguous();
	let a_cols_contiguous = a.rows.is_contiguous();
	let transa = !a_rows_contiguous;
	let lda = if a_rows_contiguous { a.rows.stride } else { a.cols.stride };
	assert!(
		a_rows_contiguous || a_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	let b_rows_contiguous = b.cols.is_contiguous();
	let b_cols_contiguous = b.rows.is_contiguous();
	let transb = !b_rows_contiguous;
	let ldb = if b_rows_contiguous { b.rows.stride } else { b.cols.stride };
	assert!(
		b_rows_contiguous || b_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	let m = c_rows.size;
	let n = c_cols.size;
	let k = a.cols.size;

	let c_rows_contiguous = c_cols.is_contiguous();
	let c_cols_contiguous = c_rows.is_contiguous();
	let transc = !c_rows_contiguous;
	let ldc = if c_rows_contiguous { c_rows.stride } else { c_cols.stride };
	assert!(
		c_rows_contiguous || c_cols_contiguous,
		"at least one of the matrix dimensions must be contiguous"
	);

	let (a, lda, transa, b, ldb, transb, m, n) = {
		if transc {
			(b, ldb, !transb, a, lda, !transa, n, m)
		} else {
			(a, lda, transa, b, ldb, transb, m, n)
		}
	};

	let a_buf = buf_to_base(a.tensor.buffer.as_ref());
	let b_buf = buf_to_base(b.tensor.buffer.as_ref());
	#[rustfmt::skip]
	batch.run(
		c.make_buffer(), [a.tensor.bufoff(), b.tensor.bufoff()],
		&|
			c: BatchBufOff<&dyn Buffer>,
			[a, b]: [BatchBufOff<&BufferBase>; 2],
			batch_size: usize
		| unsafe {
			c.buffer.gemm(
				dtype, c.offset, ldc, c.batch_stride,
				m, n, k,
				a_buf, a.offset, lda, transa, a.batch_stride,
				b_buf, b.offset, ldb, transb, b.batch_stride,
				alpha, beta,
				batch_size,
			);
		},
	);
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

#[derive(Clone, Copy)]
pub struct MatMul<'a> {
	pub a: Matrix<'a>,
	pub b: Matrix<'a>,
	pub scale: f64,
}

impl<'a> MatMul<'a> {
	pub fn new(a: Matrix<'a>, b: Matrix<'a>) -> MatMul<'a> {
		assert!(a.cols.size == b.rows.size);
		MatMul { a, b, scale: 1.0 }
	}

	/// We have:
	/// - matrix `A` with shape `[m, k]` and elements with mean = `0`, variance = `var_a`
	/// - matrix `B` with shape `[k, n]` and elements with mean = `0`, variance = `var_b`
	///
	/// The matrix multiplication `mm(A, B)` will have elements with
	/// mean = `0`, variance = `var_a * var_b * k`.
	///
	/// If we scale the result by the constant returned by this function (`1 /
	/// sqrt(k)`), the variance of the result elements will be independent on
	/// `k`. It will just be `var_a * var_b`.
	pub fn normalizing_scale(k: usize) -> f64 {
		1.0 / (k as f64).sqrt()
	}

	pub fn scale(&self, scale: f64) -> MatMul<'a> {
		MatMul {
			a: self.a,
			b: self.b,
			scale: self.scale * scale,
		}
	}

	/// Calculate the MatMul and return a new tensor with the result.
	/// ```
	///     result = a * b * scale
	/// ```
	pub fn eval(&self) -> Tensor {
		let mut c = OutputBuilder::new(self.a.tensor.device());
		__gemm(self.a, self.b, self.scale, 0.0, &mut c);
		c.make_tensor()
	}

	pub fn eval_into(&self, into: &Tensor) {
		let mut c = OutputRef::new(into);
		__gemm(self.a, self.b, self.scale, 0.0, &mut c);
	}

	pub fn acc_into(&self, into: &Tensor) {
		let mut c = OutputRef::new(into);
		__gemm(self.a, self.b, self.scale, 1.0, &mut c);
	}

	/// In debug builds, assert that the current scale is equal to the
	/// normalizing scale.
	///
	/// I.e., to the value returned by `normalizing_scale(self.a.cols.size)`.
	///
	/// If ok, return `self`. Otherwise, panic.
	pub fn assert_normalizing_scale(&self) -> MatMul<'a> {
		debug_assert!(self.scale == Self::normalizing_scale(self.a.cols.size));
		*self
	}

	pub fn backward<M: MatrixLike<'a>>(&self, dy: M) -> MatMulBackward<'a> {
		let dy = dy.as_matrix();
		assert!(dy.rows.size == self.a.rows.size);
		assert!(dy.cols.size == self.b.cols.size);
		debug_assert!(
			self.scale == 1.0,
			concat!(
				"TODO - we'd need to propagate the scale to MatMulBackward. ",
				"Not sure if it's worth it. ",
				"The backward pass should normally use its own scale."
			)
		);
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
	a: Option<Matrix<'a>>, b: Option<Matrix<'a>>, dy: Matrix<'a>,
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

fn __norm<
	O: OutputHandler,
	RunOp: Fn(BatchBufOff<&dyn Buffer>, BatchBufOff<&BufferBase>, CommonArgs1D),
>(
	a: &Tensor, out: &mut O, run_op: RunOp,
) {
	let ndim = a.ndim();
	let dtype = a.dtype;

	assert!(ndim > 0);
	let batch_ndim = ndim - 1;

	// The last dimension is the one we are normalizing
	let dim = a.dims[ndim - 1];
	assert!(dim.is_contiguous(), "the normalized dimension must be contiguous");

	out.init(ndim, dtype);
	let out_dim = out.prepend_dim(dim.size, false);
	assert!(out_dim.is_contiguous(), "the output dimension must be contiguous");
	let batch = Batch::new(batch_ndim, [&a.dims[..batch_ndim]], out);

	#[rustfmt::skip]
	batch.run(
		out.make_buffer(), [a.bufoff()],
		&|
			out: BatchBufOff<&dyn Buffer>,
			[a]: [BatchBufOff<&BufferBase>; 1],
			batch_size: usize
		| {
			run_op(
				out, a,
				CommonArgs1D {
					dtype,
					len: dim.size,
					batch_size,
				},
			);
		}
	);
}

pub fn rms_norm(a: &Tensor, eps: f64) -> Tensor {
	let mut out = OutputBuilder::new(a.device());
	#[rustfmt::skip]
	__norm(
		a, &mut out,
		|
			o: BatchBufOff<&dyn Buffer>,
			a: BatchBufOff<&BufferBase>,
			common: CommonArgs1D
		| unsafe {
			o.buffer.rms_norm(o.without_buf(), a, common, eps)
		},
	);
	out.make_tensor()
}

pub fn softmax(a: &Tensor) -> Tensor {
	let mut out = OutputBuilder::new(a.device());
	#[rustfmt::skip]
	__norm(
		a, &mut out,
		|
			o: BatchBufOff<&dyn Buffer>,
			a: BatchBufOff<&BufferBase>,
			common: CommonArgs1D
		| unsafe {
			o.buffer.softmax(o.without_buf(), a, common)
		},
	);
	out.make_tensor()
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
*/
