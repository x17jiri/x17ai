// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use core::panic;
use smallvec::{smallvec, SmallVec};
use std::alloc::Layout;
use std::fmt;
use std::intrinsics::{likely, unlikely};
use std::iter::ExactSizeIterator;
use std::mem::MaybeUninit;
use std::ops::Index;
use std::rc::{Rc, Weak};
use thin_vec::ThinVec;

pub const INLINE_DIMS: usize = 5;

#[derive(Clone, Copy, PartialEq, Default)]
pub struct SizeAndStride {
	pub size: usize,
	pub stride: usize,
}

impl SizeAndStride {
	pub fn is_contiguous(&self) -> bool {
		self.size == 1 || self.stride == 1
	}
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
		self.dims.iter().map(|x| &x.size)
	}
}

pub trait OutputHandler {
	fn init(&mut self, ndim: usize, dtype: DType);

	/// Adds a new dimension with the given size.
	///
	/// Returns the size and stride of the new dimension.
	fn prepend_dim(&mut self, size: usize, allow_broadcast: bool) -> SizeAndStride;

	/// Create a new buffer that can fit a new tensor with all the previously
	/// added dimensions and the given dtype.
	///
	/// Return the buffer and the offset in the buffer where the tensor should
	/// start.
	fn make_buffer(&mut self) -> BufOff<&dyn Buffer>;
}

/// This struct is used mainly to ensure that the total number of elements does
/// not overflow
struct OutputBuilder {
	dims: SmallVec<[SizeAndStride; INLINE_DIMS]>,
	dtype: DType,

	/// How many more dims do we need to prepend.
	remaining_dims: usize,

	elems: usize,

	/// Total number of elements in the dimensions processed so far but ignoring
	/// zero length dimensions.
	nonzero_elems: usize,

	device: Option<Rc<dyn Device>>,
	buffer: Option<Rc<dyn Buffer>>,
}

impl OutputBuilder {
	pub fn new(device: Rc<dyn Device>) -> OutputBuilder {
		OutputBuilder {
			dims: SmallVec::new(),
			dtype: DType::f32(),
			remaining_dims: 0,
			elems: 0,
			nonzero_elems: 0,
			device: Some(device),
			buffer: None,
		}
	}

	// Note: this function expects that make_buffer() has been called before.
	pub fn make_tensor(self) -> Tensor {
		Tensor {
			dims: self.dims,
			offset: 0,
			dtype: self.dtype,
			elems: self.elems,
			buffer: self.buffer.unwrap(),
		}
	}
}

impl OutputHandler for OutputBuilder {
	fn init(&mut self, ndim: usize, dtype: DType) {
		self.dims = SmallVec::with_capacity(ndim);
		unsafe { self.dims.set_len(ndim) };
		self.dtype = dtype;
		self.remaining_dims = ndim;
		self.elems = 1;
		self.nonzero_elems = 1;
	}

	fn prepend_dim(&mut self, size: usize, _allow_broadcast: bool) -> SizeAndStride {
		debug_assert!(self.remaining_dims > 0);

		// Check that if we ignore zero length dimensions, the number of elements does not overflow.
		// This is done to make sure our calculations would not overflow even if we had
		// the same dimensions but in different order.
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

		SizeAndStride { size, stride }
	}

	fn make_buffer(&mut self) -> BufOff<&dyn Buffer> {
		if unlikely(self.remaining_dims != 0) {
			panic!("not all dimensions were added");
		}
		let Some(device) = self.device.take() else {
			panic!("buffer already created");
		};

		self.buffer = Some(device.new_buffer(self.dtype, self.elems));

		BufOff {
			buffer: self.buffer.as_ref().unwrap().as_ref(),
			offset: 0,
		}
	}
}

pub struct OutputRef<'a> {
	tensor: &'a Tensor,
	remaining_dims: usize,
}

impl OutputRef<'_> {
	pub fn new(tensor: &Tensor) -> OutputRef {
		OutputRef {
			tensor,
			remaining_dims: tensor.dims.len(),
		}
	}
}

impl OutputHandler for OutputRef<'_> {
	fn init(&mut self, ndim: usize, dtype: DType) {
		assert!(self.tensor.dims.len() == ndim, "incompatible shape");
		assert!(self.tensor.dtype == dtype, "incompatible dtype");
		self.remaining_dims = ndim;
	}

	fn prepend_dim(&mut self, size: usize, allow_broadcast: bool) -> SizeAndStride {
		assert!(self.remaining_dims > 0);
		self.remaining_dims -= 1;
		debug_assert!(self.remaining_dims < self.tensor.dims.len());
		let dim = unsafe { *self.tensor.dims.get_unchecked(self.remaining_dims) };
		assert!(allow_broadcast || dim.size == size, "incompatible shape");
		dim
	}

	fn make_buffer(&mut self) -> BufOff<&dyn Buffer> {
		assert!(self.remaining_dims == 0, "not all dimensions were added");
		BufOff {
			buffer: self.tensor.buffer.as_ref(),
			offset: self.tensor.offset,
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
	/// Allocate a new tensor on the provided device.
	pub fn new_empty_on<'a, Shape>(shape: Shape, dtype: DType, device: Rc<dyn Device>) -> Tensor
	where
		Shape: IntoIterator<Item = &'a usize>,
		<Shape as IntoIterator>::IntoIter:
			DoubleEndedIterator<Item = &'a usize> + ExactSizeIterator,
	{
		let shape = shape.into_iter().copied();
		let mut builder = OutputBuilder::new(device);
		builder.init(shape.len(), dtype);
		for dim in shape.rev() {
			builder.prepend_dim(dim, false);
		}
		builder.make_buffer();
		builder.make_tensor()
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
		buf_to_base(self.buffer.as_ref()).device.clone()
	}

	pub fn shape(&self) -> ShapeView {
		ShapeView { dims: &self.dims }
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
			merged.prepend_dim(dim.size, elems, [dim.stride]);
			elems *= dim.size;
		}
		let mut merged = merged.rev_dims.iter();

		let (mut prev_stride, mut target_stride) = {
			if let Some(merged_dim) = merged.next() {
				(merged_dim.in_strides[0], merged_dim.size * merged_dim.in_strides[0])
			} else {
				cold_path();
				(1, 1)
			}
		};

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

	pub fn reshape(self, new_shape: &[usize]) -> Tensor {
		let ndim = self.dims.len();
		self.reshape_last_n(ndim, new_shape)
	}

	fn __dim_to_internal(&self, dim: isize) -> usize {
		let ndim = self.dims.len();
		let dim = if dim >= 0 { dim as usize } else { ndim - ((-dim) as usize) };
		if likely(dim < ndim) {
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

	pub fn bufoff(&self) -> BufOff<&BufferBase> {
		BufOff {
			buffer: buf_to_base(self.buffer.as_ref()),
			offset: self.offset,
		}
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

/// Fill the tensor with zeros.
pub fn zeros_(tensor: &Tensor) {
	tensor.buffer.zeros_(tensor);
}

/// Fill the tensor with random values from a normal distribution.
/// ```
///     mean = 0, var = 1
/// ```
pub fn randn_(tensor: &Tensor) {
	tensor.buffer.randn_(tensor);
}

fn __elem_wise<
	const N: usize,
	O: OutputHandler,
	RunOp: Fn(BatchBufOff<&dyn Buffer>, [BatchBufOff<&BufferBase>; N], CommonArgs1D),
>(
	ndim: usize, dtype: DType, a: [&Tensor; N], out: &mut O, run_op: RunOp,
) {
	for i in 1..N {
		assert_compatible_types(a[0], a[i]);
		assert_compatible_devices(a[0], a[i]);
		assert!(a[i].ndim() <= ndim, "incompatible number of dimensions");
	}

	out.init(ndim, dtype);
	let inputs = a.map(|a| a.dims.as_slice());
	let mut batch = Batch::new(ndim, inputs, out);

	let op_dim = batch.pop_dim();
	assert!(op_dim.out_stride == 1);
	assert!(op_dim.in_strides == [1; N]);

	#[rustfmt::skip]
	batch.run(
		out.make_buffer(), a.map(|i| i.bufoff()),
		&|out: BatchBufOff<&dyn Buffer>, a: [BatchBufOff<&BufferBase>; N], batch_size: usize| {
			run_op(
				out, a,
				CommonArgs1D {
					dtype,
					len: op_dim.size,
					batch_size,
				}
			);
		}
	)
}

/// Accumulate:
/// ```
///     a = a * alpha + b * beta
/// ```
pub fn acc_(a: &Tensor, alpha: f64, b: &Tensor, beta: f64) {
	#[rustfmt::skip] __elem_wise(
		a.ndim(), a.dtype, [b], &mut OutputRef::new(a),
		&|
			a: BatchBufOff<&dyn Buffer>,
			[b]: [BatchBufOff<&BufferBase>; 1],
			common: CommonArgs1D
		| unsafe {
			a.buffer.acc(a.without_buf(), b, common, alpha, beta)
		},
	);
}

/// Accumulate the result of element-wise multiplication:
/// ```
///     a = a * alpha + (b * c) * beta
/// ```
pub fn acc_mul_(a: &Tensor, alpha: f64, b: &Tensor, c: &Tensor, beta: f64) {
	#[rustfmt::skip] __elem_wise(
		a.ndim(), a.dtype, [b, c], &mut OutputRef::new(a),
		|
			a: BatchBufOff<&dyn Buffer>,
			[b, c]: [BatchBufOff<&BufferBase>; 2],
			common: CommonArgs1D
		| unsafe {
			a.buffer.acc_mul(a.without_buf(), b, c, common, alpha, beta)
		},
	);
}

/// Accumulate the result of sum:
/// ```
///     a = a * alpha + sum(b, keepdim) * beta
/// ```
pub fn acc_sum_(a: &Tensor, alpha: f64, b: &Tensor, keepdim: bool, beta: f64) {
	assert_compatible_types(a, b);
	assert_compatible_devices(a, b);

	// The last dimension of `b` is the one we are summing over.
	assert!(b.ndim() > 0);
	let dim = b.dims[b.dims.len() - 1];
	assert!(dim.is_contiguous(), "the summed dimension must be contiguous");

	let mut out = OutputRef::new(a);

	let ndim = a.ndim();
	let dtype = a.dtype;
	out.init(ndim, dtype);

	let batch_ndim;
	if keepdim {
		// The last dimension of `a` has size 1 and is not part of the batch dimensions.
		out.prepend_dim(1, false);
		batch_ndim = ndim - 1;
	} else {
		batch_ndim = ndim;
	};

	let batch = Batch::new(batch_ndim, [&b.dims[..b.ndim() - 1]], &mut out);

	#[rustfmt::skip]
	batch.run(
		out.make_buffer(), [b.bufoff()],
		&|
			out: BatchBufOff<&dyn Buffer>,
			[a]: [BatchBufOff<&BufferBase>; 1],
			batch_size: usize
		| unsafe {
			out.buffer.acc_sum(
				out.without_buf(), a,
				CommonArgs1D {
					dtype,
					len: dim.size,
					batch_size,
				},
				alpha, beta,
			);
	});
}

/// Accumulate the result of mean:
/// ```
///     a = a * alpha + mean(b, keepdim) * beta
/// ```
pub fn acc_mean_(a: &Tensor, alpha: f64, b: &Tensor, keepdim: bool, beta: f64) {
	// We can convert this to `acc_sum_()` because `mean = sum / n`
	let n = b.size(-1) as f64;
	acc_sum_(a, alpha, b, keepdim, beta / n);
}

/// Calculate `x ^ 2`:
/// ```
///     result = x * x
/// ```
pub fn square(x: &Tensor) -> Tensor {
	let out = x.new_empty_like();
	acc_mul_(&out, 0.0, x, x, 1.0);
	out
}

/// reciprocal of the square root:
/// ```
///     result = 1.0 / (a.sqrt() + eps)
/// ```
pub fn rsqrt(a: &Tensor, eps: f64) -> Tensor {
	let result = a.new_empty_like();
	#[rustfmt::skip] __elem_wise(
		a.ndim(), a.dtype, [a], &mut OutputRef::new(&result),
		|
			r: BatchBufOff<&dyn Buffer>,
			[a]: [BatchBufOff<&BufferBase>; 1],
			common: CommonArgs1D
		| unsafe {
			r.buffer.rsqrt(r.without_buf(), a, common, eps)
		},
	);
	result
}

pub fn rsqrt_into(a: &Tensor, eps: f64, into: &Tensor) {
	#[rustfmt::skip] __elem_wise(
		a.ndim(), a.dtype, [a], &mut OutputRef::new(into),
		|
			r: BatchBufOff<&dyn Buffer>,
			[a]: [BatchBufOff<&BufferBase>; 1],
			common: CommonArgs1D
		| unsafe {
			r.buffer.rsqrt(r.without_buf(), a, common, eps)
		},
	);
}

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

	pub popped_dims: usize,
}

impl<const N: usize> Batch<N> {
	pub fn new<O: OutputHandler>(
		ndim: usize, inputs: [&[SizeAndStride]; N], out: &mut O,
	) -> Batch<N> {
		let mut batch = Batch::new_empty(ndim);

		// take the iterator to input dimensions, reverse it and append inf number of 1s
		let mut inputs: [_; N] = inputs.map(|i| {
			i.iter().rev().copied().chain(std::iter::repeat(SizeAndStride { size: 1, stride: 1 }))
		});

		for _ in 0..ndim {
			// Get sizes and strides for the current dimension from all inputs
			let mut in_dims = [Default::default(); N];
			for i in 0..N {
				in_dims[i] = inputs[i].next().unwrap();
			}

			// The dim_size should be the same for all inputs except for broadcasted inputs
			// So max() gets the dim_size of the non-broadcasted inputs.
			let dim_size = in_dims.iter().map(|i| i.size).max();
			let dim_size = dim_size.unwrap_or(1);
			let out_dim = out.prepend_dim(dim_size, true);

			// Get input strides. If needed, try to broadcast
			let in_strides = in_dims.map(|i| {
				if i.size == out_dim.size {
					i.stride
				} else {
					assert!(i.size == 1, "cannot broadcast: incompatible dimensions");
					0
				}
			});

			batch.prepend_dim(out_dim.size, out_dim.stride, in_strides);
		}

		batch
	}

	pub fn new_empty(ndim: usize) -> Batch<N> {
		Batch {
			rev_dims: SmallVec::with_capacity(ndim),
			popped_dims: 0,
		}
	}

	// dimensions should be prepended in the order of increasing strides
	pub fn prepend_dim(&mut self, size: usize, out_stride: usize, in_strides: [usize; N]) {
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

		// Can't merge; prepend the new dimension
		self.rev_dims.push(BatchDim { size, out_stride, in_strides });
	}

	pub fn pop_dim(&mut self) -> BatchDim<N> {
		if likely(self.rev_dims.len() > self.popped_dims) {
			let result = self.rev_dims[self.popped_dims];
			self.popped_dims += 1;
			result
		} else {
			BatchDim {
				size: 1,
				out_stride: 1,
				in_strides: [1; N],
			}
		}
	}

	fn __run<O: Copy, I: Copy, F: Fn(BatchBufOff<O>, [BatchBufOff<I>; N], usize)>(
		&self, mut out: BufOff<O>, mut in_: [BufOff<I>; N], f: &F, batch: &[BatchDim<N>],
	) {
		debug_assert!(!batch.is_empty());
		let batch_dim = unsafe { batch.get_unchecked(batch.len() - 1) };
		let sub_batch = unsafe { batch.get_unchecked(..batch.len() - 1) };
		if sub_batch.is_empty() {
			let out = out.make_batch(batch_dim.out_stride);
			let mut in_ = in_.map(|i| i.make_batch(0));
			for i in 0..N {
				in_[i].batch_stride = batch_dim.in_strides[i];
			}
			let batch_size = batch_dim.size;
			f(out, in_, batch_size);
		} else {
			for _ in 0..batch_dim.size {
				self.__run(out, in_, f, sub_batch);

				out.offset += batch_dim.out_stride;
				for i in 0..N {
					in_[i].offset += batch_dim.in_strides[i];
				}
			}
		}
	}

	pub fn run<O: Copy, I: Copy, F: Fn(BatchBufOff<O>, [BatchBufOff<I>; N], usize)>(
		&self, out: BufOff<O>, in_: [BufOff<I>; N], f: &F,
	) {
		if self.rev_dims.len() <= self.popped_dims {
			let out = out.make_batch(0);
			let in_ = in_.map(|i| i.make_batch(0));
			let batch_size = 1;
			f(out, in_, batch_size);
		} else {
			self.__run(out, in_, f, &self.rev_dims[self.popped_dims..]);
		}
	}
}
