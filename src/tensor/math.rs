//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;

use crate::ErrPack;
use crate::tensor::device::DeviceBuffer;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};
use crate::tensor::dim_merger::{DimMerger, MergedDim};
use crate::tensor::generic::map::{ND, NotEnoughDimensionsError, SizeAndStride};
use crate::tensor::{Tensor, TensorOpError, generic};
use crate::util::array;

//--------------------------------------------------------------------------------------------------

pub trait EvaluatesToTensor {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` tensor.
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>>;
}

pub trait EvaluatesToMatrix {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` matrix.
	fn eval_to_matrix(self, to: &Matrix) -> Result<(), ErrPack<TensorOpError>>;
}

pub trait EvaluatesToColMatrix {
	fn eval_to_col_matrix(self, to: &ColMatrix) -> Result<(), ErrPack<TensorOpError>>;
}

pub trait EvaluatesToRowMatrix {
	fn eval_to_row_matrix(self, to: &RowMatrix) -> Result<(), ErrPack<TensorOpError>>;
}

//--------------------------------------------------------------------------------------------------

fn check_borrows(c_fail: usize, fail: usize) -> Result<(), ErrPack<TensorOpError>> {
	if fail == 0 {
		Ok(())
	} else {
		cold_path();
		if c_fail == 0 {
			Err(ErrPack {
				code: TensorOpError::CannotBorrowMut,
				extra: None,
			})
		} else {
			Err(ErrPack {
				code: TensorOpError::CannotBorrow,
				extra: None,
			})
		}
	}
}

//--------------------------------------------------------------------------------------------------

unsafe fn run_batch<
	'a,
	const N: usize,
	const M: usize,
	const C: usize,
	const O: usize,
	const K: usize,
>(
	batch_dim: MergedDim<{ M + C }>,
	m_tensors: &mut [generic::Tensor<ND<N>, DeviceBufferRefMut<'a>>; O],
	c_tensors: &mut [generic::Tensor<ND<N>, DeviceBufferRef<'a>>; K],
	mut f: impl FnMut(
		&mut [generic::Tensor<ND<N>, DeviceBufferRefMut<'a>>; O],
		&[generic::Tensor<ND<N>, DeviceBufferRef<'a>>; K],
	) -> Result<(), ErrPack<TensorOpError>>,
) -> Result<(), ErrPack<TensorOpError>>
where
	[(); M - O]:,
	[(); C - K]:,
{
	if batch_dim.size == 0 {
		cold_path();
		return Ok(());
	}

	let i = batch_dim.size - 1;
	loop {
		f(m_tensors, c_tensors)?;
		if i > 0 {
			cold_path();
			for (tensor, &stride) in m_tensors.iter_mut().zip(&batch_dim.strides) {
				unsafe { tensor.map_mut().offset += stride }
			}
			for (tensor, &stride) in c_tensors.iter_mut().zip(&batch_dim.strides[M..]) {
				unsafe { tensor.map_mut().offset += stride }
			}
		} else {
			break;
		}
	}
	Ok(())
}

//--------------------------------------------------------------------------------------------------

// TODO - disable broadcast for M tensors.

pub struct ElemWise<'a, const M: usize, const C: usize>
where
	[(); M + C]:,
{
	m: [&'a Tensor; M],
	c: [&'a Tensor; C],
	dims: [MergedDim<{ M + C }>; 3],
}

impl<'a, const M: usize, const C: usize> ElemWise<'a, M, C>
where
	[(); M + C]:,
{
	pub fn new(m: [&'a Tensor; M], c: [&'a Tensor; C]) -> Result<Self, ErrPack<TensorOpError>> {
		let m_dims = m.map(|t| t.map().dims.as_slice());
		let c_dims = c.map(|t| t.map().dims.as_slice());
		let dims = DimMerger::merge::<3>(array::concat_arrays(m_dims, c_dims))?;
		Ok(Self { m, c, dims })
	}

	/// Note that we use 'K' instead of 'C' and it is allowed that 'K <= C'.
	///
	/// So we don't have to use all the `c` tensors processed during `::new()`.
	pub fn run<const O: usize, const K: usize>(
		&self,
		mut f: impl FnMut(
			&mut [generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>; O],
			&[generic::Tensor<ND<2>, DeviceBufferRef<'a>>; K],
		) -> Result<(), ErrPack<TensorOpError>>,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); M - O]:,
		[(); C - K]:,
	{
		unsafe {
			let mut c_fail = 0;
			let mut c_tensors = std::array::from_fn(|i| {
				generic::Tensor::new_unchecked(
					ND {
						dims: [self.dims[1].get(M + i), self.dims[2].get(M + i)],
						offset: self.c[i].map().offset,
					},
					DeviceBufferRef::new_unsafe(self.c[i].buf().as_ref(), &mut c_fail),
				)
			});
			let mut fail = c_fail;
			let mut m_tensors = std::array::from_fn(|i| {
				generic::Tensor::new_unchecked(
					ND {
						dims: [self.dims[1].get(i), self.dims[2].get(i)],
						offset: self.m[i].map().offset,
					},
					DeviceBufferRefMut::new_unsafe(self.m[i].buf().as_ref(), &mut fail),
				)
			});
			check_borrows(c_fail, fail)?;

			run_batch(self.dims[0], &mut m_tensors, &mut c_tensors, f)
		}
	}

	pub fn are_identical<const MI: usize, const CI: usize>(&self) -> bool
	where
		[(); C - 1 - CI]:,
		[(); M - 1 - MI]:,
	{
		let m_tensor = &self.m[MI];
		let m_offset = m_tensor.map().offset;
		let m_buf = m_tensor.buf().as_ref();

		let c_tensor = &self.c[CI];
		let c_offset = c_tensor.map().offset;
		let c_buf = c_tensor.buf().as_ref();

		let m_stride0 = self.dims[0].strides[MI];
		let m_stride1 = self.dims[1].strides[MI];
		let m_stride2 = self.dims[2].strides[MI];

		let c_stride0 = self.dims[0].strides[M + CI];
		let c_stride1 = self.dims[1].strides[M + CI];
		let c_stride2 = self.dims[2].strides[M + CI];

		let buf_eq = std::ptr::eq(m_buf, c_buf);

		let a = buf_eq
			&& m_offset == c_offset
			&& (self.dims[0].size <= 1 || m_stride0 == c_stride0)
			&& (self.dims[1].size <= 1 || m_stride1 == c_stride1)
			&& (self.dims[2].size <= 1 || m_stride2 == c_stride2);

		let b = ((m_buf as *const DeviceBuffer as usize) ^ (c_buf as *const DeviceBuffer as usize))
			| (m_offset ^ c_offset)
			| (m_stride0 ^ c_stride0)
			| (m_stride1 ^ c_stride1)
			| (m_stride2 ^ c_stride2);
		let b = b == 0;

		debug_assert!(a == b);
		b
	}
}

//--------------------------------------------------------------------------------------------------

pub struct VecWise<'a, const M: usize, const C: usize>
where
	[(); M + C]:,
{
	m: [&'a Tensor; M],
	c: [&'a Tensor; C],
	dims: [MergedDim<{ M + C }>; 2],
	m_vec: [SizeAndStride; M],
	c_vec: [SizeAndStride; C],
}

impl<'a, const M: usize, const C: usize> VecWise<'a, M, C>
where
	[(); M + C]:,
{
	pub fn new(m: [&'a Tensor; M], c: [&'a Tensor; C]) -> Result<Self, ErrPack<TensorOpError>> {
		if m.iter().any(|t| t.ndim() < 1) || c.iter().any(|t| t.ndim() < 1) {
			return Err(TensorOpError::missing_vec_dimension());
		}

		// All dimensions except the feature dimension
		let m_dims = m.map(|t| t.map().dims.as_slice());
		let c_dims = c.map(|t| t.map().dims.as_slice());

		// the feature dimension
		let m_vec = m_dims.map(|d| *d.last().unwrap());
		let c_vec = c_dims.map(|d| *d.last().unwrap());

		let m_dims = m_dims.map(|d| &d[..d.len() - 1]);
		let c_dims = c_dims.map(|d| &d[..d.len() - 1]);
		let dims = DimMerger::merge::<2>(array::concat_arrays(m_dims, c_dims))?;

		Ok(Self { m, c, dims, m_vec, c_vec })
	}

	pub fn run<const O: usize, const K: usize>(
		&self,
		mut f: impl FnMut(
			&mut [generic::Tensor<ND<2>, DeviceBufferRefMut<'a>>; O],
			&[generic::Tensor<ND<2>, DeviceBufferRef<'a>>; K],
		) -> Result<(), ErrPack<TensorOpError>>,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); M - O]:,
		[(); C - K]:,
	{
		unsafe {
			let mut c_fail = 0;
			let mut c_tensors = std::array::from_fn(|i| {
				generic::Tensor::new_unchecked(
					ND {
						dims: [self.dims[1].get(M + i), self.c_vec[i]],
						offset: self.c[i].map().offset,
					},
					DeviceBufferRef::new_unsafe(self.c[i].buf().as_ref(), &mut c_fail),
				)
			});

			let mut fail = c_fail;
			let mut m_tensors = std::array::from_fn(|i| {
				generic::Tensor::new_unchecked(
					ND {
						dims: [self.dims[1].get(i), self.m_vec[i]],
						offset: self.m[i].map().offset,
					},
					DeviceBufferRefMut::new_unsafe(self.m[i].buf().as_ref(), &mut fail),
				)
			});
			check_borrows(c_fail, fail)?;

			run_batch(self.dims[0], &mut m_tensors, &mut c_tensors, f)
		}
	}

	pub fn are_identical<const MI: usize, const CI: usize>(&self) -> bool
	where
		[(); C - 1 - CI]:,
		[(); M - 1 - MI]:,
	{
		let m_tensor = &self.m[MI];
		let m_offset = m_tensor.map().offset;
		let m_buf = m_tensor.buf().as_ref();

		let c_tensor = &self.c[CI];
		let c_offset = c_tensor.map().offset;
		let c_buf = c_tensor.buf().as_ref();

		let m_size0 = self.m_vec[MI].size;
		let m_stride0 = self.m_vec[MI].stride;
		let m_stride1 = self.dims[0].strides[MI];
		let m_stride2 = self.dims[1].strides[MI];

		let c_size0 = self.c_vec[CI].size;
		let c_stride0 = self.c_vec[CI].stride;
		let c_stride1 = self.dims[0].strides[M + CI];
		let c_stride2 = self.dims[1].strides[M + CI];

		let a = //.
			std::ptr::eq(m_buf, c_buf)
			&& m_offset == c_offset
			&& (m_size0 == c_size0)
			&& (m_size0 <= 1 || m_stride0 == c_stride0)
			&& (self.dims[0].size <= 1 || m_stride1 == c_stride1)
			&& (self.dims[1].size <= 1 || m_stride2 == c_stride2);

		let b = ((m_buf as *const DeviceBuffer as usize) ^ (c_buf as *const DeviceBuffer as usize))
			| (m_offset ^ c_offset)
			| (m_stride1 ^ c_stride1)
			| (m_stride2 ^ c_stride2);
		let b = //.
			b == 0
			&& (m_size0 == c_size0)
			&& (m_size0 <= 1 || m_stride0 == c_stride0);

		debug_assert!(a == b);
		b
	}
}

//--------------------------------------------------------------------------------------------------

/*
/// At least one of the matrix dimensions should be contiguous.
///
/// Batch dimensions broadcast is disabled for matrices[0] and enabled for matrices[1..].
/// This is by design.
fn __mat_wise<'a, const N: usize, F: Fn([MatrixSet; N])>(
	matrices: [&Matrix; N], batch_dims: MergedDimIter<N>, f: F,
) {
	assert!(matrices.iter().all(|m| {
		let row_dim = SizeAndStride { size: m.rows.get(), stride: m.row_stride };
		let col_dim = SizeAndStride { size: m.cols.get(), stride: m.col_stride };
		row_dim.is_contiguous() || col_dim.is_contiguous()
	}));
	batch::run(
		batch_dims,
		matrices.map(|m| m.tensor.offset),
		|batch_size: usize, batch_strides: [usize; N], offsets: [usize; N]| {
			f(std::array::from_fn(|i| MatrixSet {
				slice_set: SliceSet {
					buffer: matrices[i].tensor.buffer.as_ref(),
					dtype: matrices[i].tensor.dtype(),
					offset: offsets[i],
					len: MatrixSet::slice_len(
						matrices[i].rows,
						matrices[i].cols,
						matrices[i].row_stride,
						matrices[i].col_stride,
					),
					count: batch_size,
					stride: batch_strides[i],
				},

				rows: matrices[i].rows,
				cols: matrices[i].cols,
				row_stride: matrices[i].row_stride,
				col_stride: matrices[i].col_stride,
			}));
		},
	);
}
*/

//--------------------------------------------------------------------------------------------------

pub struct ZerosExpr();

pub fn zeros() -> ZerosExpr {
	ZerosExpr()
}

impl EvaluatesToTensor for ZerosExpr {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		ElemWise::new([to], [])?.run(|[to], []| {
			executor.zeros(to)?;
			Ok(())
		})
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RandnClampedExpr();

pub fn randn_clamped() -> RandnClampedExpr {
	RandnClampedExpr()
}

impl EvaluatesToTensor for RandnClampedExpr {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		ElemWise::new([to], [])?.run(|[to], []| {
			executor.randn_clamped(to)?;
			Ok(())
		})
	}
}

//--------------------------------------------------------------------------------------------------
/*
impl EvaluatesToTensor for Tensor {
	#[inline(never)]
	fn eval_to_tensor(&self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		ElemWise::new([to], [self])?.run(|[to], [input]| {
			executor.copy(to, input)?;
			Ok(())
		})
	}
}
*/
//--------------------------------------------------------------------------------------------------
// Scaling, i.e., multyplication by a scalar.

pub trait Scalable {
	type Output;
	fn scale(self, scale: f64) -> Self::Output;
}

#[derive(Clone, Copy)]
pub struct ScaledTensorExpr<'a> {
	pub tensor: &'a Tensor,
	pub scale: f64,
}

impl<'a> From<&'a Tensor> for ScaledTensorExpr<'a> {
	fn from(tensor: &'a Tensor) -> Self {
		Self { tensor, scale: 1.0 }
	}
}

impl<'a> Scalable for &'a Tensor {
	type Output = ScaledTensorExpr<'a>;
	fn scale(self, scale: f64) -> Self::Output {
		ScaledTensorExpr { tensor: self, scale }
	}
}

impl<'a> Scalable for ScaledTensorExpr<'a> {
	type Output = Self;
	fn scale(self, scale: f64) -> Self::Output {
		Self {
			tensor: self.tensor,
			scale: self.scale * scale,
		}
	}
}

impl<'a> std::ops::Mul<f64> for &'a Tensor {
	type Output = ScaledTensorExpr<'a>;
	fn mul(self, scale: f64) -> Self::Output {
		self.scale(scale)
	}
}

impl<'a> std::ops::Mul<f64> for ScaledTensorExpr<'a> {
	type Output = Self;
	fn mul(self, scale: f64) -> Self::Output {
		self.scale(scale)
	}
}

impl<'a> std::ops::Mul<f64> for AddWeightedExpr<'a> {
	type Output = Self;
	fn mul(self, scale: f64) -> Self::Output {
		self.scale(scale)
	}
}

impl<'a> std::ops::Mul<f64> for DotExpr<'a> {
	type Output = Self;
	fn mul(self, scale: f64) -> Self::Output {
		self.scale(scale)
	}
}

impl<'a> std::ops::Mul<f64> for DotAddExpr<'a> {
	type Output = Self;
	fn mul(self, scale: f64) -> Self::Output {
		self.scale(scale)
	}
}

impl<'a> std::ops::Mul<f64> for MulExpr<'a> {
	type Output = ScaledMulExpr<'a>;
	fn mul(self, scale: f64) -> Self::Output {
		self.scale(scale)
	}
}

impl<'a> std::ops::Mul<f64> for ScaledMulExpr<'a> {
	type Output = Self;
	fn mul(self, scale: f64) -> Self::Output {
		self.scale(scale)
	}
}

impl<'a> std::ops::Mul<&'a Tensor> for f64 {
	type Output = ScaledTensorExpr<'a>;
	fn mul(self, tensor: &'a Tensor) -> Self::Output {
		tensor.scale(self)
	}
}

impl<'a> std::ops::Mul<ScaledTensorExpr<'a>> for f64 {
	type Output = ScaledTensorExpr<'a>;
	fn mul(self, tensor: ScaledTensorExpr<'a>) -> Self::Output {
		tensor.scale(self)
	}
}

impl<'a> std::ops::Mul<AddWeightedExpr<'a>> for f64 {
	type Output = AddWeightedExpr<'a>;
	fn mul(self, expr: AddWeightedExpr<'a>) -> Self::Output {
		expr.scale(self)
	}
}

impl<'a> std::ops::Mul<DotExpr<'a>> for f64 {
	type Output = DotExpr<'a>;
	fn mul(self, expr: DotExpr<'a>) -> Self::Output {
		expr.scale(self)
	}
}

impl<'a> std::ops::Mul<DotAddExpr<'a>> for f64 {
	type Output = DotAddExpr<'a>;
	fn mul(self, expr: DotAddExpr<'a>) -> Self::Output {
		expr.scale(self)
	}
}

impl<'a> std::ops::Mul<MulExpr<'a>> for f64 {
	type Output = ScaledMulExpr<'a>;
	fn mul(self, expr: MulExpr<'a>) -> Self::Output {
		expr.scale(self)
	}
}

impl<'a> std::ops::Mul<ScaledMulExpr<'a>> for f64 {
	type Output = ScaledMulExpr<'a>;
	fn mul(self, expr: ScaledMulExpr<'a>) -> Self::Output {
		expr.scale(self)
	}
}

impl<'a> std::ops::Neg for &'a Tensor {
	type Output = ScaledTensorExpr<'a>;
	fn neg(self) -> Self::Output {
		self.scale(-1.0)
	}
}

impl<'a> std::ops::Neg for ScaledTensorExpr<'a> {
	type Output = Self;
	fn neg(self) -> Self::Output {
		self.scale(-1.0)
	}
}

impl<'a> std::ops::Neg for AddWeightedExpr<'a> {
	type Output = Self;
	fn neg(self) -> Self::Output {
		self.scale(-1.0)
	}
}

impl<'a> std::ops::Neg for DotExpr<'a> {
	type Output = Self;
	fn neg(self) -> Self::Output {
		self.scale(-1.0)
	}
}

impl<'a> std::ops::Neg for DotAddExpr<'a> {
	type Output = Self;
	fn neg(self) -> Self::Output {
		self.scale(-1.0)
	}
}

impl<'a> std::ops::Neg for MulExpr<'a> {
	type Output = ScaledMulExpr<'a>;
	fn neg(self) -> Self::Output {
		self.scale(-1.0)
	}
}

impl<'a> std::ops::Neg for ScaledMulExpr<'a> {
	type Output = Self;
	fn neg(self) -> Self::Output {
		self.scale(-1.0)
	}
}

//--------------------------------------------------------------------------------------------------
// Adding & Subtracting tensors.

pub struct AddWeightedExpr<'a> {
	pub a: ScaledTensorExpr<'a>,
	pub b: ScaledTensorExpr<'a>,
}

impl<'a> EvaluatesToTensor for AddWeightedExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();

		// If any of the inputs overlaps with output, make sure it's 'b'
		let a = self.a;
		let b = self.b;
		let a_buf = a.tensor.buf().as_ref();
		let to_buf = to.buf().as_ref();
		let (a, b) = if std::ptr::eq(to_buf, a_buf) { (b, a) } else { (a, b) };

		let ew = ElemWise::new([to], [a.tensor, b.tensor])?;
		let overlap = ew.are_identical::<0, 1>();
		if overlap {
			ew.run(|[to_tensor], [a_tensor]| {
				executor.acc_weighted(to_tensor, b.scale, a_tensor, a.scale)?;
				Ok(())
			})
		} else {
			ew.run(|[to_], [a_, b_]| {
				executor.add_weighted(to_, a_, self.a.scale, b_, self.b.scale)?;
				Ok(())
			})
		}
	}
}

impl<'a> Scalable for AddWeightedExpr<'a> {
	type Output = Self;
	fn scale(self, scale: f64) -> Self::Output {
		Self {
			a: self.a.scale(scale),
			b: self.b.scale(scale),
		}
	}
}

impl<'a> std::ops::Add<&'a Tensor> for &'a Tensor {
	type Output = AddWeightedExpr<'a>;
	fn add(self, b: &'a Tensor) -> Self::Output {
		AddWeightedExpr { a: self.into(), b: b.into() }
	}
}

impl<'a> std::ops::Add<ScaledTensorExpr<'a>> for &'a Tensor {
	type Output = AddWeightedExpr<'a>;
	fn add(self, b: ScaledTensorExpr<'a>) -> Self::Output {
		AddWeightedExpr { a: self.into(), b }
	}
}

impl<'a> std::ops::Add<&'a Tensor> for ScaledTensorExpr<'a> {
	type Output = AddWeightedExpr<'a>;
	fn add(self, b: &'a Tensor) -> Self::Output {
		AddWeightedExpr { a: self, b: b.into() }
	}
}

impl<'a> std::ops::Add<Self> for ScaledTensorExpr<'a> {
	type Output = AddWeightedExpr<'a>;
	fn add(self, b: Self) -> Self::Output {
		AddWeightedExpr { a: self, b }
	}
}

impl<'a> std::ops::Sub<&'a Tensor> for &'a Tensor {
	type Output = AddWeightedExpr<'a>;
	fn sub(self, b: &'a Tensor) -> Self::Output {
		AddWeightedExpr { a: self.into(), b: b.scale(-1.0) }
	}
}

impl<'a> std::ops::Sub<ScaledTensorExpr<'a>> for &'a Tensor {
	type Output = AddWeightedExpr<'a>;
	fn sub(self, b: ScaledTensorExpr<'a>) -> Self::Output {
		AddWeightedExpr { a: self.into(), b: b.scale(-1.0) }
	}
}

impl<'a> std::ops::Sub<&'a Tensor> for ScaledTensorExpr<'a> {
	type Output = AddWeightedExpr<'a>;
	fn sub(self, b: &'a Tensor) -> Self::Output {
		AddWeightedExpr { a: self, b: b.scale(-1.0) }
	}
}

impl<'a> std::ops::Sub<Self> for ScaledTensorExpr<'a> {
	type Output = AddWeightedExpr<'a>;
	fn sub(self, b: Self) -> Self::Output {
		AddWeightedExpr { a: self, b: b.scale(-1.0) }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct DotExpr<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
	pub scale: f64,
}

pub fn dot<'a>(a: &'a Tensor, b: &'a Tensor) -> DotExpr<'a> {
	DotExpr { a, b, scale: 1.0 }
}

impl<'a> EvaluatesToTensor for DotExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		VecWise::new([to], [self.a, self.b])?.run(|[to], [a, b]| {
			executor.dot(to, a, b, self.scale)?;
			Ok(())
		})
	}
}

impl<'a> Scalable for DotExpr<'a> {
	type Output = Self;
	fn scale(self, scale: f64) -> Self::Output {
		Self { scale: self.scale * scale, ..self }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct DotAddExpr<'a> {
	pub dot: DotExpr<'a>,
	pub add: ScaledTensorExpr<'a>,
}

impl<'a> EvaluatesToTensor for DotAddExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		VecWise::new([to], [self.add.tensor, self.dot.a, self.dot.b])?.run(|[to], [x, a, b]| {
			executor.dot_add(to, a, b, self.dot.scale, x, self.add.scale)?;
			Ok(())
		})
	}
}

impl<'a> Scalable for DotAddExpr<'a> {
	type Output = Self;
	fn scale(self, scale: f64) -> Self::Output {
		Self {
			dot: self.dot.scale(scale),
			add: self.add.scale(scale),
		}
	}
}

impl<'a> std::ops::Add<&'a Tensor> for DotExpr<'a> {
	type Output = DotAddExpr<'a>;
	fn add(self, add: &'a Tensor) -> Self::Output {
		DotAddExpr { dot: self, add: add.into() }
	}
}

impl<'a> std::ops::Add<ScaledTensorExpr<'a>> for DotExpr<'a> {
	type Output = DotAddExpr<'a>;
	fn add(self, add: ScaledTensorExpr<'a>) -> Self::Output {
		DotAddExpr { dot: self, add }
	}
}

impl<'a> std::ops::Add<DotExpr<'a>> for &'a Tensor {
	type Output = DotAddExpr<'a>;
	fn add(self, dot: DotExpr<'a>) -> Self::Output {
		DotAddExpr { dot, add: self.into() }
	}
}

impl<'a> std::ops::Add<DotExpr<'a>> for ScaledTensorExpr<'a> {
	type Output = DotAddExpr<'a>;
	fn add(self, dot: DotExpr<'a>) -> Self::Output {
		DotAddExpr { dot, add: self }
	}
}

impl<'a> std::ops::Sub<&'a Tensor> for DotExpr<'a> {
	type Output = DotAddExpr<'a>;
	fn sub(self, sub: &'a Tensor) -> Self::Output {
		DotAddExpr { dot: self, add: sub.scale(-1.0) }
	}
}

impl<'a> std::ops::Sub<ScaledTensorExpr<'a>> for DotExpr<'a> {
	type Output = DotAddExpr<'a>;
	fn sub(self, sub: ScaledTensorExpr<'a>) -> Self::Output {
		DotAddExpr { dot: self, add: sub.scale(-1.0) }
	}
}

impl<'a> std::ops::Sub<DotExpr<'a>> for &'a Tensor {
	type Output = DotAddExpr<'a>;
	fn sub(self, dot: DotExpr<'a>) -> Self::Output {
		DotAddExpr { dot: dot.scale(-1.0), add: self.into() }
	}
}

impl<'a> std::ops::Sub<DotExpr<'a>> for ScaledTensorExpr<'a> {
	type Output = DotAddExpr<'a>;
	fn sub(self, dot: DotExpr<'a>) -> Self::Output {
		DotAddExpr { dot: dot.scale(-1.0), add: self }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct MulAddExpr<'a> {
	pub mul: ScaledMulExpr<'a>,
	pub add: ScaledTensorExpr<'a>,
}

impl<'a> EvaluatesToTensor for MulAddExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		ElemWise::new([to], [self.mul.a, self.mul.b, self.add.tensor])?.run(|[to], [a, b, add]| {
			executor.mul_add(to, a, b, self.mul.scale, add, self.add.scale)?;
			Ok(())
		})
	}
}

impl<'a> Scalable for MulAddExpr<'a> {
	type Output = Self;
	fn scale(self, scale: f64) -> Self::Output {
		Self {
			mul: self.mul.scale(scale),
			add: self.add.scale(scale),
		}
	}
}

impl<'a> std::ops::Add<&'a Tensor> for ScaledMulExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn add(self, add: &'a Tensor) -> Self::Output {
		MulAddExpr { mul: self, add: add.into() }
	}
}

impl<'a> std::ops::Add<&'a Tensor> for MulExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn add(self, add: &'a Tensor) -> Self::Output {
		MulAddExpr { mul: self.scale(1.0), add: add.into() }
	}
}

impl<'a> std::ops::Add<ScaledTensorExpr<'a>> for ScaledMulExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn add(self, add: ScaledTensorExpr<'a>) -> Self::Output {
		MulAddExpr { mul: self, add }
	}
}

impl<'a> std::ops::Add<ScaledTensorExpr<'a>> for MulExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn add(self, add: ScaledTensorExpr<'a>) -> Self::Output {
		MulAddExpr { mul: self.scale(1.0), add }
	}
}

impl<'a> std::ops::Add<ScaledMulExpr<'a>> for &'a Tensor {
	type Output = MulAddExpr<'a>;
	fn add(self, mul: ScaledMulExpr<'a>) -> Self::Output {
		MulAddExpr { mul, add: self.into() }
	}
}

impl<'a> std::ops::Add<MulExpr<'a>> for &'a Tensor {
	type Output = MulAddExpr<'a>;
	fn add(self, mul: MulExpr<'a>) -> Self::Output {
		MulAddExpr { mul: mul.scale(1.0), add: self.into() }
	}
}

impl<'a> std::ops::Add<ScaledMulExpr<'a>> for ScaledTensorExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn add(self, mul: ScaledMulExpr<'a>) -> Self::Output {
		MulAddExpr { mul, add: self }
	}
}

impl<'a> std::ops::Add<MulExpr<'a>> for ScaledTensorExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn add(self, mul: MulExpr<'a>) -> Self::Output {
		MulAddExpr { mul: mul.scale(1.0), add: self }
	}
}

impl<'a> std::ops::Sub<&'a Tensor> for ScaledMulExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn sub(self, sub: &'a Tensor) -> Self::Output {
		MulAddExpr { mul: self, add: sub.scale(-1.0) }
	}
}

impl<'a> std::ops::Sub<&'a Tensor> for MulExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn sub(self, sub: &'a Tensor) -> Self::Output {
		MulAddExpr {
			mul: self.scale(1.0),
			add: sub.scale(-1.0),
		}
	}
}

impl<'a> std::ops::Sub<ScaledTensorExpr<'a>> for ScaledMulExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn sub(self, sub: ScaledTensorExpr<'a>) -> Self::Output {
		MulAddExpr { mul: self, add: sub.scale(-1.0) }
	}
}

impl<'a> std::ops::Sub<ScaledTensorExpr<'a>> for MulExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn sub(self, sub: ScaledTensorExpr<'a>) -> Self::Output {
		MulAddExpr {
			mul: self.scale(1.0),
			add: sub.scale(-1.0),
		}
	}
}

impl<'a> std::ops::Sub<ScaledMulExpr<'a>> for &'a Tensor {
	type Output = MulAddExpr<'a>;
	fn sub(self, mul: ScaledMulExpr<'a>) -> Self::Output {
		MulAddExpr { mul: mul.scale(-1.0), add: self.into() }
	}
}

impl<'a> std::ops::Sub<MulExpr<'a>> for &'a Tensor {
	type Output = MulAddExpr<'a>;
	fn sub(self, mul: MulExpr<'a>) -> Self::Output {
		MulAddExpr { mul: mul.scale(-1.0), add: self.into() }
	}
}

impl<'a> std::ops::Sub<ScaledMulExpr<'a>> for ScaledTensorExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn sub(self, mul: ScaledMulExpr<'a>) -> Self::Output {
		MulAddExpr { mul: mul.scale(-1.0), add: self }
	}
}

impl<'a> std::ops::Sub<MulExpr<'a>> for ScaledTensorExpr<'a> {
	type Output = MulAddExpr<'a>;
	fn sub(self, mul: MulExpr<'a>) -> Self::Output {
		MulAddExpr { mul: mul.scale(-1.0), add: self }
	}
}

//--------------------------------------------------------------------------------------------------

pub trait Sum {
	type Output;
	fn sum(self) -> Self::Output;
}

pub struct SumExpr<'a> {
	pub tensor: &'a Tensor,
}

impl<'a> Sum for &'a Tensor {
	type Output = SumExpr<'a>;
	fn sum(self) -> Self::Output {
		SumExpr { tensor: self }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct MulExpr<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
}

impl<'a> std::ops::Mul<&'a Tensor> for &'a Tensor {
	type Output = MulExpr<'a>;
	fn mul(self, b: &'a Tensor) -> Self::Output {
		MulExpr { a: self, b }
	}
}

impl<'a> EvaluatesToTensor for MulExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();

		// If any of the inputs overlaps with output, make sure it's 'b'
		let a = self.a;
		let b = self.b;
		let a_buf = a.buf().as_ref();
		let to_buf = to.buf().as_ref();
		let (a, b) = if std::ptr::eq(to_buf, a_buf) { (b, a) } else { (a, b) };

		let ew = ElemWise::new([to], [a, b])?;
		let overlap = ew.are_identical::<0, 1>();
		if overlap {
			ew.run(|[to_tensor], [a_tensor]| {
				executor.mul_(to_tensor, a_tensor)?;
				Ok(())
			})
		} else {
			ew.run(|[to_tensor], [a_tensor, b_tensor]| {
				executor.mul(to_tensor, a_tensor, b_tensor)?;
				Ok(())
			})
		}
	}
}

impl<'a> Sum for MulExpr<'a> {
	type Output = DotExpr<'a>;
	fn sum(self) -> Self::Output {
		DotExpr { a: self.a, b: self.b, scale: 1.0 }
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ScaledMulExpr<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
	pub scale: f64,
}

impl<'a> Scalable for MulExpr<'a> {
	type Output = ScaledMulExpr<'a>;
	fn scale(self, scale: f64) -> Self::Output {
		ScaledMulExpr { a: self.a, b: self.b, scale }
	}
}

impl<'a> Scalable for ScaledMulExpr<'a> {
	type Output = Self;
	fn scale(self, scale: f64) -> Self::Output {
		Self {
			a: self.a,
			b: self.b,
			scale: self.scale * scale,
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RSqrtExpr<'a> {
	pub tensor: &'a Tensor,
	pub scale: f64,
	pub eps: f64,
}

pub trait RSqrt {
	type Output;
	fn rsqrt(self, eps: f64) -> Self::Output;
}

impl<'a> RSqrt for &'a Tensor {
	type Output = RSqrtExpr<'a>;
	fn rsqrt(self, eps: f64) -> Self::Output {
		RSqrtExpr { tensor: self, scale: 1.0, eps }
	}
}

impl<'a> RSqrt for ScaledTensorExpr<'a> {
	type Output = RSqrtExpr<'a>;
	fn rsqrt(self, eps: f64) -> Self::Output {
		RSqrtExpr {
			tensor: self.tensor,
			scale: self.scale,
			eps,
		}
	}
}

impl<'a> EvaluatesToTensor for RSqrtExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		ElemWise::new([to], [self.tensor])?.run(|[to], [input]| {
			executor.rsqrt(to, input, self.scale, self.eps)?;
			Ok(())
		})
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RSqrtDotExpr<'a> {
	pub a: &'a Tensor,
	pub b: &'a Tensor,
	pub scale: f64,
	pub eps: f64,
}

impl<'a> RSqrt for DotExpr<'a> {
	type Output = RSqrtDotExpr<'a>;
	fn rsqrt(self, eps: f64) -> Self::Output {
		RSqrtDotExpr {
			a: self.a,
			b: self.b,
			scale: self.scale,
			eps,
		}
	}
}

impl<'a> EvaluatesToTensor for RSqrtDotExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		VecWise::new([to], [self.a, self.b])?.run(|[to], [a, b]| {
			executor.rsqrt_dot(to, a, b, self.scale, self.eps)?;
			Ok(())
		})
	}
}

//--------------------------------------------------------------------------------------------------

pub trait LnClamped {
	type Output;

	/// Calculates:
	///
	///     low_bound = max(-1000, DType.MAX_NEGATIVE);
	///     dst = max(ln(a), low_bound);
	///
	/// So the output is defined even for a <= 0.
	fn ln_clamped(self) -> Self::Output;
}

pub struct LnClampedExpr<'a> {
	pub tensor: &'a Tensor,
}

impl<'a> LnClamped for &'a Tensor {
	type Output = LnClampedExpr<'a>;
	fn ln_clamped(self) -> Self::Output {
		LnClampedExpr { tensor: self }
	}
}

impl<'a> EvaluatesToTensor for LnClampedExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		ElemWise::new([to], [self.tensor])?.run(|[to], [input]| {
			executor.ln_clamped(to, input)?;
			Ok(())
		})
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SwiGLUExpr<'a> {
	pub lin: &'a Tensor,
	pub gate: &'a Tensor,
}

pub fn swiglu<'a>(lin: &'a Tensor, gate: &'a Tensor) -> SwiGLUExpr<'a> {
	SwiGLUExpr { lin, gate }
}

impl<'a> EvaluatesToTensor for SwiGLUExpr<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		ElemWise::new([to], [self.lin, self.gate])?.run(|[to], [lin, gate]| {
			executor.swiglu(to, lin, gate)?;
			Ok(())
		})
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SwiGLUBackwardExpr<'a> {
	pub d_out: &'a Tensor,
	pub lin: &'a Tensor,
	pub gate: &'a Tensor,
}

pub fn swiglu_backward<'a>(
	d_out: &'a Tensor,
	lin: &'a Tensor,
	gate: &'a Tensor,
) -> SwiGLUBackwardExpr<'a> {
	SwiGLUBackwardExpr { d_out, lin, gate }
}

// Note: We cannot implement `EvaluatesToTensor` because there are two output tensors

impl<'a> SwiGLUBackwardExpr<'a> {
	#[inline(never)]
	pub fn eval_to_tensors(
		&self,
		d_lin: &Tensor,
		d_gate: &Tensor,
	) -> Result<(), ErrPack<TensorOpError>> {
		let executor = d_lin.executor();
		let mut ew = ElemWise::new([d_lin, d_gate], [self.lin, self.gate, self.d_out])?;

		let d_lin_buf = d_lin.buf().as_ref();
		let d_gate_buf = d_gate.buf().as_ref();
		if std::ptr::eq(d_lin_buf, d_gate_buf) {
			let size = ew.dims[0].size;
			let swapped = d_lin.map().offset > d_gate.map().offset;
			if swapped {
				ew.m = [d_gate, d_lin];
			}
			let shift = ew.m[1].map().offset - ew.m[0].map().offset;

			let min_stride1 = ew.dims[0].size;
			let min_stride2 = ew.dims[1].size * ew.dims[1].strides[0];

			if shift >= size
				&& ew.dims[0].strides[0] == 1
				&& ew.dims[0].strides[0] == ew.dims[0].strides[1]
				&& ew.dims[1].strides[0] == ew.dims[1].strides[1]
				&& ew.dims[2].strides[0] == ew.dims[2].strides[1]
				&& (ew.dims[1].size <= 1
					|| (ew.dims[1].strides[0] >= min_stride1
						&& (ew.dims[2].size <= 1 || ew.dims[2].strides[0] >= min_stride2)))
			{
				ew.run(|[d_lin_gate], [lin, gate, d_out]| {
					unsafe { d_lin_gate.map_mut().dims[1].size = size + shift };
					executor.swiglu_backward2(d_lin_gate, size, swapped, lin, gate, d_out)?;
					Ok(())
				})
			} else {
				cold_path();
				Err(ErrPack {
					code: TensorOpError::CannotBorrowMut,
					extra: None,
				})
			}
		} else {
			ew.run(|[d_lin, d_gate], [lin, gate, d_out]| {
				executor.swiglu_backward(d_lin, d_gate, lin, gate, d_out)?;
				Ok(())
			})
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub fn sum_all(tensor: &Tensor) -> Result<f64, ErrPack<TensorOpError>> {
	let executor = tensor.executor();
	let mut sum = 0.0;
	// TODO - `__elem_wise()` disables broadcast for tensor at position 0.
	// In the case of a `sum_all()`, it would make sense to enable it,
	// but it would require some refactoring. Not sure if it is worth it.
	ElemWise::new([], [tensor])?.run(|[], [a]| {
		sum += executor.sum_all(a)?;
		Ok(())
	})?;
	Ok(sum)
}

pub fn approx_eq(a: &Tensor, b: &Tensor, eps: f64) -> Result<bool, ErrPack<TensorOpError>> {
	let executor = a.executor();
	let mut result = true;
	ElemWise::new([], [a, b])?.run(|[], [a, b]| {
		result &= executor.approx_eq(a, b, eps)?;
		Ok(())
	})?;
	Ok(result)
}

//--------------------------------------------------------------------------------------------------

pub struct Softmax<'a> {
	pub tensor: &'a Tensor,
}

pub fn softmax<'a>(tensor: &'a Tensor) -> Softmax<'a> {
	Softmax { tensor }
}

impl<'a> EvaluatesToTensor for Softmax<'a> {
	#[inline(never)]
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let executor = to.executor();
		let vw = VecWise::new([to], [self.tensor])?;
		let overlap = vw.are_identical::<0, 0>();
		if overlap {
			vw.run(|[to], []| {
				executor.softmax_(to)?;
				Ok(())
			})
		} else {
			vw.run(|[to], [input]| {
				executor.softmax(to, input)?;
				Ok(())
			})
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Matrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],
	pub rows: SizeAndStride,
	pub cols: SizeAndStride,
}

impl<'a> Matrix<'a> {
	pub fn T(self) -> Self {
		Matrix { rows: self.cols, cols: self.rows, ..self }
	}

	pub fn assign<Expr: EvaluatesToMatrix>(
		&self,
		expr: Expr,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.eval_to_matrix(self)
	}
}

pub fn matrix<'a>(tensor: &'a Tensor) -> Result<Matrix<'a>, NotEnoughDimensionsError> {
	let dims = tensor.map().dims.as_slice();
	if dims.len() < 2 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		Ok(Matrix {
			tensor,
			batch_dims: &dims[..dims.len() - 2],
			rows: dims[dims.len() - 2],
			cols: dims[dims.len() - 1],
		})
	}
}

#[derive(Clone, Copy)]
pub struct RowMatrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],
	pub cols: SizeAndStride,
}

impl<'a> RowMatrix<'a> {
	pub fn T(self) -> ColMatrix<'a> {
		ColMatrix {
			tensor: self.tensor,
			batch_dims: self.batch_dims,
			rows: self.cols,
		}
	}

	pub fn assign<Expr: EvaluatesToRowMatrix>(
		&self,
		expr: Expr,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.eval_to_row_matrix(self)
	}
}

pub fn row_matrix<'a>(tensor: &'a Tensor) -> Result<RowMatrix<'a>, NotEnoughDimensionsError> {
	let dims = tensor.map().dims.as_slice();
	#[allow(clippy::len_zero)]
	if dims.len() < 1 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		Ok(RowMatrix {
			tensor,
			batch_dims: &dims[..dims.len() - 1],
			cols: dims[dims.len() - 1],
		})
	}
}

#[derive(Clone, Copy)]
pub struct ColMatrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],
	pub rows: SizeAndStride,
}

impl<'a> ColMatrix<'a> {
	pub fn T(self) -> RowMatrix<'a> {
		RowMatrix {
			tensor: self.tensor,
			batch_dims: self.batch_dims,
			cols: self.rows,
		}
	}

	pub fn assign<Expr: EvaluatesToColMatrix>(
		&self,
		expr: Expr,
	) -> Result<(), ErrPack<TensorOpError>> {
		expr.eval_to_col_matrix(self)
	}
}

pub fn column_matrix<'a>(tensor: &'a Tensor) -> Result<ColMatrix<'a>, NotEnoughDimensionsError> {
	let dims = tensor.map().dims.as_slice();
	#[allow(clippy::len_zero)]
	if dims.len() < 1 {
		cold_path();
		Err(NotEnoughDimensionsError)
	} else {
		Ok(ColMatrix {
			tensor,
			batch_dims: &dims[..dims.len() - 1],
			rows: dims[dims.len() - 1],
		})
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ColTimesRow<'a> {
	pub col: ColMatrix<'a>,
	pub row: RowMatrix<'a>,
	pub scale: f64,
}

pub struct MatTimesCol<'a> {
	pub mat: Matrix<'a>,
	pub col: ColMatrix<'a>,
	pub scale: f64,
}

impl<'a> std::ops::Mul<RowMatrix<'a>> for ColMatrix<'a> {
	type Output = ColTimesRow<'a>;

	fn mul(self, row: RowMatrix<'a>) -> Self::Output {
		ColTimesRow { col: self, row, scale: 1.0 }
	}
}

impl<'a> std::ops::Mul<ColMatrix<'a>> for Matrix<'a> {
	type Output = MatTimesCol<'a>;

	fn mul(self, col: ColMatrix<'a>) -> Self::Output {
		MatTimesCol { mat: self, col, scale: 1.0 }
	}
}
/*
impl<'a> EvaluatesToMatrix for ColTimesRow<'a> {
	#[inline(never)]
	fn eval_to_matrix(self, to: &Matrix) -> Result<(), ErrPack<TensorOpError>> {
		// TODO
		1
	}
}
*/

/*
impl<'a> EvaluatesToColMatrix for MatTimesCol<'a> {
	#[inline(never)]
	fn eval_to_col_matrix(self, to: &ColMatrix) -> Result<(), ErrPack<TensorOpError>> {
		unsafe {
			let Self { mat, col, scale } = self;
			let mut c_fail = 0;
			let mat_buf = DeviceBufferRef::new_unsafe(mat.tensor.buf().as_ref(), &mut c_fail);
			let col_buf = DeviceBufferRef::new_unsafe(col.tensor.buf().as_ref(), &mut c_fail);
			let mut fail = c_fail;
			let to_buf = DeviceBufferRefMut::new_unsafe(to.tensor.buf().as_ref(), &mut fail);
			check_borrows(c_fail, fail)?;

			if mat.batch_dims.is_empty() {
				// TODO - optimize
				let dims = DimMerger::merge::<3>([to.batch_dims, col.batch_dims]);
				let mat = generic::Tensor::new_unchecked(
					ND {
						dims: [SizeAndStride { size: 1, stride: 0 }, mat.rows, mat.cols],
						offset: mat.tensor.map().offset,
					},
					mat_buf,
				);
				let col = generic::Tensor::new_unchecked(
					ND {
						dims: [
							SizeAndStride
							col.rows
							],
						offset: col.tensor.map().offset,
					},
					col_buf,
				);
			} else {
				// no optimizaiton
			}
		}
		1
	}
}
*/

//--------------------------------------------------------------------------------------------------

/*
struct MatMulPrep<'a> {
	to: Matrix<'a>,
	a: Matrix<'a>,
	b: Matrix<'a>,
	batch_dims: MergedDimList<3>,
}

impl<'a> MatMulPrep<'a> {
	fn new(mm: MatMul<'a>, mut to: Matrix<'a>) -> MatMulPrep<'a> {
		let mut a = mm.a;
		let mut b = mm.b;

		const TO: usize = 0;
		const A: usize = 1;
		const B: usize = 2;
		// TODO - the output tensor could be broadcasted
		let merger = DimMerger::new([to.batch_dims, a.batch_dims, b.batch_dims]);

		// Is this actually a vector dot product?
		if to.rows.get() == 1 && to.cols.get() == 1 {
			todo!("implement vector dot product");
		}

		if to.rows.get() == 1 {
			// `to` has single row, so `a` also has to have just one row
			assert!(a.rows.get() == 1);

			// Do we have a batch of matrix multiplications
			// where the matrix `b` is the same for all items in the batch?
			let batch_dim = merger.smallest_dim();
			if batch_dim.size > 1 && batch_dim.strides[B] == 0 {
				a.rows = NonZeroUsize::new(batch_dim.size).unwrap();
				a.row_stride = batch_dim.strides[A];

				to.rows = NonZeroUsize::new(batch_dim.size).unwrap();
				to.row_stride = batch_dim.strides[TO];

				return MatMulPrep {
					to,
					a,
					b,
					batch_dims: merger.dims_increasing_without_smallest(),
				};
			}
		} else if to.cols.get() == 1 {
			// `to` has single column, so `b` also has to have just one column
			assert!(b.cols.get() == 1);

			// Do we have a batch of matrix multiplications
			// where the matrix `a` is the same for all items in the batch?
			let batch_dim = merger.smallest_dim();
			if batch_dim.size > 1 && batch_dim.strides[A] == 0 {
				b.cols = NonZeroUsize::new(batch_dim.size).unwrap();
				b.col_stride = batch_dim.strides[B];

				to.cols = NonZeroUsize::new(batch_dim.size).unwrap();
				to.col_stride = batch_dim.strides[TO];

				return MatMulPrep {
					to,
					a,
					b,
					batch_dims: merger.dims_increasing_without_smallest(),
				};
			}
		}

		MatMulPrep {
			to,
			a,
			b,
			batch_dims: merger.dims_increasing(),
		}
	}
}

impl<'a> MatrixSavable for MatMul<'a> {
	#[inline(never)]
	fn save_to(self, to: Matrix) {
		let executor = to.tensor.buffer.executor();
		let scale = self.scale;
		let prep = MatMulPrep::new(self, to);
		__mat_wise([&prep.to, &prep.a, &prep.b], prep.batch_dims.iter(), |[to, a, b]| {
			executor.gemm(&to, 0.0, &a, &b, scale);
		});
	}
}

impl<'a> MatrixAccumulable for MatMul<'a> {
	#[inline(never)]
	fn acc_to(self, to: Matrix, to_weight: f64, expr_weight: f64) {
		let executor = to.tensor.buffer.executor();
		let scale = self.scale * expr_weight;
		let prep = MatMulPrep::new(self, to);
		__mat_wise([&prep.to, &prep.a, &prep.b], prep.batch_dims.iter(), |[to, a, b]| {
			executor.gemm(&to, to_weight, &a, &b, scale);
		});
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Attention<'a> {
	pub q: &'a Tensor,
	pub k: &'a Tensor,
	pub v: &'a Tensor,
}

/// Requirements:
///
///    q.shape = [..., inputs, q_heads, qk_features]
///    k.shape = [..., inputs, k_heads, qk_features]
///    v.shape = [..., inputs, v_heads, v_features]
///
///    q_heads >= k_heads && q_heads % k_heads == 0 && (q_heads / k_heads).is_power_of_two()
///    q_heads >= v_heads && q_heads % v_heads == 0 && (q_heads / v_heads).is_power_of_two()
///
/// The output shape is:
///
///    [..., inputs, q_heads, v_features]
pub fn attention<'a>(q: &'a Tensor, k: &'a Tensor, v: &'a Tensor) -> Attention<'a> {
	Attention { q, k, v }
}

impl<'a> EvaluatesToTensor for Attention<'a> {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let tensors = [self.q, self.k, self.v, to];
		assert!(tensors.iter().all(|t| t.ndim() >= 3));

		let q_input_dim = self.q.dim(-3);
		let q_head_dim = self.q.dim(-2);
		let q_feature_dim = self.q.dim(-1);

		let k_input_dim = self.k.dim(-3);
		let k_head_dim = self.k.dim(-2);
		let k_feature_dim = self.k.dim(-1);

		let v_input_dim = self.v.dim(-3);
		let v_head_dim = self.v.dim(-2);
		let v_feature_dim = self.v.dim(-1);

		let to_output_dim = to.dim(-3);
		let to_head_dim = to.dim(-2);
		let to_feature_dim = to.dim(-1);

		let qk_features = q_feature_dim.size;
		assert!(q_feature_dim.is_contiguous());
		assert!(k_feature_dim.size == qk_features);
		assert!(k_feature_dim.is_contiguous());

		let v_features = v_feature_dim.size;
		assert!(v_feature_dim.is_contiguous());

		let q_heads = q_head_dim.size;
		assert!(
			q_head_dim.stride == qk_features,
			"TODO: If this were useful, we'd need to capture the stride in AttentionParams"
		);

		let k_heads = k_head_dim.size;
		assert!(
			k_head_dim.stride == qk_features,
			"TODO: If this were useful, we'd need to capture the stride in AttentionParams"
		);
		assert!(q_heads >= k_heads);
		assert!(q_heads % k_heads == 0);
		assert!((q_heads / k_heads).is_power_of_two());
		let k_shift = (q_heads / k_heads).trailing_zeros() as usize;

		let v_heads = v_head_dim.size;
		assert!(
			v_head_dim.stride == v_features,
			"TODO: If this were useful, we'd need to capture the stride in AttentionParams"
		);
		assert!(q_heads >= v_heads);
		assert!(q_heads % v_heads == 0);
		assert!((q_heads / v_heads).is_power_of_two());
		let v_shift = (q_heads / v_heads).trailing_zeros() as usize;

		let inputs = q_input_dim.size;
		assert!(k_input_dim.size == inputs);
		assert!(v_input_dim.size == inputs);

		assert!(to_output_dim.size == inputs);
		assert!(to_head_dim.size == q_heads);
		assert!(to_feature_dim.size == v_features);
		assert!(to_feature_dim.is_contiguous());
		assert!(
			to_head_dim.stride == v_features,
			"TODO: If this were useful, we'd need to capture the stride in AttentionParams"
		);

		let q_input_stride = q_input_dim.stride;
		let k_input_stride = k_input_dim.stride;
		let v_input_stride = v_input_dim.stride;
		let to_output_stride = to_output_dim.stride;

		let params = AttentionParams {
			heads: q_heads,
			qk_features,
			v_features,
			k_shift,
			v_shift,
		};

		let merger = DimMerger::new(tensors.map(|t| t.dim_slice(..t.ndim() - 3)));
		let batch_dims = merger.dims_increasing();
		let batch_iter = batch_dims.iter();

		let mut q = SliceSet {
			buffer: self.q.buffer.as_ref(),
			dtype: self.q.dtype(),
			offset: 0,
			len: q_heads * qk_features,
			count: inputs,
			stride: q_input_stride,
		};

		let mut k = SliceSet {
			buffer: self.k.buffer.as_ref(),
			dtype: self.k.dtype(),
			offset: 0,
			len: k_heads * qk_features,
			count: inputs,
			stride: k_input_stride,
		};

		let mut v = SliceSet {
			buffer: self.v.buffer.as_ref(),
			dtype: self.v.dtype(),
			offset: 0,
			len: v_heads * v_features,
			count: inputs,
			stride: v_input_stride,
		};

		let mut to = SliceSet {
			buffer: to.buffer.as_ref(),
			dtype: to.dtype(),
			offset: 0,
			len: q_heads * v_features,
			count: inputs,
			stride: to_output_stride,
		};

		let executor = to.buffer.executor();

		batch::run(
			batch_iter,
			tensors.map(|t| t.offset),
			|batch_size: usize, batch_strides: [usize; 4], offsets: [usize; 4]| {
				for i in 0..batch_size {
					q.offset = offsets[0] + i * batch_strides[0];
					k.offset = offsets[1] + i * batch_strides[1];
					v.offset = offsets[2] + i * batch_strides[2];
					to.offset = offsets[3] + i * batch_strides[3];
					executor.attention(&to, &q, &k, &v, &params);
				}
			},
		);
	}
}

*/
//--------------------------------------------------------------------------------------------------
