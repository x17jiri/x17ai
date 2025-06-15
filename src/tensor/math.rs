//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::Result;
use crate::tensor::device::buffer::{DeviceBufferRef, DeviceBufferRefMut};
use crate::tensor::dim_merger::DimMerger;
use crate::tensor::generic::map::{ND, SizeAndStride};
use crate::tensor::{Tensor, generic};
use crate::util::array;

pub trait EvaluatesToTensor {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` tensor.
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()>;
}

/*
pub trait MatrixSavable {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` matrix.
	fn eval_to_matrix(self, to: Matrix) -> Result<()>;
}
*/

//--------------------------------------------------------------------------------------------------

pub(crate) fn __elem_wise<const O: usize, const C: usize>(
	o: [&Tensor; O],
	c: [&Tensor; C],
	mut f: impl FnMut(
		&[generic::Tensor<ND<2>, DeviceBufferRefMut>; O],
		&[generic::Tensor<ND<2>, DeviceBufferRef>; C],
	) -> Result<()>,
) -> Result<()>
where
	[(); O + C]:,
{
	let o_dims = o.map(|t| t.map.dims.as_slice());
	let c_dims = c.map(|t| t.map.dims.as_slice());
	let merger = DimMerger::new(array::concat_arrays(o_dims, c_dims))?;
	let (dims, rest) = merger.split::<3>();
	assert!(rest.is_empty());

	let mut o_tensors = array::try_map(&o, |i, o| {
		o.buf.try_borrow_mut().map(|buf| generic::Tensor {
			map: ND {
				dims: [
					SizeAndStride {
						size: dims[1].size,
						stride: dims[1].strides[i],
					},
					SizeAndStride {
						size: dims[0].size,
						stride: dims[0].strides[i],
					},
				],
				offset: o.map.offset,
			},
			buf,
		})
	})?;
	let mut c_tensors = array::try_map(&c, |i, c| {
		c.buf.try_borrow().map(|buf| generic::Tensor {
			map: ND {
				dims: [
					SizeAndStride {
						size: dims[1].size,
						stride: dims[1].strides[O + i],
					},
					SizeAndStride {
						size: dims[0].size,
						stride: dims[0].strides[O + i],
					},
				],
				offset: c.map.offset,
			},
			buf,
		})
	})?;

	for _ in 0..dims[2].size {
		f(&o_tensors, &c_tensors)?;

		for j in 0..O {
			o_tensors[j].map.offset += dims[2].strides[j];
		}
		for j in 0..C {
			c_tensors[j].map.offset += dims[2].strides[O + j];
		}
	}
	Ok(())
}

/// Data dimension broadcast is disabled for all tensors.
/// This could be improved.
///
/// Batch dimensions broadcast is disabled for tensors[0] and enabled for tensors[1..].
/// This is by design.
fn __vec_wise<const O: usize, const C: usize>(
	o: [&Tensor; O],
	c: [&Tensor; C],
	f: impl Fn(
		&[generic::Tensor<ND<2>, DeviceBufferRefMut>; O],
		&[generic::Tensor<ND<2>, DeviceBufferRef>; C],
	) -> Result<()>,
) -> Result<()>
where
	[(); O + C]:,
{
	assert!(o.iter().all(|t| t.ndim() >= 1));
	assert!(c.iter().all(|t| t.ndim() >= 1));
	let o_dims = o.map(|t| t.map.dims.as_slice());
	let c_dims = c.map(|t| t.map.dims.as_slice());
	let o_vec = o_dims.map(|d| *d.last().unwrap());
	let c_vec = c_dims.map(|d| *d.last().unwrap());

	let o_dims = o_dims.map(|d| &d[..d.len() - 1]);
	let c_dims = c_dims.map(|d| &d[..d.len() - 1]);
	let merger = DimMerger::new(array::concat_arrays(o_dims, c_dims))?;
	let (dims, rest) = merger.split::<2>();
	assert!(rest.is_empty());

	let mut o_tensors = array::try_map(&o, |i, o| {
		o.buf.try_borrow_mut().map(|buf| generic::Tensor {
			map: ND {
				dims: [
					SizeAndStride {
						size: dims[0].size,
						stride: dims[0].strides[i],
					},
					SizeAndStride {
						size: o_vec[i].size,
						stride: o_vec[i].stride,
					},
				],
				offset: o.map.offset,
			},
			buf,
		})
	})?;
	let mut c_tensors = array::try_map(&c, |i, c| {
		c.buf.try_borrow().map(|buf| generic::Tensor {
			map: ND {
				dims: [
					SizeAndStride {
						size: dims[0].size,
						stride: dims[0].strides[O + i],
					},
					SizeAndStride {
						size: c_vec[i].size,
						stride: c_vec[i].stride,
					},
				],
				offset: c.map.offset,
			},
			buf,
		})
	})?;

	for _ in 0..dims[1].size {
		f(&o_tensors, &c_tensors)?;

		for j in 0..O {
			o_tensors[j].map.offset += dims[1].strides[j];
		}
		for j in 0..C {
			c_tensors[j].map.offset += dims[1].strides[O + j];
		}
	}
	Ok(())
}

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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [], |[to], []| executor.zeros(&to))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RandnClampedExpr();

pub fn randn_clamped() -> RandnClampedExpr {
	RandnClampedExpr()
}

impl EvaluatesToTensor for RandnClampedExpr {
	#[inline(never)]
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [], |[to], []| executor.randn_clamped(&to))
	}
}

//--------------------------------------------------------------------------------------------------

impl EvaluatesToTensor for Tensor {
	#[inline(never)]
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [self], |[to], [input]| executor.copy(&to, &input))
	}
}

//--------------------------------------------------------------------------------------------------
// Scaling, i.e., multyplication by a scalar.

pub trait Scalable {
	type Output;
	fn scale(self, scale: f64) -> Self::Output;
}

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

impl EvaluatesToTensor for AddWeightedExpr<'_> {
	#[inline(never)]
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [self.a.tensor, self.b.tensor], |[to], [a, b]| {
			executor.add_weighted(&to, &a, self.a.scale, &b, self.b.scale)
		})
	}
}

impl Scalable for AddWeightedExpr<'_> {
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

impl<'a> std::ops::Add<ScaledTensorExpr<'a>> for ScaledTensorExpr<'a> {
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

impl<'a> std::ops::Sub<ScaledTensorExpr<'a>> for ScaledTensorExpr<'a> {
	type Output = AddWeightedExpr<'a>;
	fn sub(self, b: ScaledTensorExpr<'a>) -> Self::Output {
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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__vec_wise([to], [self.a, self.b], |[to], [a, b]| executor.dot(&to, &a, &b, self.scale))
	}
}

impl Scalable for DotExpr<'_> {
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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__vec_wise([to], [self.add.tensor, self.dot.a, self.dot.b], |[to], [x, a, b]| {
			executor.dot_add(&to, &a, &b, self.dot.scale, &x, self.add.scale)
		})
	}
}

impl Scalable for DotAddExpr<'_> {
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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [self.mul.a, self.mul.b, self.add.tensor], |[to], [a, b, add]| {
			executor.mul_add(&to, &a, &b, self.mul.scale, &add, self.add.scale)
		})
	}
}

impl Scalable for MulAddExpr<'_> {
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

impl EvaluatesToTensor for MulExpr<'_> {
	#[inline(never)]
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [self.a, self.b], |[to], [a, b]| executor.mul(&to, &a, &b))
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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [self.tensor], |[to], [input]| {
			executor.rsqrt(&to, &input, self.scale, self.eps)
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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__vec_wise([to], [self.a, self.b], |[to], [a, b]| {
			executor.rsqrt_dot(&to, &a, &b, self.scale, self.eps)
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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [self.tensor], |[to], [input]| executor.ln_clamped(&to, &input))
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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__elem_wise([to], [self.lin, self.gate], |[to], [lin, gate]| {
			executor.swiglu(&to, &lin, &gate)
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
	pub fn eval_to_tensors(&self, d_lin: &Tensor, d_gate: &Tensor) -> Result<()> {
		let executor = d_lin.executor();
		__elem_wise(
			[d_lin, d_gate],
			[self.lin, self.gate, self.d_out],
			|[d_lin, d_gate], [lin, gate, d_out]| {
				executor.swiglu_backward(&d_lin, &d_gate, &lin, &gate, &d_out)
			},
		)
	}
}

//--------------------------------------------------------------------------------------------------

pub fn sum_all(tensor: &Tensor) -> Result<f64> {
	let executor = tensor.executor();
	let mut sum = 0.0;
	// TODO - `__elem_wise()` disables broadcast for tensor at position 0.
	// In the case of a `sum_all()`, it would make sense to enable it,
	// but it would require some refactoring. Not sure if it is worth it.
	__elem_wise([], [tensor], |[], [a]| {
		sum += executor.sum_all(&a)?;
		Ok(())
	})?;
	Ok(sum)
}

pub fn approx_eq(a: &Tensor, b: &Tensor, eps: f64) -> Result<bool> {
	let executor = a.executor();
	let mut result = true;
	__elem_wise([], [a, b], |[], [a, b]| {
		result &= executor.approx_eq(&a, &b, eps)?;
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
	fn eval_to_tensor(&self, to: &Tensor) -> Result<()> {
		let executor = to.executor();
		__vec_wise([to], [self.tensor], |[to], [input]| executor.softmax(&to, &input))
	}
}

//--------------------------------------------------------------------------------------------------

/*
pub struct RMSNorm<'a> {
	pub tensor: &'a Tensor,
	pub eps: f64,
	pub scale_storage: Option<&'a Tensor>,
}

pub fn rms_norm<'a>(tensor: &'a Tensor, eps: f64) -> RMSNorm<'a> {
	RMSNorm { tensor, eps, scale_storage: None }
}

impl<'a> RMSNorm<'a> {
	pub fn scale_storage(self, scale_storage: &'a Tensor) -> RMSNorm<'a> {
		RMSNorm {
			scale_storage: Some(scale_storage),
			..self
		}
	}
}

impl<'a> EvaluatesToTensor for RMSNorm<'a> {
	#[inline(never)]
	fn save_to(&self, to: &Tensor) {
		let executor = to.buffer.executor();
		if let Some(scale_storage) = self.scale_storage {
			// TODO - could this broadcast the `scale_storage` tensor?
			__vec_wise([to, self.tensor, scale_storage], |[to, input, scale_storage]| {
				executor.rms_norm(&to, &input, self.eps, Some(&scale_storage));
			});
		} else {
			__vec_wise([to, self.tensor], |[to, input]| {
				executor.rms_norm(&to, &input, self.eps, None);
			});
		}
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Matrix<'a> {
	pub tensor: &'a Tensor,
	pub batch_dims: &'a [SizeAndStride],

	pub rows: NonZeroUsize,
	pub cols: NonZeroUsize,
	pub row_stride: usize,
	pub col_stride: usize,
}

impl<'a> Matrix<'a> {
	pub fn T(self) -> Matrix<'a> {
		Matrix {
			rows: self.cols,
			cols: self.rows,
			row_stride: self.col_stride,
			col_stride: self.row_stride,
			..self
		}
	}
}

pub fn matrix<'a>(tensor: &'a Tensor) -> Matrix<'a> {
	assert!(tensor.ndim() >= 2);
	let rows = tensor.dim(-2);
	let cols = tensor.dim(-1);
	Matrix {
		tensor,
		batch_dims: tensor.dim_slice(..tensor.ndim() - 2),
		rows: NonZeroUsize::new(rows.size).unwrap(),
		cols: NonZeroUsize::new(cols.size).unwrap(),
		row_stride: rows.stride,
		col_stride: cols.stride,
	}
}

pub fn row_matrix<'a>(tensor: &'a Tensor) -> Matrix<'a> {
	assert!(tensor.ndim() >= 1);
	let rows = SizeAndStride { size: 1, stride: 0 };
	let cols = tensor.dim(-1);
	Matrix {
		tensor,
		batch_dims: tensor.dim_slice(..tensor.ndim() - 1),
		rows: NonZeroUsize::new(rows.size).unwrap(),
		cols: NonZeroUsize::new(cols.size).unwrap(),
		row_stride: rows.stride,
		col_stride: cols.stride,
	}
}

pub fn col_matrix<'a>(tensor: &'a Tensor) -> Matrix<'a> {
	assert!(tensor.ndim() >= 1);
	let rows = tensor.dim(-1);
	let cols = SizeAndStride { size: 1, stride: 0 };
	Matrix {
		tensor,
		batch_dims: tensor.dim_slice(..tensor.ndim() - 1),
		rows: NonZeroUsize::new(rows.size).unwrap(),
		cols: NonZeroUsize::new(cols.size).unwrap(),
		row_stride: rows.stride,
		col_stride: cols.stride,
	}
}

//--------------------------------------------------------------------------------------------------

pub struct MatMul<'a> {
	pub a: Matrix<'a>,
	pub b: Matrix<'a>,
	pub scale: f64,
}

pub fn mm<'a>(a: Matrix<'a>, b: Matrix<'a>) -> MatMul<'a> {
	MatMul { a, b, scale: 1.0 }
}

impl<'a> MatMul<'a> {
	pub fn scale(self, scale: f64) -> MatMul<'a> {
		MatMul { scale: self.scale * scale, ..self }
	}
}

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

//--------------------------------------------------------------------------------------------------
*/
