//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::math::EvaluatesToTensor;
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

use super::library::KernelLibrary;

//--------------------------------------------------------------------------------------------------

pub trait LookupExpr {
	fn last_dim_size(&self) -> usize;
}

pub trait KernelCall<Expr: LookupExpr> {
	fn call(&self, to: &Tensor, expr: LookupWrapper<Expr>) -> Result<(), ErrPack<TensorOpError>>;
}

//--------------------------------------------------------------------------------------------------

pub struct LookupWrapper<Expr: LookupExpr>(pub Expr);

impl<Expr: LookupExpr> LookupWrapper<Expr> {
	pub fn sum(self) -> LookupWrapper<SumLookupExpr<Expr>> {
		LookupWrapper(SumLookupExpr(self.0))
	}

	pub fn mean(self) -> LookupWrapper<MulLookupExpr<SumLookupExpr<Expr>, f64>> {
		let n = self.0.last_dim_size();
		let sum_to_mean = 1.0 / n.lossy_into();
		LookupWrapper(MulLookupExpr(SumLookupExpr(self.0), sum_to_mean))
	}

	pub fn sigmoid(self) -> LookupWrapper<SigmoidLookupExpr<Expr>> {
		LookupWrapper(SigmoidLookupExpr(self.0))
	}

	pub fn swish(self) -> LookupWrapper<SwishLookupExpr<Expr>> {
		LookupWrapper(SwishLookupExpr(self.0))
	}

	pub fn sqrt(self) -> LookupWrapper<SqrtLookupExpr<Expr>> {
		LookupWrapper(SqrtLookupExpr(self.0))
	}

	pub fn recip<E: LookupExpr>(
		self,
		eps: LookupWrapper<E>,
	) -> LookupWrapper<RecipLookupExpr<Expr, E>> {
		LookupWrapper(RecipLookupExpr(self.0, eps.0))
	}

	pub fn ln_clamped(self) -> LookupWrapper<LnClampedLookupExpr<Expr>> {
		LookupWrapper(LnClampedLookupExpr(self.0))
	}
}

//--------------------------------------------------------------------------------------------------

impl<Expr: LookupExpr> EvaluatesToTensor for LookupWrapper<Expr>
where
	KernelLibrary: KernelCall<Expr>,
{
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		let library = to.buf().builtin_kernels;
		library.call(to, self)
	}
}

//--------------------------------------------------------------------------------------------------

impl LookupExpr for &Tensor {
	fn last_dim_size(&self) -> usize {
		self.size(-1).unwrap_or(1)
	}
}

pub fn tsr(tensor: &Tensor) -> LookupWrapper<&Tensor> {
	LookupWrapper(tensor)
}

//--------------------------------------------------------------------------------------------------

impl LookupExpr for f64 {
	fn last_dim_size(&self) -> usize {
		1
	}
}

pub fn scalar(value: f64) -> LookupWrapper<f64> {
	LookupWrapper(value)
}

pub fn zero() -> LookupWrapper<f64> {
	scalar(0.0)
}

//--------------------------------------------------------------------------------------------------

pub struct AddLookupExpr<A: LookupExpr, B: LookupExpr>(pub A, pub B);

impl<A: LookupExpr, B: LookupExpr> LookupExpr for AddLookupExpr<A, B> {
	fn last_dim_size(&self) -> usize {
		self.0.last_dim_size().max(self.1.last_dim_size())
	}
}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Add<LookupWrapper<B>> for LookupWrapper<A> {
	type Output = LookupWrapper<AddLookupExpr<A, B>>;

	fn add(self, rhs: LookupWrapper<B>) -> LookupWrapper<AddLookupExpr<A, B>> {
		let LookupWrapper(lhs) = self;
		let LookupWrapper(rhs) = rhs;
		LookupWrapper(AddLookupExpr(lhs, rhs))
	}
}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Add<B> for LookupWrapper<A> {
	type Output = LookupWrapper<AddLookupExpr<A, B>>;

	fn add(self, rhs: B) -> LookupWrapper<AddLookupExpr<A, B>> {
		let LookupWrapper(lhs) = self;
		LookupWrapper(AddLookupExpr(lhs, rhs))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SubLookupExpr<A: LookupExpr, B: LookupExpr>(pub A, pub B);

impl<A: LookupExpr, B: LookupExpr> LookupExpr for SubLookupExpr<A, B> {
	fn last_dim_size(&self) -> usize {
		self.0.last_dim_size().max(self.1.last_dim_size())
	}
}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Sub<LookupWrapper<B>> for LookupWrapper<A> {
	type Output = LookupWrapper<SubLookupExpr<A, B>>;

	fn sub(self, rhs: LookupWrapper<B>) -> LookupWrapper<SubLookupExpr<A, B>> {
		let LookupWrapper(lhs) = self;
		let LookupWrapper(rhs) = rhs;
		LookupWrapper(SubLookupExpr(lhs, rhs))
	}
}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Sub<B> for LookupWrapper<A> {
	type Output = LookupWrapper<SubLookupExpr<A, B>>;

	fn sub(self, rhs: B) -> LookupWrapper<SubLookupExpr<A, B>> {
		let LookupWrapper(lhs) = self;
		LookupWrapper(SubLookupExpr(lhs, rhs))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct MulLookupExpr<A: LookupExpr, B: LookupExpr>(pub A, pub B);

impl<A: LookupExpr, B: LookupExpr> LookupExpr for MulLookupExpr<A, B> {
	fn last_dim_size(&self) -> usize {
		self.0.last_dim_size().max(self.1.last_dim_size())
	}
}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Mul<LookupWrapper<B>> for LookupWrapper<A> {
	type Output = LookupWrapper<MulLookupExpr<A, B>>;

	fn mul(self, rhs: LookupWrapper<B>) -> LookupWrapper<MulLookupExpr<A, B>> {
		let LookupWrapper(lhs) = self;
		let LookupWrapper(rhs) = rhs;
		LookupWrapper(MulLookupExpr(lhs, rhs))
	}
}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Mul<B> for LookupWrapper<A> {
	type Output = LookupWrapper<MulLookupExpr<A, B>>;

	fn mul(self, rhs: B) -> LookupWrapper<MulLookupExpr<A, B>> {
		let LookupWrapper(lhs) = self;
		LookupWrapper(MulLookupExpr(lhs, rhs))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SumLookupExpr<A: LookupExpr>(pub A);

impl<A: LookupExpr> LookupExpr for SumLookupExpr<A> {
	fn last_dim_size(&self) -> usize {
		1
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SigmoidLookupExpr<A: LookupExpr>(pub A);

impl<A: LookupExpr> LookupExpr for SigmoidLookupExpr<A> {
	fn last_dim_size(&self) -> usize {
		self.0.last_dim_size()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SwishLookupExpr<A: LookupExpr>(pub A);

impl<A: LookupExpr> LookupExpr for SwishLookupExpr<A> {
	fn last_dim_size(&self) -> usize {
		self.0.last_dim_size()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SqrtLookupExpr<A: LookupExpr>(pub A);

impl<A: LookupExpr> LookupExpr for SqrtLookupExpr<A> {
	fn last_dim_size(&self) -> usize {
		self.0.last_dim_size()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RecipLookupExpr<A: LookupExpr, E: LookupExpr>(pub A, pub E);

impl<A: LookupExpr, E: LookupExpr> LookupExpr for RecipLookupExpr<A, E> {
	fn last_dim_size(&self) -> usize {
		self.0.last_dim_size().max(self.1.last_dim_size())
	}
}

//--------------------------------------------------------------------------------------------------

pub struct LnClampedLookupExpr<A: LookupExpr>(pub A);

impl<A: LookupExpr> LookupExpr for LnClampedLookupExpr<A> {
	fn last_dim_size(&self) -> usize {
		self.0.last_dim_size()
	}
}

//--------------------------------------------------------------------------------------------------
