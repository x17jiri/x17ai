//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::Tensor;
use crate::tensor::math::EvaluatesToTensor;

//--------------------------------------------------------------------------------------------------

pub trait LookupExpr {}

pub trait KernelLookup<Expr: LookupExpr> {
	type CallType: EvaluatesToTensor;

	fn create_call(&self, expr: LookupWrapper<Expr>) -> Self::CallType;
}

//--------------------------------------------------------------------------------------------------

pub struct LookupWrapper<Expr: LookupExpr>(pub Expr);

impl<Expr: LookupExpr> LookupWrapper<Expr> {
	pub fn sum(self) -> LookupWrapper<SumLookupExpr<Expr>> {
		LookupWrapper(SumLookupExpr(self.0))
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
}

//--------------------------------------------------------------------------------------------------

impl LookupExpr for &Tensor {}

pub fn tsr(tensor: &Tensor) -> LookupWrapper<&Tensor> {
	LookupWrapper(tensor)
}

//--------------------------------------------------------------------------------------------------

impl LookupExpr for f64 {}

pub fn scalar(value: f64) -> LookupWrapper<f64> {
	LookupWrapper(value)
}

//--------------------------------------------------------------------------------------------------

pub struct AddLookupExpr<A: LookupExpr, B: LookupExpr>(pub A, pub B);

impl<A: LookupExpr, B: LookupExpr> LookupExpr for AddLookupExpr<A, B> {}

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

impl<A: LookupExpr, B: LookupExpr> LookupExpr for SubLookupExpr<A, B> {}

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

impl<A: LookupExpr, B: LookupExpr> LookupExpr for MulLookupExpr<A, B> {}

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

impl<A: LookupExpr> LookupExpr for SumLookupExpr<A> {}

//--------------------------------------------------------------------------------------------------

pub struct SqrtLookupExpr<A: LookupExpr>(pub A);

impl<A: LookupExpr> LookupExpr for SqrtLookupExpr<A> {}

//--------------------------------------------------------------------------------------------------

pub struct RecipLookupExpr<A: LookupExpr, E: LookupExpr>(pub A, pub E);

impl<A: LookupExpr, E: LookupExpr> LookupExpr for RecipLookupExpr<A, E> {}

//--------------------------------------------------------------------------------------------------
