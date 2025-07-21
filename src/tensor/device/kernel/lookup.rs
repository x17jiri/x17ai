//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::math::EvaluatesToTensor;
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

pub trait LookupExpr {}

pub trait KernelLookup<Expr: LookupExpr> {
	type CallType: EvaluatesToTensor;

	fn create_call(&self, expr: Wrapper<Expr>) -> Self::CallType;
}

//--------------------------------------------------------------------------------------------------

pub struct Wrapper<Expr: LookupExpr>(pub Expr);

//--------------------------------------------------------------------------------------------------

impl LookupExpr for &Tensor {}

pub fn tensor(tensor: &Tensor) -> Wrapper<&Tensor> {
	Wrapper(tensor)
}

//--------------------------------------------------------------------------------------------------

impl LookupExpr for f64 {}

pub fn scalar(value: f64) -> Wrapper<f64> {
	Wrapper(value)
}

//--------------------------------------------------------------------------------------------------

pub struct AddLookupExpr<A: LookupExpr, B: LookupExpr>(pub A, pub B);

impl<A: LookupExpr, B: LookupExpr> LookupExpr for AddLookupExpr<A, B> {}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Add<Wrapper<B>> for Wrapper<A> {
	type Output = Wrapper<AddLookupExpr<A, B>>;

	fn add(self, rhs: Wrapper<B>) -> Wrapper<AddLookupExpr<A, B>> {
		let Wrapper(lhs) = self;
		let Wrapper(rhs) = rhs;
		Wrapper(AddLookupExpr(lhs, rhs))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SubLookupExpr<A: LookupExpr, B: LookupExpr>(pub A, pub B);

impl<A: LookupExpr, B: LookupExpr> LookupExpr for SubLookupExpr<A, B> {}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Sub<Wrapper<B>> for Wrapper<A> {
	type Output = Wrapper<SubLookupExpr<A, B>>;

	fn sub(self, rhs: Wrapper<B>) -> Wrapper<SubLookupExpr<A, B>> {
		let Wrapper(lhs) = self;
		let Wrapper(rhs) = rhs;
		Wrapper(SubLookupExpr(lhs, rhs))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct MulLookupExpr<A: LookupExpr, B: LookupExpr>(pub A, pub B);

impl<A: LookupExpr, B: LookupExpr> LookupExpr for MulLookupExpr<A, B> {}

#[allow(clippy::use_self)]
impl<A: LookupExpr, B: LookupExpr> std::ops::Mul<Wrapper<B>> for Wrapper<A> {
	type Output = Wrapper<MulLookupExpr<A, B>>;

	fn mul(self, rhs: Wrapper<B>) -> Wrapper<MulLookupExpr<A, B>> {
		let Wrapper(lhs) = self;
		let Wrapper(rhs) = rhs;
		Wrapper(MulLookupExpr(lhs, rhs))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SumLookupExpr<A: LookupExpr>(pub A);

impl<A: LookupExpr> LookupExpr for SumLookupExpr<A> {}

//--------------------------------------------------------------------------------------------------
