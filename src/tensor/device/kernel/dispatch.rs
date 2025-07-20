//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::ErrPack;
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

pub trait DispatchExpr {}

pub trait KernelDispatch<Expr: DispatchExpr> {
	fn dispatch(&self, out: &Tensor, expr: Expr) -> Result<(), ErrPack<TensorOpError>>;
}

//--------------------------------------------------------------------------------------------------

impl DispatchExpr for &Tensor {}

impl DispatchExpr for f64 {}

//--------------------------------------------------------------------------------------------------

pub struct AddDispatchExpr<A: DispatchExpr, B: DispatchExpr> {
	pub lhs: A,
	pub rhs: B,
}

impl<A: DispatchExpr, B: DispatchExpr> DispatchExpr for AddDispatchExpr<A, B> {}

//--------------------------------------------------------------------------------------------------

pub struct SubDispatchExpr<A: DispatchExpr, B: DispatchExpr> {
	pub lhs: A,
	pub rhs: B,
}

impl<A: DispatchExpr, B: DispatchExpr> DispatchExpr for SubDispatchExpr<A, B> {}

//--------------------------------------------------------------------------------------------------

pub struct MulDispatchExpr<A: DispatchExpr, B: DispatchExpr> {
	pub lhs: A,
	pub rhs: B,
}

impl<A: DispatchExpr, B: DispatchExpr> DispatchExpr for MulDispatchExpr<A, B> {}

//--------------------------------------------------------------------------------------------------

pub struct SumDispatchExpr<A: DispatchExpr> {
	pub expr: A,
}

impl<A: DispatchExpr> DispatchExpr for SumDispatchExpr<A> {}

//--------------------------------------------------------------------------------------------------
