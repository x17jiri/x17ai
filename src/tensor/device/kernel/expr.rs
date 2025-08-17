//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::tensor::Tensor;

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ExprDiscriminant {
	TensorArg,
	ScalarArg,

	SumExpr,

	SigmoidExpr,
	SwishExpr,
	SqrtExpr,
	RecipExpr,
	LnClampedExpr,
	AddExpr,
	MulExpr,

	Invalid,
}

//--------------------------------------------------------------------------------------------------

pub enum DynExpr {
	ElemwiseTensorArg(usize),
	ReduceTensorArg(usize),
	ScalarArg(usize),

	SumExpr(Arc<DynExpr>),

	SigmoidExpr(Arc<DynExpr>),
	SwishExpr(Arc<DynExpr>),
	SqrtExpr(Arc<DynExpr>),
	RecipExpr(Arc<DynExpr>, Arc<DynExpr>),
	LnClampedExpr(Arc<DynExpr>),
	AddExpr(Arc<DynExpr>, Arc<DynExpr>),
	MulExpr(Arc<DynExpr>, Arc<DynExpr>),
}

//--------------------------------------------------------------------------------------------------

#[const_trait]
pub trait Expr {
	const CONST: bool;
	const ELEMWISE_COUNT: usize;
	const REDUCE_COUNT: usize;
	const SCALAR_COUNT: usize;
	const REDUCE_OP_COUNT: usize;
	const KEY_LEN: usize;

	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize;
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't;
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't;
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize;
}

pub trait ExprToDyn {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr>;
}

pub struct SumExpr<A: Expr + ExprToDyn>(pub A);

pub struct SigmoidExpr<A: Expr + ExprToDyn>(pub A);
pub struct SwishExpr<A: Expr + ExprToDyn>(pub A);
pub struct SqrtExpr<A: Expr + ExprToDyn>(pub A);
pub struct RecipExpr<A: Expr + ExprToDyn, B: Expr + ExprToDyn>(pub A, pub B);
pub struct LnClampedExpr<A: Expr + ExprToDyn>(pub A);
pub struct AddExpr<A: Expr + ExprToDyn, B: Expr + ExprToDyn>(pub A, pub B);
pub struct MulExpr<A: Expr + ExprToDyn, B: Expr + ExprToDyn>(pub A, pub B);

//--------------------------------------------------------------------------------------------------

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<'a> const Expr for &'a Tensor {
	const CONST: bool = true;
	const ELEMWISE_COUNT: usize = 1;
	const REDUCE_COUNT: usize = 0;
	const SCALAR_COUNT: usize = 0;
	const REDUCE_OP_COUNT: usize = 0;
	const KEY_LEN: usize = 1;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::TensorArg;
		i + 1
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		tensors[i].write(*self);
		i + 1
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, _tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		i
	}

	#[inline(always)]
	fn scalars(&self, _scalars: &mut [f64], i: usize) -> usize {
		i
	}
}

impl<'a> ExprToDyn for &'a Tensor {
	fn to_dyn(e: &mut usize, r: &mut usize, _s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		if reduce {
			let node = Arc::new(DynExpr::ElemwiseTensorArg(*r));
			*r += 1;
			node
		} else {
			let node = Arc::new(DynExpr::ReduceTensorArg(*e));
			*e += 1;
			node
		}
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl const Expr for f64 {
	const CONST: bool = true;
	const ELEMWISE_COUNT: usize = 0;
	const REDUCE_COUNT: usize = 0;
	const SCALAR_COUNT: usize = 1;
	const REDUCE_OP_COUNT: usize = 0;
	const KEY_LEN: usize = 1;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::ScalarArg;
		i + 1
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, _tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		i
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, _tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		i
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		scalars[i] = *self;
		i + 1
	}
}

impl ExprToDyn for f64 {
	fn to_dyn(_e: &mut usize, _r: &mut usize, s: &mut usize, _reduce: bool) -> Arc<DynExpr> {
		let result = Arc::new(DynExpr::ScalarArg(*s));
		*s += 1;
		result
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr + ExprToDyn> const Expr for SumExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = 0;
	const REDUCE_COUNT: usize = A::ELEMWISE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const REDUCE_OP_COUNT: usize = 1 + A::REDUCE_OP_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::SumExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, _tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		i
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

impl<A: const Expr + ExprToDyn> ExprToDyn for SumExpr<A> {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		assert!(!reduce);
		Arc::new(DynExpr::SumExpr(A::to_dyn(e, r, s, true)))
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr + ExprToDyn> const Expr for SigmoidExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::SigmoidExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.reduce_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

impl<A: const Expr + ExprToDyn> ExprToDyn for SigmoidExpr<A> {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		Arc::new(DynExpr::SigmoidExpr(A::to_dyn(e, r, s, reduce)))
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr + ExprToDyn> const Expr for SwishExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::SwishExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.reduce_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

impl<A: const Expr + ExprToDyn> ExprToDyn for SwishExpr<A> {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		Arc::new(DynExpr::SwishExpr(A::to_dyn(e, r, s, reduce)))
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr + ExprToDyn> const Expr for SqrtExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::SqrtExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.reduce_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

impl<A: const Expr + ExprToDyn> ExprToDyn for SqrtExpr<A> {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		Arc::new(DynExpr::SqrtExpr(A::to_dyn(e, r, s, reduce)))
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr + ExprToDyn, B: const Expr + ExprToDyn> const Expr for RecipExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT + B::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT + B::REDUCE_OP_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN + B::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::RecipExpr;
		let m = A::key(id, i + 1);
		assert!(m == i + 1 + A::KEY_LEN);
		B::key(id, i + 1 + A::KEY_LEN)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		let m = self.0.elemwise_tensors(tensors, i);
		assert!(m == i + A::ELEMWISE_COUNT);
		self.1.elemwise_tensors(tensors, i + A::ELEMWISE_COUNT)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		let m = self.0.reduce_tensors(tensors, i);
		assert!(m == i + A::REDUCE_COUNT);
		self.1.reduce_tensors(tensors, i + A::REDUCE_COUNT)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		let m = self.0.scalars(scalars, i);
		assert!(m == i + A::SCALAR_COUNT);
		self.1.scalars(scalars, i + A::SCALAR_COUNT)
	}
}

impl<A: const Expr + ExprToDyn, B: const Expr + ExprToDyn> ExprToDyn for RecipExpr<A, B> {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		let a = A::to_dyn(e, r, s, reduce);
		let b = B::to_dyn(e, r, s, reduce);
		Arc::new(DynExpr::RecipExpr(a, b))
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr + ExprToDyn> const Expr for LnClampedExpr<A> {
	const CONST: bool = A::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::LnClampedExpr;
		A::key(id, i + 1)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.elemwise_tensors(tensors, i)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		self.0.reduce_tensors(tensors, i)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		self.0.scalars(scalars, i)
	}
}

impl<A: const Expr + ExprToDyn> ExprToDyn for LnClampedExpr<A> {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		Arc::new(DynExpr::LnClampedExpr(A::to_dyn(e, r, s, reduce)))
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr + ExprToDyn, B: const Expr + ExprToDyn> const Expr for AddExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT + B::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT + B::REDUCE_OP_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN + B::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::AddExpr;
		let m = A::key(id, i + 1);
		assert!(m == i + 1 + A::KEY_LEN);
		B::key(id, i + 1 + A::KEY_LEN)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		let m = self.0.elemwise_tensors(tensors, i);
		assert!(m == i + A::ELEMWISE_COUNT);
		self.1.elemwise_tensors(tensors, i + A::ELEMWISE_COUNT)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		let m = self.0.reduce_tensors(tensors, i);
		assert!(m == i + A::REDUCE_COUNT);
		self.1.reduce_tensors(tensors, i + A::REDUCE_COUNT)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		let m = self.0.scalars(scalars, i);
		assert!(m == i + A::SCALAR_COUNT);
		self.1.scalars(scalars, i + A::SCALAR_COUNT)
	}
}

impl<A: const Expr + ExprToDyn, B: const Expr + ExprToDyn> ExprToDyn for AddExpr<A, B> {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		let a = A::to_dyn(e, r, s, reduce);
		let b = B::to_dyn(e, r, s, reduce);
		Arc::new(DynExpr::AddExpr(a, b))
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl<A: const Expr + ExprToDyn, B: const Expr + ExprToDyn> const Expr for MulExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT + B::ELEMWISE_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT + B::REDUCE_OP_COUNT;
	const KEY_LEN: usize = 1 + A::KEY_LEN + B::KEY_LEN;

	#[inline(always)]
	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
		id[i] = ExprDiscriminant::MulExpr;
		let m = A::key(id, i + 1);
		assert!(m == i + 1 + A::KEY_LEN);
		B::key(id, i + 1 + A::KEY_LEN)
	}

	#[inline(always)]
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		let m = self.0.elemwise_tensors(tensors, i);
		assert!(m == i + A::ELEMWISE_COUNT);
		self.1.elemwise_tensors(tensors, i + A::ELEMWISE_COUNT)
	}

	#[inline(always)]
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't,
	{
		let m = self.0.reduce_tensors(tensors, i);
		assert!(m == i + A::REDUCE_COUNT);
		self.1.reduce_tensors(tensors, i + A::REDUCE_COUNT)
	}

	#[inline(always)]
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize {
		let m = self.0.scalars(scalars, i);
		assert!(m == i + A::SCALAR_COUNT);
		self.1.scalars(scalars, i + A::SCALAR_COUNT)
	}
}

impl<A: const Expr + ExprToDyn, B: const Expr + ExprToDyn> ExprToDyn for MulExpr<A, B> {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		let a = A::to_dyn(e, r, s, reduce);
		let b = B::to_dyn(e, r, s, reduce);
		Arc::new(DynExpr::MulExpr(a, b))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ExprWrapper<E: const Expr + ExprToDyn>(E);

impl<E: const Expr + ExprToDyn> From<E> for ExprWrapper<E> {
	fn from(expr: E) -> Self {
		Self(expr)
	}
}

impl<E, W> std::ops::Add<W> for &Tensor
where
	E: const Expr + ExprToDyn,
	W: Into<ExprWrapper<E>>,
{
	type Output = ExprWrapper<AddExpr<Self, E>>;

	fn add(self, rhs: W) -> Self::Output {
		ExprWrapper(
			AddExpr(
				*self,
				rhs.into().0,
			),
		)
	}
}

//--------------------------------------------------------------------------------------------------
