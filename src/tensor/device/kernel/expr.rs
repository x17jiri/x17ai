//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::ErrPack;
use crate::tensor::math::EvaluatesToTensor;
use crate::tensor::{Tensor, TensorOpError};
use crate::util::LossyInto;

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ExprDiscriminant {
	TensorArg,
	ScalarArg,

	SumExpr,
	MaxExpr,

	RandnExpr,

	ExpExpr,
	AbsExpr,
	SigmoidExpr,
	SwishExpr,
	SqrtExpr,
	LnClampedExpr,

	AddExpr,
	MulExpr,
	RecipExpr,

	Invalid,
}

//--------------------------------------------------------------------------------------------------

pub enum DynExpr {
	ElemwiseTensorArg(usize),
	ReduceTensorArg(usize),
	ScalarArg(usize),

	SumExpr(Arc<DynExpr>),
	MaxExpr(Arc<DynExpr>),

	RandnExpr(),

	ExpExpr(Arc<DynExpr>),
	AbsExpr(Arc<DynExpr>),
	SigmoidExpr(Arc<DynExpr>),
	SwishExpr(Arc<DynExpr>),
	SqrtExpr(Arc<DynExpr>),
	LnClampedExpr(Arc<DynExpr>),

	AddExpr(Arc<DynExpr>, Arc<DynExpr>),
	MulExpr(Arc<DynExpr>, Arc<DynExpr>),
	RecipExpr(Arc<DynExpr>, Arc<DynExpr>),
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
	const PADDED_KEY_LEN: usize = Self::KEY_LEN.next_multiple_of(super::runner::KEY_BATCH_SIZE);
	const BATCHED_KEY_LEN: usize = Self::PADDED_KEY_LEN / super::runner::KEY_BATCH_SIZE;

	fn key(id: &mut [ExprDiscriminant], i: usize) -> usize;
	fn elemwise_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't;
	fn reduce_tensors<'t>(&self, tensors: &mut [MaybeUninit<&'t Tensor>], i: usize) -> usize
	where
		Self: 't;
	fn scalars(&self, scalars: &mut [f64], i: usize) -> usize;

	fn a_tensor(&self) -> Option<&Tensor>;
}

pub trait ExprToDyn {
	fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr>;
}

pub struct Scalar(pub f64);

pub struct SumExpr<A: Expr + ExprToDyn>(pub A);
pub struct MaxExpr<A: Expr + ExprToDyn>(pub A);

pub struct RandnExpr();

pub struct ExpExpr<A: Expr + ExprToDyn>(pub A);
pub struct AbsExpr<A: Expr + ExprToDyn>(pub A);
pub struct SigmoidExpr<A: Expr + ExprToDyn>(pub A);
pub struct SwishExpr<A: Expr + ExprToDyn>(pub A);
pub struct SqrtExpr<A: Expr + ExprToDyn>(pub A);
pub struct LnClampedExpr<A: Expr + ExprToDyn>(pub A);

pub struct AddExpr<A: Expr + ExprToDyn, B: Expr + ExprToDyn>(pub A, pub B);
pub struct MulExpr<A: Expr + ExprToDyn, B: Expr + ExprToDyn>(pub A, pub B);
pub struct RecipExpr<A: Expr + ExprToDyn, B: Expr + ExprToDyn>(pub A, pub B);

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

	#[inline(always)]
	fn a_tensor(&self) -> Option<&Tensor> {
		Some(*self)
	}
}

impl ExprToDyn for &Tensor {
	fn to_dyn(e: &mut usize, r: &mut usize, _s: &mut usize, reduce: bool) -> Arc<DynExpr> {
		if reduce {
			let node = Arc::new(DynExpr::ReduceTensorArg(*r));
			*r += 1;
			node
		} else {
			let node = Arc::new(DynExpr::ElemwiseTensorArg(*e));
			*e += 1;
			node
		}
	}
}

#[allow(clippy::inline_always)]
#[allow(clippy::indexing_slicing)]
impl const Expr for Scalar {
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
		scalars[i] = self.0;
		i + 1
	}

	#[inline(always)]
	fn a_tensor(&self) -> Option<&Tensor> {
		None
	}
}

impl ExprToDyn for Scalar {
	fn to_dyn(_e: &mut usize, _r: &mut usize, s: &mut usize, _reduce: bool) -> Arc<DynExpr> {
		let result = Arc::new(DynExpr::ScalarArg(*s));
		*s += 1;
		result
	}
}

macro_rules! impl_expr_reduce {
	($name:ident) => {
		#[allow(clippy::inline_always)]
		#[allow(clippy::indexing_slicing)]
		impl<A: const Expr + ExprToDyn> const Expr for $name<A> {
			const CONST: bool = A::CONST;
			const ELEMWISE_COUNT: usize = 0;
			const REDUCE_COUNT: usize = A::ELEMWISE_COUNT;
			const SCALAR_COUNT: usize = A::SCALAR_COUNT;
			const REDUCE_OP_COUNT: usize = 1 + A::REDUCE_OP_COUNT;
			const KEY_LEN: usize = 1 + A::KEY_LEN;

			#[inline(always)]
			fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
				id[i] = ExprDiscriminant::$name;
				A::key(id, i + 1)
			}

			#[inline(always)]
			fn elemwise_tensors<'t>(
				&self,
				_tensors: &mut [MaybeUninit<&'t Tensor>],
				i: usize,
			) -> usize
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

			#[inline(always)]
			fn a_tensor(&self) -> Option<&Tensor> {
				self.0.a_tensor()
			}
		}

		impl<A: const Expr + ExprToDyn> ExprToDyn for $name<A> {
			fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
				assert!(!reduce);
				Arc::new(DynExpr::$name(A::to_dyn(e, r, s, true)))
			}
		}
	};
}

macro_rules! impl_expr_nullary {
	($name:ident) => {
		#[allow(clippy::inline_always)]
		#[allow(clippy::indexing_slicing)]
		impl const Expr for $name {
			const CONST: bool = true;
			const ELEMWISE_COUNT: usize = 0;
			const REDUCE_COUNT: usize = 0;
			const SCALAR_COUNT: usize = 0;
			const REDUCE_OP_COUNT: usize = 0;
			const KEY_LEN: usize = 1;

			#[inline(always)]
			fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
				id[i] = ExprDiscriminant::$name;
				i + 1
			}

			#[inline(always)]
			fn elemwise_tensors<'t>(
				&self,
				_tensors: &mut [MaybeUninit<&'t Tensor>],
				i: usize,
			) -> usize
			where
				Self: 't,
			{
				i
			}

			#[inline(always)]
			fn reduce_tensors<'t>(
				&self,
				_tensors: &mut [MaybeUninit<&'t Tensor>],
				i: usize,
			) -> usize
			where
				Self: 't,
			{
				i
			}

			#[inline(always)]
			fn scalars(&self, _scalars: &mut [f64], i: usize) -> usize {
				i
			}

			#[inline(always)]
			fn a_tensor(&self) -> Option<&Tensor> {
				None
			}
		}

		impl ExprToDyn for $name {
			fn to_dyn(
				_e: &mut usize,
				_r: &mut usize,
				_s: &mut usize,
				_reduce: bool,
			) -> Arc<DynExpr> {
				Arc::new(DynExpr::$name())
			}
		}
	};
}

macro_rules! impl_expr_unary {
	($name:ident) => {
		#[allow(clippy::inline_always)]
		#[allow(clippy::indexing_slicing)]
		impl<A: const Expr + ExprToDyn> const Expr for $name<A> {
			const CONST: bool = A::CONST;
			const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT;
			const REDUCE_COUNT: usize = A::REDUCE_COUNT;
			const SCALAR_COUNT: usize = A::SCALAR_COUNT;
			const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT;
			const KEY_LEN: usize = 1 + A::KEY_LEN;

			#[inline(always)]
			fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
				id[i] = ExprDiscriminant::$name;
				A::key(id, i + 1)
			}

			#[inline(always)]
			fn elemwise_tensors<'t>(
				&self,
				tensors: &mut [MaybeUninit<&'t Tensor>],
				i: usize,
			) -> usize
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

			#[inline(always)]
			fn a_tensor(&self) -> Option<&Tensor> {
				self.0.a_tensor()
			}
		}

		impl<A: const Expr + ExprToDyn> ExprToDyn for $name<A> {
			fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
				Arc::new(DynExpr::$name(A::to_dyn(e, r, s, reduce)))
			}
		}
	};
}

macro_rules! impl_expr_binary {
	($name:ident) => {
		#[allow(clippy::inline_always)]
		#[allow(clippy::indexing_slicing)]
		impl<A: const Expr + ExprToDyn, B: const Expr + ExprToDyn> const Expr for $name<A, B> {
			const CONST: bool = A::CONST && B::CONST;
			const ELEMWISE_COUNT: usize = A::ELEMWISE_COUNT + B::ELEMWISE_COUNT;
			const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
			const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
			const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT + B::REDUCE_OP_COUNT;
			const KEY_LEN: usize = 1 + A::KEY_LEN + B::KEY_LEN;

			#[inline(always)]
			fn key(id: &mut [ExprDiscriminant], i: usize) -> usize {
				id[i] = ExprDiscriminant::$name;
				let m = A::key(id, i + 1);
				assert!(m == i + 1 + A::KEY_LEN);
				B::key(id, i + 1 + A::KEY_LEN)
			}

			#[inline(always)]
			fn elemwise_tensors<'t>(
				&self,
				tensors: &mut [MaybeUninit<&'t Tensor>],
				i: usize,
			) -> usize
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

			#[inline(always)]
			fn a_tensor(&self) -> Option<&Tensor> {
				self.0.a_tensor().or(self.1.a_tensor())
			}
		}

		impl<A: const Expr + ExprToDyn, B: const Expr + ExprToDyn> ExprToDyn for $name<A, B> {
			fn to_dyn(e: &mut usize, r: &mut usize, s: &mut usize, reduce: bool) -> Arc<DynExpr> {
				let a = A::to_dyn(e, r, s, reduce);
				let b = B::to_dyn(e, r, s, reduce);
				Arc::new(DynExpr::$name(a, b))
			}
		}
	};
}

impl_expr_reduce!(SumExpr);
impl_expr_reduce!(MaxExpr);

impl_expr_nullary!(RandnExpr);

impl_expr_unary!(ExpExpr);
impl_expr_unary!(AbsExpr);
impl_expr_unary!(SigmoidExpr);
impl_expr_unary!(SwishExpr);
impl_expr_unary!(SqrtExpr);
impl_expr_unary!(LnClampedExpr);

impl_expr_binary!(AddExpr);
impl_expr_binary!(MulExpr);
impl_expr_binary!(RecipExpr);

//--------------------------------------------------------------------------------------------------

impl<E: const Expr + ExprToDyn> EvaluatesToTensor for ExprWrapper<E>
where
	[(); 1 - E::REDUCE_OP_COUNT]:,
	[(); E::ELEMWISE_COUNT]:,
	[(); E::REDUCE_COUNT]:,
	[(); E::SCALAR_COUNT]:,
	[(); E::PADDED_KEY_LEN]:,
	[(); E::BATCHED_KEY_LEN]:,
	[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
{
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		to.vmt().kernel_runner().run(to, self.0)
	}
}

impl EvaluatesToTensor for Scalar {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.wrap().eval_to_tensor(to)
	}
}

impl EvaluatesToTensor for f64 {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.wrap().eval_to_tensor(to)
	}
}

impl EvaluatesToTensor for &Tensor {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		self.wrap().eval_to_tensor(to)
	}
}

//--------------------------------------------------------------------------------------------------

pub struct ExprWrapper<E: const Expr + ExprToDyn>(E);

pub trait Wrappable {
	type E: const Expr + ExprToDyn;

	fn wrap(self) -> ExprWrapper<Self::E>;
}

impl<'t> Wrappable for &'t Tensor {
	type E = Self;

	fn wrap(self) -> ExprWrapper<Self> {
		ExprWrapper(self)
	}
}

impl Wrappable for Scalar {
	type E = Self;

	fn wrap(self) -> ExprWrapper<Self> {
		ExprWrapper(self)
	}
}

impl Wrappable for f64 {
	type E = Scalar;

	fn wrap(self) -> ExprWrapper<Self::E> {
		ExprWrapper(Scalar(self))
	}
}

impl<E: const Expr + ExprToDyn> Wrappable for ExprWrapper<E> {
	type E = E;

	fn wrap(self) -> ExprWrapper<E> {
		self
	}
}

//--------------------------------------------------------------------------------------------------

impl std::ops::Add<Self> for &Tensor {
	type Output = ExprWrapper<AddExpr<Self, Self>>;

	fn add(self, rhs: Self) -> ExprWrapper<AddExpr<Self, Self>> {
		ExprWrapper(AddExpr(self, rhs))
	}
}

impl std::ops::Add<f64> for &Tensor {
	type Output = ExprWrapper<AddExpr<Self, Scalar>>;

	fn add(self, rhs: f64) -> ExprWrapper<AddExpr<Self, Scalar>> {
		ExprWrapper(AddExpr(self, Scalar(rhs)))
	}
}

impl std::ops::Add<Scalar> for &Tensor {
	type Output = ExprWrapper<AddExpr<Self, Scalar>>;

	fn add(self, rhs: Scalar) -> ExprWrapper<AddExpr<Self, Scalar>> {
		ExprWrapper(AddExpr(self, rhs))
	}
}

impl<E: const Expr + ExprToDyn> std::ops::Add<ExprWrapper<E>> for &Tensor {
	type Output = ExprWrapper<AddExpr<Self, E>>;

	fn add(self, rhs: ExprWrapper<E>) -> ExprWrapper<AddExpr<Self, E>> {
		ExprWrapper(AddExpr(self, rhs.0))
	}
}

impl<'t, A: const Expr + ExprToDyn> std::ops::Add<&'t Tensor> for ExprWrapper<A> {
	type Output = ExprWrapper<AddExpr<A, &'t Tensor>>;

	fn add(self, rhs: &'t Tensor) -> ExprWrapper<AddExpr<A, &'t Tensor>> {
		ExprWrapper(AddExpr(self.0, rhs))
	}
}

impl<A: const Expr + ExprToDyn> std::ops::Add<f64> for ExprWrapper<A> {
	type Output = ExprWrapper<AddExpr<A, Scalar>>;

	fn add(self, rhs: f64) -> ExprWrapper<AddExpr<A, Scalar>> {
		ExprWrapper(AddExpr(self.0, Scalar(rhs)))
	}
}

impl<A: const Expr + ExprToDyn> std::ops::Add<Scalar> for ExprWrapper<A> {
	type Output = ExprWrapper<AddExpr<A, Scalar>>;

	fn add(self, rhs: Scalar) -> ExprWrapper<AddExpr<A, Scalar>> {
		ExprWrapper(AddExpr(self.0, rhs))
	}
}

impl<A: const Expr + ExprToDyn, E: const Expr + ExprToDyn> std::ops::Add<ExprWrapper<E>>
	for ExprWrapper<A>
{
	type Output = ExprWrapper<AddExpr<A, E>>;

	fn add(self, rhs: ExprWrapper<E>) -> ExprWrapper<AddExpr<A, E>> {
		ExprWrapper(AddExpr(self.0, rhs.0))
	}
}

//--------------------------------------------------------------------------------------------------

impl std::ops::Sub<Self> for &Tensor {
	type Output = ExprWrapper<AddExpr<Self, MulExpr<Self, Scalar>>>;

	fn sub(self, rhs: Self) -> ExprWrapper<AddExpr<Self, MulExpr<Self, Scalar>>> {
		ExprWrapper(AddExpr(self, MulExpr(rhs, Scalar(-1.0))))
	}
}

impl std::ops::Sub<f64> for &Tensor {
	type Output = ExprWrapper<AddExpr<Self, Scalar>>;

	fn sub(self, rhs: f64) -> ExprWrapper<AddExpr<Self, Scalar>> {
		ExprWrapper(AddExpr(self, Scalar(-rhs)))
	}
}

impl std::ops::Sub<Scalar> for &Tensor {
	type Output = ExprWrapper<AddExpr<Self, Scalar>>;

	fn sub(self, rhs: Scalar) -> ExprWrapper<AddExpr<Self, Scalar>> {
		ExprWrapper(AddExpr(self, Scalar(rhs.0 * -1.0)))
	}
}

impl<E: const Expr + ExprToDyn> std::ops::Sub<ExprWrapper<E>> for &Tensor {
	type Output = ExprWrapper<AddExpr<Self, MulExpr<E, Scalar>>>;

	fn sub(self, rhs: ExprWrapper<E>) -> ExprWrapper<AddExpr<Self, MulExpr<E, Scalar>>> {
		ExprWrapper(AddExpr(self, MulExpr(rhs.0, Scalar(-1.0))))
	}
}

impl<'t, A: const Expr + ExprToDyn> std::ops::Sub<&'t Tensor> for ExprWrapper<A> {
	type Output = ExprWrapper<AddExpr<A, MulExpr<&'t Tensor, Scalar>>>;

	fn sub(self, rhs: &'t Tensor) -> ExprWrapper<AddExpr<A, MulExpr<&'t Tensor, Scalar>>> {
		ExprWrapper(AddExpr(self.0, MulExpr(rhs, Scalar(-1.0))))
	}
}

impl<A: const Expr + ExprToDyn> std::ops::Sub<f64> for ExprWrapper<A> {
	type Output = ExprWrapper<AddExpr<A, Scalar>>;

	fn sub(self, rhs: f64) -> ExprWrapper<AddExpr<A, Scalar>> {
		ExprWrapper(AddExpr(self.0, Scalar(-rhs)))
	}
}

impl<A: const Expr + ExprToDyn> std::ops::Sub<Scalar> for ExprWrapper<A> {
	type Output = ExprWrapper<AddExpr<A, Scalar>>;

	fn sub(self, rhs: Scalar) -> ExprWrapper<AddExpr<A, Scalar>> {
		ExprWrapper(AddExpr(self.0, Scalar(rhs.0 * -1.0)))
	}
}

impl<A: const Expr + ExprToDyn, E: const Expr + ExprToDyn> std::ops::Sub<ExprWrapper<E>>
	for ExprWrapper<A>
{
	type Output = ExprWrapper<AddExpr<A, MulExpr<E, Scalar>>>;

	fn sub(self, rhs: ExprWrapper<E>) -> ExprWrapper<AddExpr<A, MulExpr<E, Scalar>>> {
		ExprWrapper(AddExpr(self.0, MulExpr(rhs.0, Scalar(-1.0))))
	}
}

//--------------------------------------------------------------------------------------------------

impl std::ops::Mul<Self> for &Tensor {
	type Output = ExprWrapper<MulExpr<Self, Self>>;

	fn mul(self, rhs: Self) -> ExprWrapper<MulExpr<Self, Self>> {
		ExprWrapper(MulExpr(self, rhs))
	}
}

impl std::ops::Mul<f64> for &Tensor {
	type Output = ExprWrapper<MulExpr<Self, Scalar>>;

	fn mul(self, rhs: f64) -> ExprWrapper<MulExpr<Self, Scalar>> {
		ExprWrapper(MulExpr(self, Scalar(rhs)))
	}
}

impl std::ops::Mul<Scalar> for &Tensor {
	type Output = ExprWrapper<MulExpr<Self, Scalar>>;

	fn mul(self, rhs: Scalar) -> ExprWrapper<MulExpr<Self, Scalar>> {
		ExprWrapper(MulExpr(self, rhs))
	}
}

impl<E: const Expr + ExprToDyn> std::ops::Mul<ExprWrapper<E>> for &Tensor {
	type Output = ExprWrapper<MulExpr<Self, E>>;

	fn mul(self, rhs: ExprWrapper<E>) -> ExprWrapper<MulExpr<Self, E>> {
		ExprWrapper(MulExpr(self, rhs.0))
	}
}

impl<'t, A: const Expr + ExprToDyn> std::ops::Mul<&'t Tensor> for ExprWrapper<A> {
	type Output = ExprWrapper<MulExpr<A, &'t Tensor>>;

	fn mul(self, rhs: &'t Tensor) -> ExprWrapper<MulExpr<A, &'t Tensor>> {
		ExprWrapper(MulExpr(self.0, rhs))
	}
}

impl<A: const Expr + ExprToDyn> std::ops::Mul<f64> for ExprWrapper<A> {
	type Output = ExprWrapper<MulExpr<A, Scalar>>;

	fn mul(self, rhs: f64) -> ExprWrapper<MulExpr<A, Scalar>> {
		ExprWrapper(MulExpr(self.0, Scalar(rhs)))
	}
}

impl<A: const Expr + ExprToDyn> std::ops::Mul<Scalar> for ExprWrapper<A> {
	type Output = ExprWrapper<MulExpr<A, Scalar>>;

	fn mul(self, rhs: Scalar) -> ExprWrapper<MulExpr<A, Scalar>> {
		ExprWrapper(MulExpr(self.0, rhs))
	}
}

impl<A: const Expr + ExprToDyn, E: const Expr + ExprToDyn> std::ops::Mul<ExprWrapper<E>>
	for ExprWrapper<A>
{
	type Output = ExprWrapper<MulExpr<A, E>>;

	fn mul(self, rhs: ExprWrapper<E>) -> ExprWrapper<MulExpr<A, E>> {
		ExprWrapper(MulExpr(self.0, rhs.0))
	}
}

//--------------------------------------------------------------------------------------------------

pub fn randn() -> ExprWrapper<RandnExpr> {
	ExprWrapper(RandnExpr())
}

pub trait TensorOps {
	type E: const Expr + ExprToDyn;

	fn sum(self) -> ExprWrapper<SumExpr<Self::E>>;
	fn mean(self) -> ExprWrapper<MulExpr<SumExpr<Self::E>, Scalar>>;
	fn max(self) -> ExprWrapper<MaxExpr<Self::E>>;

	fn exp(self) -> ExprWrapper<ExpExpr<Self::E>>;
	fn abs(self) -> ExprWrapper<AbsExpr<Self::E>>;
	fn sigmoid(self) -> ExprWrapper<SigmoidExpr<Self::E>>;
	fn swish(self) -> ExprWrapper<SwishExpr<Self::E>>;
	fn sqrt(self) -> ExprWrapper<SqrtExpr<Self::E>>;
	fn ln_clamped(self) -> ExprWrapper<LnClampedExpr<Self::E>>;

	fn recip<B: Wrappable>(self, b: B) -> ExprWrapper<RecipExpr<Self::E, B::E>>;
}

impl<T: Wrappable> TensorOps for T {
	type E = T::E;

	fn sum(self) -> ExprWrapper<SumExpr<T::E>> {
		ExprWrapper(SumExpr(self.wrap().0))
	}

	fn mean(self) -> ExprWrapper<MulExpr<SumExpr<T::E>, Scalar>> {
		let expr = self.wrap().0;
		let n = expr.a_tensor().and_then(|t| t.size(-1).ok()).unwrap_or(1);
		let sum_to_mean = 1.0 / n.lossy_into();
		ExprWrapper(MulExpr(SumExpr(expr), Scalar(sum_to_mean)))
	}

	fn max(self) -> ExprWrapper<MaxExpr<T::E>> {
		ExprWrapper(MaxExpr(self.wrap().0))
	}

	fn exp(self) -> ExprWrapper<ExpExpr<T::E>> {
		ExprWrapper(ExpExpr(self.wrap().0))
	}

	fn abs(self) -> ExprWrapper<AbsExpr<T::E>> {
		ExprWrapper(AbsExpr(self.wrap().0))
	}

	fn sigmoid(self) -> ExprWrapper<SigmoidExpr<T::E>> {
		ExprWrapper(SigmoidExpr(self.wrap().0))
	}

	fn swish(self) -> ExprWrapper<SwishExpr<T::E>> {
		ExprWrapper(SwishExpr(self.wrap().0))
	}

	fn sqrt(self) -> ExprWrapper<SqrtExpr<T::E>> {
		ExprWrapper(SqrtExpr(self.wrap().0))
	}

	fn ln_clamped(self) -> ExprWrapper<LnClampedExpr<T::E>> {
		ExprWrapper(LnClampedExpr(self.wrap().0))
	}

	fn recip<B: Wrappable>(self, b: B) -> ExprWrapper<RecipExpr<T::E, B::E>> {
		ExprWrapper(RecipExpr(self.wrap().0, b.wrap().0))
	}
}
//--------------------------------------------------------------------------------------------------
