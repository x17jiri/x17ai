//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::Arc;

use crate::ErrPack;
use crate::tensor::device::kernel::registry::KernelMap;
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

#[macro_export]
macro_rules! custom_kernel {
	(
		[ $($tensor_id:ident : $tensor_expr:expr),* $(,)? ],
		( $($scalar_id:ident : $scalar_expr:expr),* $(,)? ),
		$body:expr
	) => {{
		$crate::tensor::device::kernel::expr::KernelCall::new(
			[ $($tensor_expr),* ],
			[ $($scalar_expr),* ],
			{
				$(
					let $tensor_id = $crate::tensor::device::kernel::expr::Expr(
						$crate::tensor::device::kernel::expr::TensorArg::<${index()}>
					);
				)*
				$(
					let $scalar_id = $crate::tensor::device::kernel::expr::Expr(
						$crate::tensor::device::kernel::expr::ScalarArg::<${index()}>
					);
				)*
				$body
			}
		)
	}};
}

//--------------------------------------------------------------------------------------------------

pub trait EvaluatesToTensor {
	/// Calculate the result of the operation represented by `self`
	/// and save it into the `to` tensor.
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>>;
}

impl EvaluatesToTensor for f64 {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		to.assign(custom_kernel!(
			[], (VALUE: self), {
				VALUE
			}
		))
	}
}

impl EvaluatesToTensor for &Tensor {
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		to.assign(custom_kernel!(
			[src: self], (), {
				src
			}
		))
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ExprDiscriminant {
	ElemwiseTensorArg,
	ReduceTensorArg,
	ScalarArg,

	SumExpr,
	MaxExpr,

	NegExpr,
	ExpExpr,
	LnExpr,
	AbsExpr,
	SqrtExpr,
	RecipExpr,

	AddExpr,
	SubExpr,
	MulExpr,

	Invalid,
}

//--------------------------------------------------------------------------------------------------

pub enum DynExpr {
	ElemwiseTensorArg(usize),
	ReduceTensorArg(usize),
	ScalarArg(usize),

	SumExpr(Arc<DynExpr>),
	MaxExpr(Arc<DynExpr>),

	NegExpr(Arc<DynExpr>),
	ExpExpr(Arc<DynExpr>),
	LnExpr(Arc<DynExpr>),
	AbsExpr(Arc<DynExpr>),
	SqrtExpr(Arc<DynExpr>),
	RecipExpr(Arc<DynExpr>),

	AddExpr(Arc<DynExpr>, Arc<DynExpr>),
	SubExpr(Arc<DynExpr>, Arc<DynExpr>),
	MulExpr(Arc<DynExpr>, Arc<DynExpr>),
}

//--------------------------------------------------------------------------------------------------

pub const KEY_BATCH_SIZE: usize = std::mem::size_of::<u64>();

union KeyUnion<const PADDED_KEY_LEN: usize, const BATCHED_KEY_LEN: usize>
where
	[(); PADDED_KEY_LEN]:,
	[(); BATCHED_KEY_LEN]:,
{
	id: [u8; PADDED_KEY_LEN],
	key: [u64; BATCHED_KEY_LEN],
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct InputMasks {
	pub elemwise: u64,
	pub reduce: u64,
	pub scalar: u64,
}

#[derive(Clone, Copy)]
pub struct InputCounts {
	pub elemwise: usize,
	pub reduce: usize,
	pub scalar: usize,
}

impl InputCounts {
	pub const fn new(masks: InputMasks) -> Self {
		Self {
			elemwise: masks.elemwise.count_ones() as usize,
			reduce: masks.reduce.count_ones() as usize,
			scalar: masks.scalar.count_ones() as usize,
		}
	}
}

#[const_trait]
pub trait ExprTrait {
	const MASKS: InputMasks;
	const COUNTS: InputCounts = InputCounts::new(Self::MASKS);

	const ELEMWISE_MASK: u64 = Self::MASKS.elemwise;
	const REDUCE_MASK: u64 = Self::MASKS.reduce;
	const SCALAR_MASK: u64 = Self::MASKS.scalar;

	const ELEMWISE_COUNT: usize = Self::COUNTS.elemwise;
	const REDUCE_COUNT: usize = Self::COUNTS.reduce;
	const SCALAR_COUNT: usize = Self::COUNTS.scalar;

	const REDUCE_OP_COUNT: usize;

	const KEY_LEN: usize;
	const PADDED_KEY_LEN: usize = Self::KEY_LEN.next_multiple_of(KEY_BATCH_SIZE);
	const BATCHED_KEY_LEN: usize = Self::PADDED_KEY_LEN / KEY_BATCH_SIZE;

	fn set_key(masks: InputMasks, reduce: bool, id: &mut [u8], i: usize) -> usize;

	#[allow(clippy::indexing_slicing)]
	fn key() -> ([u64; Self::BATCHED_KEY_LEN], u64)
	where
		[(); Self::PADDED_KEY_LEN]:,
	{
		let mut id = [255; Self::PADDED_KEY_LEN];
		let len = Self::set_key(Self::MASKS, false, &mut id, 0);
		assert!(len == Self::KEY_LEN);

		let key_union = KeyUnion { id };
		let key = unsafe { key_union.key };
		let key_hash = KernelMap::hash_key(&key);

		(key, key_hash)
	}
}

pub trait ExprToDyn {
	fn to_dyn(e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Arc<DynExpr>;
}

#[derive(Clone, Copy)]
pub struct TensorArg<const Idx: usize>;
#[derive(Clone, Copy)]
pub struct ScalarArg<const Idx: usize>;

#[derive(Clone, Copy)]
pub struct SumExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct MaxExpr<A: const ExprTrait + ExprToDyn>(pub A);

#[derive(Clone, Copy)]
pub struct NegExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct ExpExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct LnExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct AbsExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct SqrtExpr<A: const ExprTrait + ExprToDyn>(pub A);
#[derive(Clone, Copy)]
pub struct RecipExpr<A: const ExprTrait + ExprToDyn>(pub A);

#[derive(Clone, Copy)]
pub struct AddExpr<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn>(pub A, pub B);
#[derive(Clone, Copy)]
pub struct SubExpr<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn>(pub A, pub B);
#[derive(Clone, Copy)]
pub struct MulExpr<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn>(pub A, pub B);

//--------------------------------------------------------------------------------------------------

impl<const Idx: usize> const ExprTrait for TensorArg<Idx> {
	const MASKS: InputMasks = InputMasks { elemwise: 1 << Idx, reduce: 0, scalar: 0 };

	const REDUCE_OP_COUNT: usize = 0;

	const KEY_LEN: usize = 2;

	fn set_key(masks: InputMasks, reduce: bool, id: &mut [u8], i: usize) -> usize {
		let bit = 1_u64 << Idx;
		if reduce {
			let idx = (masks.reduce & (bit - 1)).count_ones() as usize;
			id[i + 0] = ExprDiscriminant::ReduceTensorArg as u8;
			id[i + 1] = idx as u8;
		} else {
			let idx = (masks.elemwise & (bit - 1)).count_ones() as usize;
			id[i + 0] = ExprDiscriminant::ElemwiseTensorArg as u8;
			id[i + 1] = idx as u8;
		}
		i + 2
	}
}

impl<const Idx: usize> ExprToDyn for TensorArg<Idx> {
	fn to_dyn(em: u64, rm: u64, _sm: u64, reduce: bool) -> Arc<DynExpr> {
		let bit = 1_u64 << Idx;
		if reduce {
			let idx = (rm & (bit - 1)).count_ones() as usize;
			Arc::new(DynExpr::ReduceTensorArg(idx))
		} else {
			let idx = (em & (bit - 1)).count_ones() as usize;
			Arc::new(DynExpr::ElemwiseTensorArg(idx))
		}
	}
}

impl<const Idx: usize> const ExprTrait for ScalarArg<Idx> {
	const MASKS: InputMasks = InputMasks { elemwise: 0, reduce: 0, scalar: 1 << Idx };

	const REDUCE_OP_COUNT: usize = 0;

	const KEY_LEN: usize = 2;

	fn set_key(masks: InputMasks, _reduce: bool, id: &mut [u8], i: usize) -> usize {
		let bit = 1_u64 << Idx;
		let idx = (masks.scalar & (bit - 1)).count_ones() as usize;
		id[i + 0] = ExprDiscriminant::ScalarArg as u8;
		id[i + 1] = idx as u8;
		i + 2
	}
}

impl<const Idx: usize> ExprToDyn for ScalarArg<Idx> {
	fn to_dyn(_e_mask: u64, _r_mask: u64, s_mask: u64, _reduce: bool) -> Arc<DynExpr> {
		let bit = 1_u64 << Idx;
		let idx = (s_mask & (bit - 1)).count_ones() as usize;
		Arc::new(DynExpr::ScalarArg(idx))
	}
}

macro_rules! impl_expr_reduce {
	($name:ident) => {
		impl<A: const ExprTrait + ExprToDyn> const ExprTrait for $name<A> {
			const MASKS: InputMasks = InputMasks {
				elemwise: 0,
				reduce: A::MASKS.reduce | A::MASKS.elemwise,
				scalar: A::MASKS.scalar,
			};

			const REDUCE_OP_COUNT: usize = 1 + A::REDUCE_OP_COUNT;

			const KEY_LEN: usize = 1 + A::KEY_LEN;

			fn set_key(masks: InputMasks, _reduce: bool, id: &mut [u8], i: usize) -> usize {
				id[i] = ExprDiscriminant::$name as u8;
				A::set_key(masks, true, id, i + 1)
			}
		}

		impl<A: const ExprTrait + ExprToDyn> ExprToDyn for $name<A> {
			fn to_dyn(e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Arc<DynExpr> {
				assert!(!reduce);
				Arc::new(DynExpr::$name(A::to_dyn(e_mask, r_mask, s_mask, true)))
			}
		}
	};
}

macro_rules! impl_expr_unary {
	($name:ident) => {
		impl<A: const ExprTrait + ExprToDyn> const ExprTrait for $name<A> {
			const MASKS: InputMasks = A::MASKS;

			const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT;

			const KEY_LEN: usize = 1 + A::KEY_LEN;

			fn set_key(masks: InputMasks, reduce: bool, id: &mut [u8], i: usize) -> usize {
				id[i + 0] = ExprDiscriminant::$name as u8;
				A::set_key(masks, reduce, id, i + 1)
			}
		}

		impl<A: const ExprTrait + ExprToDyn> ExprToDyn for $name<A> {
			fn to_dyn(e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Arc<DynExpr> {
				Arc::new(DynExpr::$name(A::to_dyn(e_mask, r_mask, s_mask, reduce)))
			}
		}
	};
}

macro_rules! impl_expr_binary {
	($name:ident, $commutative:expr) => {
		impl<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn> const ExprTrait
			for $name<A, B>
		{
			const MASKS: InputMasks = InputMasks {
				elemwise: A::MASKS.elemwise | B::MASKS.elemwise,
				reduce: A::MASKS.reduce | B::MASKS.reduce,
				scalar: A::MASKS.scalar | B::MASKS.scalar,
			};

			const REDUCE_OP_COUNT: usize = A::REDUCE_OP_COUNT + B::REDUCE_OP_COUNT;

			const KEY_LEN: usize = 1 + A::KEY_LEN + B::KEY_LEN;

			fn set_key(masks: InputMasks, reduce: bool, id: &mut [u8], i: usize) -> usize {
				id[i + 0] = ExprDiscriminant::$name as u8;
				let begin = i + 1;
				let mid = A::set_key(masks, reduce, id, begin);
				assert!(mid == begin + A::KEY_LEN);
				let end = B::set_key(masks, reduce, id, mid);
				assert!(end == mid + B::KEY_LEN);
				if !($commutative) {
					return end;
				}

				if A::KEY_LEN <= B::KEY_LEN {
					if A::KEY_LEN == B::KEY_LEN {
						let mut swap = false;
						let mut i = 0;
						while i < A::KEY_LEN {
							let a = id[begin + i];
							let b = id[mid + i];
							if a != b {
								swap = a > b;
								break;
							}
							i += 1;
						}
						if swap {
							let mut i = 0;
							while i < A::KEY_LEN {
								let tmp = id[begin + i];
								id[begin + i] = id[mid + i];
								id[mid + i] = tmp;
								i += 1;
							}
						}
					}
					end
				} else {
					let mid = B::set_key(masks, reduce, id, begin);
					assert!(mid == begin + B::KEY_LEN);
					let end = A::set_key(masks, reduce, id, mid);
					assert!(end == mid + A::KEY_LEN);
					end
				}
			}
		}

		impl<A: const ExprTrait + ExprToDyn, B: const ExprTrait + ExprToDyn> ExprToDyn
			for $name<A, B>
		{
			fn to_dyn(e_mask: u64, r_mask: u64, s_mask: u64, reduce: bool) -> Arc<DynExpr> {
				let a = A::to_dyn(e_mask, r_mask, s_mask, reduce);
				let b = B::to_dyn(e_mask, r_mask, s_mask, reduce);
				Arc::new(DynExpr::$name(a, b))
			}
		}
	};
}

impl_expr_reduce!(SumExpr);
impl_expr_reduce!(MaxExpr);

impl_expr_unary!(NegExpr);
impl_expr_unary!(ExpExpr);
impl_expr_unary!(LnExpr);
impl_expr_unary!(AbsExpr);
impl_expr_unary!(SqrtExpr);
impl_expr_unary!(RecipExpr);

impl_expr_binary!(AddExpr, true);
impl_expr_binary!(SubExpr, false);
impl_expr_binary!(MulExpr, true);

//--------------------------------------------------------------------------------------------------

pub struct KernelCall<'a, E: const ExprTrait + ExprToDyn>
where
	[(); E::ELEMWISE_COUNT]:,
	[(); E::REDUCE_COUNT]:,
	[(); E::SCALAR_COUNT]:,
{
	elem_args: [&'a Tensor; E::ELEMWISE_COUNT],
	reduce_args: [&'a Tensor; E::REDUCE_COUNT],
	scalar_args: [f64; E::SCALAR_COUNT],
}

impl<'a, E: const ExprTrait + ExprToDyn + Copy> KernelCall<'a, E>
where
	[(); E::ELEMWISE_COUNT]:,
	[(); E::REDUCE_COUNT]:,
	[(); E::SCALAR_COUNT]:,
{
	const fn mask_to_indexes<const N: usize>(mut mask: u64) -> [usize; N] {
		assert!(mask.count_ones() as usize == N);
		let mut indexes = [0_usize; N];
		let mut i = 0;
		while i < N {
			let idx = mask.trailing_zeros() as usize;
			indexes[i] = idx;
			mask &= !(1 << idx);
			i += 1;
		}
		indexes
	}

	#[inline]
	pub fn new<const TC: usize, const SC: usize>(
		tensors: [&'a Tensor; TC],
		scalars: [f64; SC],
		_expr: Expr<E>,
	) -> Self {
		let elem_arg_indexes: [usize; E::ELEMWISE_COUNT] =
			const { Self::mask_to_indexes(E::ELEMWISE_MASK) };
		let reduce_arg_indexes: [usize; E::REDUCE_COUNT] =
			const { Self::mask_to_indexes(E::REDUCE_MASK) };
		let scalar_arg_indexes: [usize; E::SCALAR_COUNT] =
			const { Self::mask_to_indexes(E::SCALAR_MASK) };
		Self {
			elem_args: std::array::from_fn(|i| tensors[elem_arg_indexes[i]]),
			reduce_args: std::array::from_fn(|i| tensors[reduce_arg_indexes[i]]),
			scalar_args: std::array::from_fn(|i| scalars[scalar_arg_indexes[i]]),
		}
	}
}

impl<'a, E: const ExprTrait + ExprToDyn> EvaluatesToTensor for KernelCall<'a, E>
where
	[(); 1 - E::REDUCE_OP_COUNT]:,
	[(); E::ELEMWISE_COUNT]:,
	[(); E::REDUCE_COUNT]:,
	[(); E::SCALAR_COUNT]:,
	[(); 1 + E::ELEMWISE_COUNT + E::REDUCE_COUNT]:,
	[(); E::PADDED_KEY_LEN]:,
	[(); E::BATCHED_KEY_LEN]:,
{
	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {
		to.vmt().kernel_runner().run(to, self.elem_args, self.reduce_args, self.scalar_args)
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Expr<E: const ExprTrait + ExprToDyn + Copy>(pub E);

impl<E: const ExprTrait + ExprToDyn + Copy> Expr<E> {
	/// Computes the sum along the last dimension.
	///
	/// This is equivalent to PyTorch's `tensor.sum(dim=-1, keepdim=True)`.
	pub const fn sum(self) -> Expr<SumExpr<E>> {
		Expr(SumExpr(self.0))
	}

	/// Computes the maximum along the last dimension.
	///
	/// This is equivalent to PyTorch's `tensor.max(dim=-1, keepdim=True)`.
	pub const fn max(self) -> Expr<MaxExpr<E>> {
		Expr(MaxExpr(self.0))
	}

	pub const fn exp(self) -> Expr<ExpExpr<E>> {
		Expr(ExpExpr(self.0))
	}

	pub const fn ln(self) -> Expr<LnExpr<E>> {
		Expr(LnExpr(self.0))
	}

	pub const fn abs(self) -> Expr<AbsExpr<E>> {
		Expr(AbsExpr(self.0))
	}

	pub const fn sqrt(self) -> Expr<SqrtExpr<E>> {
		Expr(SqrtExpr(self.0))
	}

	pub const fn recip(self) -> Expr<RecipExpr<E>> {
		Expr(RecipExpr(self.0))
	}
}

impl<E: const ExprTrait + ExprToDyn + Copy> std::ops::Neg for Expr<E> {
	type Output = Expr<NegExpr<E>>;

	fn neg(self) -> Expr<NegExpr<E>> {
		Expr(NegExpr(self.0))
	}
}

impl<A: const ExprTrait + ExprToDyn + Copy, E: const ExprTrait + ExprToDyn + Copy>
	std::ops::Add<Expr<E>> for Expr<A>
{
	type Output = Expr<AddExpr<A, E>>;

	fn add(self, rhs: Expr<E>) -> Expr<AddExpr<A, E>> {
		Expr(AddExpr(self.0, rhs.0))
	}
}

impl<A: const ExprTrait + ExprToDyn + Copy, E: const ExprTrait + ExprToDyn + Copy>
	std::ops::Sub<Expr<E>> for Expr<A>
{
	type Output = Expr<SubExpr<A, E>>;

	fn sub(self, rhs: Expr<E>) -> Expr<SubExpr<A, E>> {
		Expr(SubExpr(self.0, rhs.0))
	}
}

impl<A: const ExprTrait + ExprToDyn + Copy, E: const ExprTrait + ExprToDyn + Copy>
	std::ops::Mul<Expr<E>> for Expr<A>
{
	type Output = Expr<MulExpr<A, E>>;

	fn mul(self, rhs: Expr<E>) -> Expr<MulExpr<A, E>> {
		Expr(MulExpr(self.0, rhs.0))
	}
}

//--------------------------------------------------------------------------------------------------
