//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::collections::HashMap;
use std::hint::cold_path;
use std::rc::Rc;
use std::sync::Arc;

use const_siphasher::sip::SipHasher13;

use crate::ErrPack;
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ExprDiscriminant {
	TensorArg,
	ScalarArg,

	DotExpr,

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
	TensorArg(usize),
	ScalarArg(usize),

	DotExpr(Rc<DynExpr>, Rc<DynExpr>),

	SigmoidExpr(Rc<DynExpr>),
	SwishExpr(Rc<DynExpr>),
	SqrtExpr(Rc<DynExpr>),
	RecipExpr(Rc<DynExpr>, Rc<DynExpr>),
	LnClampedExpr(Rc<DynExpr>),
	AddExpr(Rc<DynExpr>, Rc<DynExpr>),
	MulExpr(Rc<DynExpr>, Rc<DynExpr>),
}

//--------------------------------------------------------------------------------------------------

#[const_trait]
pub trait Expr {
	const CONST: bool;
	const TENSOR_COUNT: usize;
	const REDUCE_COUNT: usize;
	const SCALAR_COUNT: usize;
	const ID_LEN: usize;

	fn const_id(i: usize) -> ExprDiscriminant;
}

pub struct TensorArg<'a>(pub &'a Tensor);
pub struct ScalarArg(pub f64);

pub struct SumExpr<A: Expr>(pub A);

pub struct SigmoidExpr<A: Expr>(pub A);
pub struct SwishExpr<A: Expr>(pub A);
pub struct SqrtExpr<A: Expr>(pub A);
pub struct RecipExpr<A: Expr, B: Expr>(pub A, pub B);
pub struct LnClampedExpr<A: Expr>(pub A);
pub struct AddExpr<A: Expr, B: Expr>(pub A, pub B);
pub struct MulExpr<A: Expr, B: Expr>(pub A, pub B);

//--------------------------------------------------------------------------------------------------

impl<'a> const Expr for TensorArg<'a> {
	const CONST: bool = true;
	const TENSOR_COUNT: usize = 1;
	const REDUCE_COUNT: usize = 0;
	const SCALAR_COUNT: usize = 0;
	const ID_LEN: usize = 1;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 { ExprDiscriminant::TensorArg } else { ExprDiscriminant::Invalid }
	}
}
impl const Expr for ScalarArg {
	const CONST: bool = true;
	const TENSOR_COUNT: usize = 0;
	const REDUCE_COUNT: usize = 0;
	const SCALAR_COUNT: usize = 1;
	const ID_LEN: usize = 1;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 { ExprDiscriminant::ScalarArg } else { ExprDiscriminant::Invalid }
	}
}

impl<A: const Expr> const Expr for SumExpr<A> {
	const CONST: bool = A::CONST;
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::TENSOR_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const ID_LEN: usize = 1 + A::ID_LEN;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 { ExprDiscriminant::DotExpr } else { A::const_id(i - 1) }
	}
}

impl<A: const Expr> const Expr for SigmoidExpr<A> {
	const CONST: bool = A::CONST;
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const ID_LEN: usize = 1 + A::ID_LEN;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 { ExprDiscriminant::SigmoidExpr } else { A::const_id(i - 1) }
	}
}
impl<A: const Expr> const Expr for SwishExpr<A> {
	const CONST: bool = A::CONST;
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const ID_LEN: usize = 1 + A::ID_LEN;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 { ExprDiscriminant::SwishExpr } else { A::const_id(i - 1) }
	}
}
impl<A: const Expr> const Expr for SqrtExpr<A> {
	const CONST: bool = A::CONST;
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const ID_LEN: usize = 1 + A::ID_LEN;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 { ExprDiscriminant::SqrtExpr } else { A::const_id(i - 1) }
	}
}
impl<A: const Expr, B: const Expr> const Expr for RecipExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const TENSOR_COUNT: usize = A::TENSOR_COUNT + B::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const ID_LEN: usize = 1 + A::ID_LEN + B::ID_LEN;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 {
			ExprDiscriminant::RecipExpr
		} else if i < 1 + A::ID_LEN {
			A::const_id(i - 1)
		} else {
			B::const_id(i - 1 - A::ID_LEN)
		}
	}
}

impl<A: const Expr> const Expr for LnClampedExpr<A> {
	const CONST: bool = A::CONST;
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;
	const ID_LEN: usize = 1 + A::ID_LEN;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 { ExprDiscriminant::LnClampedExpr } else { A::const_id(i - 1) }
	}
}
impl<A: const Expr, B: const Expr> const Expr for AddExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const TENSOR_COUNT: usize = A::TENSOR_COUNT + B::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const ID_LEN: usize = 1 + A::ID_LEN + B::ID_LEN;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 {
			ExprDiscriminant::AddExpr
		} else if i < 1 + A::ID_LEN {
			A::const_id(i - 1)
		} else {
			B::const_id(i - 1 - A::ID_LEN)
		}
	}
}
impl<A: const Expr, B: const Expr> const Expr for MulExpr<A, B> {
	const CONST: bool = A::CONST && B::CONST;
	const TENSOR_COUNT: usize = A::TENSOR_COUNT + B::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;
	const ID_LEN: usize = 1 + A::ID_LEN + B::ID_LEN;

	fn const_id(i: usize) -> ExprDiscriminant {
		if i < 1 {
			ExprDiscriminant::MulExpr
		} else if i < 1 + A::ID_LEN {
			A::const_id(i - 1)
		} else {
			B::const_id(i - 1 - A::ID_LEN)
		}
	}
}

//--------------------------------------------------------------------------------------------------

const fn const_id<E: const Expr>() -> ([ExprDiscriminant; E::ID_LEN.next_multiple_of(8)], u64) {
	let mut id = [ExprDiscriminant::Invalid; E::ID_LEN.next_multiple_of(8)];
	let mut i = 0;
	while i < E::ID_LEN {
		id[i] = E::const_id(i);
		i += 1;
	}

	let id_bytes: &[u8] = unsafe {
		std::slice::from_raw_parts(
			id.as_ptr().cast(),
			id.len() * std::mem::size_of::<ExprDiscriminant>(),
		)
	};

	#[allow(clippy::inconsistent_digit_grouping)]
	#[allow(clippy::large_digit_groups)]
	let hasher = SipHasher13::new_with_keys(3141_5926_5358_9793_u64, 2384_6264_3383_2795_u64);
	(id, hasher.hash(id_bytes))
}

//--------------------------------------------------------------------------------------------------

struct Key {
	id: Box<[ExprDiscriminant]>,
	hash: u64,
}

pub struct KernelData {
	id: usize,
	expr_id: Box<[ExprDiscriminant]>,
	expr_id_hash: u64,
	expr: Rc<DynExpr>,
}

pub struct KernelCache {
	kernels: Vec<Arc<KernelData>>,
}

impl KernelCache {
	fn find(&self, id: &[ExprDiscriminant], id_hash: u64) -> Result<&KernelData, usize> {
		let t = self.kernels.binary_search_by(|x| {
			let x = x.as_ref();
			match x.expr_id_hash.cmp(&id_hash) {
				std::cmp::Ordering::Equal => match x.expr_id.len().cmp(&id.len()) {
					std::cmp::Ordering::Equal => x.expr_id.as_ref().cmp(id),
					other => other,
				},
				other => other,
			}
		});
		match t {
			Ok(idx) => Ok(unsafe { self.kernels.get_unchecked(idx) }),
			Err(idx) => Err(idx),
		}
	}

	pub fn run<E: const Expr>(
		&self,
		output: &Tensor,
		expr: &E,
	) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); E::TENSOR_COUNT]:,
		[(); E::REDUCE_COUNT]:,
		[(); E::SCALAR_COUNT]:,
		[(); E::ID_LEN.next_multiple_of(8)]:,
	{
		if !E::CONST {
			todo!("Handle non-constant exprs");
		}
		let (id, id_hash) = const { const_id::<E>() };
		match self.find(&id, id_hash) {
			Ok(kernel) => {
				let a = [0_usize; E::TENSOR_COUNT];
				let r = [0_usize; E::REDUCE_COUNT];
				let c = [0_f64; E::SCALAR_COUNT];
				Ok(())
			},
			Err(idx) => {
				cold_path();
				todo!("Create new kernel at index {idx}");
			},
		}
	}
}
