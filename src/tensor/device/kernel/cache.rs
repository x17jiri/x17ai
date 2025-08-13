//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

use crate::ErrPack;
use crate::tensor::{Tensor, TensorOpError};

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

pub trait Expr {
	const TENSOR_COUNT: usize;
	const REDUCE_COUNT: usize;
	const SCALAR_COUNT: usize;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node>;
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

impl<'a> Expr for TensorArg<'a> {
	const TENSOR_COUNT: usize = 1;
	const REDUCE_COUNT: usize = 0;
	const SCALAR_COUNT: usize = 0;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::TensorArg)
	}
}
impl Expr for ScalarArg {
	const TENSOR_COUNT: usize = 0;
	const REDUCE_COUNT: usize = 0;
	const SCALAR_COUNT: usize = 1;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::ScalarArg)
	}
}

impl<A: Expr> Expr for SumExpr<A> {
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::TENSOR_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::DotExpr).and_then(|n| self.0.find_in(n))
	}
}

impl<A: Expr> Expr for SigmoidExpr<A> {
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::SigmoidExpr).and_then(|n| self.0.find_in(n))
	}
}
impl<A: Expr> Expr for SwishExpr<A> {
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::SwishExpr).and_then(|n| self.0.find_in(n))
	}
}
impl<A: Expr> Expr for SqrtExpr<A> {
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::SqrtExpr).and_then(|n| self.0.find_in(n))
	}
}
impl<A: Expr, B: Expr> Expr for RecipExpr<A, B> {
	const TENSOR_COUNT: usize = A::TENSOR_COUNT + B::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::RecipExpr)
			.and_then(|n| self.0.find_in(n))
			.and_then(|n| self.1.find_in(n))
	}
}
impl<A: Expr> Expr for LnClampedExpr<A> {
	const TENSOR_COUNT: usize = A::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::LnClampedExpr).and_then(|n| self.0.find_in(n))
	}
}
impl<A: Expr, B: Expr> Expr for AddExpr<A, B> {
	const TENSOR_COUNT: usize = A::TENSOR_COUNT + B::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::AddExpr)
			.and_then(|n| self.0.find_in(n))
			.and_then(|n| self.1.find_in(n))
	}
}
impl<A: Expr, B: Expr> Expr for MulExpr<A, B> {
	const TENSOR_COUNT: usize = A::TENSOR_COUNT + B::TENSOR_COUNT;
	const REDUCE_COUNT: usize = A::REDUCE_COUNT + B::REDUCE_COUNT;
	const SCALAR_COUNT: usize = A::SCALAR_COUNT + B::SCALAR_COUNT;

	fn find_in<'n>(&self, node: &'n Node) -> Option<&'n Node> {
		node.next(ExprDiscriminant::MulExpr)
			.and_then(|n| self.0.find_in(n))
			.and_then(|n| self.1.find_in(n))
	}
}

//--------------------------------------------------------------------------------------------------

pub struct KernelData {
	id: usize,
	name: String,
	expr: Rc<DynExpr>,
}

pub struct Node {
	children: Vec<(ExprDiscriminant, Box<Node>)>,
	data: Rc<KernelData>,
}

impl Node {
	fn next(&self, discriminant: ExprDiscriminant) -> Option<&Self> {
		self.children.iter().find(|(d, _)| *d == discriminant).map(|(_, child)| child.as_ref())
	}
}

pub struct KernelCache {
	head: Node,
}

impl KernelCache {
	pub fn run<E: Expr>(&self, output: &Tensor, expr: &E) -> Result<(), ErrPack<TensorOpError>>
	where
		[(); E::TENSOR_COUNT]:,
		[(); E::REDUCE_COUNT]:,
		[(); E::SCALAR_COUNT]:,
	{
		let a = [0_usize; E::TENSOR_COUNT];
		let r = [0_usize; E::REDUCE_COUNT];
		let c = [0_f64; E::SCALAR_COUNT];
		Ok(())
	}
}
