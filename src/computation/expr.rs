//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#![allow(clippy::use_self)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::implicit_hasher)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::tensor::{DType, Tensor};

//--------------------------------------------------------------------------------------------------

pub struct Node<'a> {
	pub expr: &'a Expr,
	pub parents: Vec<NodeIndex>,
	pub children: Vec<NodeIndex>,
	pub capture: Vec<Rc<ExprTensorRef>>,

	/// `fragment_head` is a node whose result we may have to store into a tensor.
	pub fragment_head: bool,
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeIndex(usize);

pub struct NodeVec<'a>(Vec<Node<'a>>);

impl<'a> NodeVec<'a> {
	pub fn with_capacity(capacity: usize) -> Self {
		Self(Vec::with_capacity(capacity))
	}

	pub fn add(&mut self, node: Node<'a>) -> NodeIndex {
		let index = NodeIndex(self.0.len());
		self.0.push(node);
		index
	}

	pub fn get(&self, index: NodeIndex) -> &Node<'a> {
		&self.0[index.0]
	}
}

impl<'a> std::ops::Index<NodeIndex> for NodeVec<'a> {
	type Output = Node<'a>;

	fn index(&self, index: NodeIndex) -> &Node<'a> {
		&self.0[index.0]
	}
}

impl<'a> std::ops::IndexMut<NodeIndex> for NodeVec<'a> {
	fn index_mut(&mut self, index: NodeIndex) -> &mut Node<'a> {
		&mut self.0[index.0]
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RcExpr {
	pub rc_expr: Rc<Expr>,
}

/// `ExprRef` is used instead of `&Expr` in places that require
/// `Eq` and `Hash` based on pointer equality.
#[derive(Clone, Copy)]
pub struct ExprRef<'a> {
	pub expr: &'a Expr,
}

impl<'a> PartialEq for ExprRef<'a> {
	fn eq(&self, other: &Self) -> bool {
		std::ptr::eq(self.expr, other.expr)
	}
}

impl<'a> Eq for ExprRef<'a> {}

impl<'a> std::hash::Hash for ExprRef<'a> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		std::ptr::from_ref(self.expr).hash(state);
	}
}

pub enum Expr {
	Input(ExprInput),
	Capture(ExprCapture),
	Cast(ExprCast),
	Unary(ExprUnary),
	Binary(ExprBinary),
	Reduction(ExprReduction),
}

pub enum ExprInput {
	Tensor(Rc<ExprTensorRef>),
	Scalar(Rc<ExprScalarRef>),
}

// The tensor may be replaced before running the computation,
// but the dtype needs to be correct.
pub struct ExprTensorRef {
	pub tensor: RefCell<Option<Tensor>>,
	pub dtype: DType,
}

pub struct ExprScalarRef {
	pub value: RefCell<Option<f64>>,
}

pub struct ExprCapture {
	pub expr: RcExpr,
	pub tensor_ref: Rc<ExprTensorRef>,
}

pub struct ExprCast {
	pub expr: RcExpr,
	pub dtype: DType,
}

pub struct ExprUnary {
	pub kind: ExprUnaryKind,
	pub expr: RcExpr,
}

pub enum ExprUnaryKind {
	Neg,
	Exp,
	Ln,
	Abs,
	Sqrt,
	Recip,
}

pub struct ExprBinary {
	pub kind: ExprBinaryKind,
	pub lhs: RcExpr,
	pub rhs: RcExpr,
}

pub enum ExprBinaryKind {
	Add,
	Sub,
	Mul,

	First,
}

pub struct ExprReduction {
	pub kind: ExprReductionKind,
	pub expr: RcExpr,
}

pub enum ExprReductionKind {
	Sum,
	Max,
}

//--------------------------------------------------------------------------------------------------

impl RcExpr {
	pub fn new_tensor_input(dtype: DType) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Input(ExprInput::Tensor(Rc::new(ExprTensorRef {
				tensor: RefCell::new(None),
				dtype,
			})))),
		}
	}

	pub fn new_scalar_input() -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Input(ExprInput::Scalar(Rc::new(ExprScalarRef {
				value: RefCell::new(None),
			})))),
		}
	}

	pub fn as_ref(&self) -> &Expr {
		&self.rc_expr
	}

	pub fn cast(self, dtype: DType) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Cast(ExprCast { expr: self, dtype })),
		}
	}

	pub fn exp(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary { kind: ExprUnaryKind::Exp, expr: self })),
		}
	}

	pub fn ln(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary { kind: ExprUnaryKind::Ln, expr: self })),
		}
	}

	pub fn abs(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary { kind: ExprUnaryKind::Abs, expr: self })),
		}
	}

	pub fn sqrt(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary { kind: ExprUnaryKind::Sqrt, expr: self })),
		}
	}

	pub fn recip(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary { kind: ExprUnaryKind::Recip, expr: self })),
		}
	}

	pub fn sum(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Reduction(ExprReduction {
				kind: ExprReductionKind::Sum,
				expr: self,
			})),
		}
	}

	pub fn max(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Reduction(ExprReduction {
				kind: ExprReductionKind::Max,
				expr: self,
			})),
		}
	}

	pub fn first(first: RcExpr, second: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Binary(ExprBinary {
				kind: ExprBinaryKind::First,
				lhs: first,
				rhs: second,
			})),
		}
	}
}

impl std::ops::Add for RcExpr {
	type Output = RcExpr;

	fn add(self, rhs: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Binary(ExprBinary {
				kind: ExprBinaryKind::Add,
				lhs: self,
				rhs,
			})),
		}
	}
}

impl std::ops::Sub for RcExpr {
	type Output = RcExpr;

	fn sub(self, rhs: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Binary(ExprBinary {
				kind: ExprBinaryKind::Sub,
				lhs: self,
				rhs,
			})),
		}
	}
}

impl std::ops::Mul for RcExpr {
	type Output = RcExpr;

	fn mul(self, rhs: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Binary(ExprBinary {
				kind: ExprBinaryKind::Mul,
				lhs: self,
				rhs,
			})),
		}
	}
}

impl std::ops::Neg for RcExpr {
	type Output = RcExpr;

	fn neg(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary { kind: ExprUnaryKind::Neg, expr: self })),
		}
	}
}

//--------------------------------------------------------------------------------------------------

pub fn __clone_expr<'a>(
	processed: &mut std::collections::HashMap<ExprRef<'a>, NodeIndex>,
	nodes: &mut NodeVec<'a>,
	expr: &'a Expr,
	need_result: bool, // TODO - will this work with 2 parents where only one needs the result?
) -> NodeIndex {
	match processed.entry(ExprRef { expr }) {
		std::collections::hash_map::Entry::Occupied(entry) => {
			let index = *entry.get();
			nodes[index].fragment_head = true; // TODO
			index
		},
		std::collections::hash_map::Entry::Vacant(entry) => {
			if let Expr::Capture(capture) = expr {
				let child = __clone_expr(processed, nodes, capture.expr.as_ref());
				nodes[child].capture.push(capture.tensor_ref.clone());
				nodes[child].fragment_head = true;
				child
			} else {
				let index = nodes.add(Node {
					expr,
					parents: Vec::new(),
					children: Vec::new(),
					capture: Vec::new(),
				});
				entry.insert(index);
				match expr {
					Expr::Input(..) | Expr::Capture(..) => {},
					Expr::Cast(cast) => {
						let child = __clone_expr(processed, nodes, cast.expr.as_ref());
						nodes[index].children.push(child);
						nodes[child].parents.push(index);
					},
					Expr::Unary(unary) => {
						let child = __clone_expr(processed, nodes, unary.expr.as_ref());
						nodes[index].children.push(child);
						nodes[child].parents.push(index);
					},
					Expr::Binary(binary) => {
						let left_child = __clone_expr(processed, nodes, binary.lhs.as_ref());
						nodes[index].children.push(left_child);
						nodes[left_child].parents.push(index);
						let right_child = __clone_expr(processed, nodes, binary.rhs.as_ref());
						nodes[index].children.push(right_child);
						nodes[right_child].parents.push(index);
					},
					Expr::Reduction(reduction) => {
						let child = __clone_expr(processed, nodes, reduction.expr.as_ref());
						nodes[index].fragment_head = true;
						nodes[index].children.push(child);
						nodes[child].parents.push(index);
					},
				}
				index
			}
		},
	}
}

pub fn clone_expr<'a>(expr: &'a Expr) -> NodeVec<'a> {
	let mut processed = std::collections::HashMap::new();
	let mut nodes = NodeVec::with_capacity(32);
	__clone_expr(&mut processed, &mut nodes, expr);
	nodes
}

//--------------------------------------------------------------------------------------------------
