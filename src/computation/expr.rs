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
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::missing_panics_doc)]

use std::borrow::Cow;
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

impl<'a> Node<'a> {
	pub fn graphviz_label(&self) -> String {
		match self.expr {
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					if let Some(name) = &tensor_ref.name {
						format!("Tensor '{}'", name)
					} else {
						format!("Tensor: {:?}", std::ptr::from_ref(tensor_ref.as_ref()))
					}
				},
				ExprInput::Scalar(scalar_ref) => {
					if let Some(name) = &scalar_ref.name {
						format!("Scalar: '{}'", name)
					} else {
						format!("Scalar: {:?}", std::ptr::from_ref(scalar_ref.as_ref()))
					}
				},
			},
			Expr::Capture(..) => {
				unreachable!() // `clone_expr()` removes Capture nodes
			},
			Expr::Cast(cast) => format!("Cast to {:?}", cast.dtype),
			Expr::Unary(unary) => match unary.kind {
				ExprUnaryKind::Neg => "Neg".to_string(),
				ExprUnaryKind::Exp => "Exp".to_string(),
				ExprUnaryKind::Ln => "Ln".to_string(),
				ExprUnaryKind::Abs => "Abs".to_string(),
				ExprUnaryKind::Sqrt => "Sqrt".to_string(),
				ExprUnaryKind::Recip => "Recip".to_string(),
			},
			Expr::Binary(binary) => match binary.kind {
				ExprBinaryKind::Add => "+".to_string(),
				ExprBinaryKind::Sub => "-".to_string(),
				ExprBinaryKind::Mul => "*".to_string(),
				ExprBinaryKind::First => "First".to_string(),
			},
			Expr::Reduction(reduction) => match reduction.kind {
				ExprReductionKind::Sum => "Sum".to_string(),
				ExprReductionKind::Max => "Max".to_string(),
			},
		}
	}

	pub fn is_input(&self) -> bool {
		let result = self.children.is_empty();
		debug_assert!(result == matches!(self.expr, Expr::Input(_)));
		result
	}
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

#[derive(Clone)]
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
	pub name: Option<Cow<'static, str>>,
}

pub struct ExprScalarRef {
	pub value: RefCell<Option<f64>>,
	pub name: Option<Cow<'static, str>>,
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

impl ExprTensorRef {
	pub fn new(name: Option<Cow<'static, str>>, dtype: DType) -> Rc<ExprTensorRef> {
		Rc::new(ExprTensorRef { tensor: RefCell::new(None), dtype, name })
	}
}

//--------------------------------------------------------------------------------------------------

impl RcExpr {
	pub fn new_tensor_input(dtype: DType, name: Cow<'static, str>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Input(ExprInput::Tensor(Rc::new(ExprTensorRef {
				tensor: RefCell::new(None),
				dtype,
				name: Some(name),
			})))),
		}
	}

	pub fn new_scalar_input(name: Cow<'static, str>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Input(ExprInput::Scalar(Rc::new(ExprScalarRef {
				value: RefCell::new(None),
				name: Some(name),
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

	pub fn capture(self, tensor_ref: Rc<ExprTensorRef>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Capture(ExprCapture { expr: self, tensor_ref })),
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
) -> NodeIndex {
	fn add_child<'a>(nodes: &mut NodeVec<'a>, parent: NodeIndex, child: NodeIndex) {
		nodes[parent].children.push(child);
		nodes[child].parents.push(parent);
		if nodes[child].parents.len() > 1
			&& !nodes[child].is_input()
			&& nodes[child].parents[0] != parent
		{
			nodes[child].fragment_head = true;
		}
	}

	match processed.entry(ExprRef { expr }) {
		std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
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
					fragment_head: false,
				});
				entry.insert(index);
				match expr {
					Expr::Input(..) => {},
					Expr::Capture(..) => {
						unreachable!() // handled above
					},
					Expr::Cast(cast) => {
						let child = __clone_expr(processed, nodes, cast.expr.as_ref());
						add_child(nodes, index, child);
					},
					Expr::Unary(unary) => {
						let child = __clone_expr(processed, nodes, unary.expr.as_ref());
						add_child(nodes, index, child);
					},
					Expr::Binary(binary) => {
						let left_child = __clone_expr(processed, nodes, binary.lhs.as_ref());
						add_child(nodes, index, left_child);
						let right_child = __clone_expr(processed, nodes, binary.rhs.as_ref());
						add_child(nodes, index, right_child);
					},
					Expr::Reduction(reduction) => {
						nodes[index].fragment_head = true;
						let child = __clone_expr(processed, nodes, reduction.expr.as_ref());
						add_child(nodes, index, child);
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
	let top_node = __clone_expr(&mut processed, &mut nodes, expr);
	assert!(top_node.0 == 0);
	nodes[top_node].fragment_head = true;
	nodes
}

pub fn compile(expr: &Expr) {
	let mut nodes = clone_expr(expr);

	let mut tensor_ref_map = std::collections::HashMap::new(); // *ExprTensorRef -> Index
	let mut tensor_ref_vec = Vec::new(); // Index -> &ExprTensorRef
	for node in nodes.0 {
		if let Expr::Input(ExprInput::Tensor(tensor_ref)) = &node.expr {
			let tensor_ref = tensor_ref.as_ref();
			if let std::collections::hash_map::Entry::Vacant(entry) =
				tensor_ref_map.entry(std::ptr::from_ref(tensor_ref))
			{
				let index = tensor_ref_vec.len();
				entry.insert(index);
				tensor_ref_vec.push(tensor_ref);
			}
		}
	}
}

pub fn print_graphviz<'a, W: std::fmt::Write>(w: &mut W, nodes: &NodeVec<'a>) -> std::fmt::Result {
	writeln!(w, "digraph G {{")?;
	writeln!(w, "\trankdir=LR;")?;
	for (i, node) in nodes.0.iter().enumerate() {
		writeln!(w, "\t{} [label=\"{}\"];", i, node.graphviz_label())?;
		if node.fragment_head {
			writeln!(w, "\t{} [style=filled, fillcolor=\"#ffcccc\"];", i)?;
		}
		for &child_index in &node.children {
			writeln!(w, "\t{} -> {};", child_index.0, i)?;
		}
		for cap in &node.capture {
			let cap_label = if let Some(name) = &cap.name {
				format!("Capture '{}'", name)
			} else {
				format!("Capture {:?}", std::ptr::from_ref(cap.as_ref()))
			};
			writeln!(
				w,
				"\tcap_{} [label=\"{}\", shape=box];",
				std::ptr::from_ref(cap.as_ref()) as usize,
				cap_label
			)?;
			writeln!(
				w,
				"\t{} -> cap_{} [style=dashed];",
				i,
				std::ptr::from_ref(cap.as_ref()) as usize,
			)?;
		}
	}
	writeln!(w, "}}")?;
	Ok(())
}

//--------------------------------------------------------------------------------------------------
