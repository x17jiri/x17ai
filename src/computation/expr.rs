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
use crate::util::union_find::UnionFind;

//--------------------------------------------------------------------------------------------------

pub struct Node<'a> {
	pub expr: &'a Expr,
	pub parents: Vec<NodeIndex>,
	pub children: Vec<NodeIndex>,
	pub capture: Vec<Rc<ExprTensorRef>>,

	pub scalar_used_by_nonscalar: bool,

	pub shape_group: usize,
	pub out_shape: Vec<usize>,
	pub out_is_scalar: bool,
}

impl<'a> Node<'a> {
	pub fn graphviz_label(&self) -> String {
		match self.expr {
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					if let Some(name) = &tensor_ref.name {
						format!("Tensor\\n'{}'", name)
					} else {
						format!("Tensor\\n{:?}", std::ptr::from_ref(tensor_ref.as_ref()))
					}
				},
				ExprInput::Scalar(scalar_ref) => {
					if let Some(name) = &scalar_ref.name {
						format!("Scalar\\n'{}'", name)
					} else {
						format!("Scalar\\n{:?}", std::ptr::from_ref(scalar_ref.as_ref()))
					}
				},
			},
			Expr::Capture(..) | Expr::First(..) => {
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
				ExprBinaryKind::Add => "Add".to_string(),
				ExprBinaryKind::Sub => "Sub".to_string(),
				ExprBinaryKind::Mul => "Mul".to_string(),
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

	pub fn is_reduction(&self) -> bool {
		matches!(self.expr, Expr::Reduction(_))
	}

	/// `fragment_head` is a node whose result we may have to store into a tensor.
	//	#[allow(clippy::nonminimal_bool)]
	#[rustfmt::skip]
	pub fn is_fragment_head(&self) -> bool {
		self.is_reduction()
		|| !self.is_input()
			&& (
				self.parents.len() != 1 // TODO - check for duplicate parents
				|| !self.capture.is_empty()
				|| self.scalar_used_by_nonscalar
			)
	}

	pub fn out_shape_as_str(&self) -> String {
		let dims: Vec<String> = self.out_shape.iter().map(|d| d.to_string()).collect();
		format!("[{}]", dims.join(", "))
	}
}

#[derive(Clone)]
pub struct MergeGroup {
	pub tensors: Vec<Rc<ExprTensorRef>>,
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIndex(usize);

pub struct NodeVec<'a> {
	vec: Vec<Node<'a>>,
	roots: Vec<NodeIndex>,
	merge_groups: Vec<MergeGroup>,
}

impl<'a> NodeVec<'a> {
	pub fn with_capacity(capacity: usize) -> Self {
		Self {
			vec: Vec::with_capacity(capacity),
			roots: Vec::new(),
			merge_groups: Vec::new(),
		}
	}

	pub fn add(&mut self, node: Node<'a>) -> NodeIndex {
		let index = NodeIndex(self.vec.len());
		self.vec.push(node);
		index
	}

	pub fn get(&self, index: NodeIndex) -> &Node<'a> {
		&self.vec[index.0]
	}

	pub fn new_from_expr(expr: &'a Expr) -> NodeVec<'a> {
		let mut processed = std::collections::HashMap::new();
		let mut nodes = Self::with_capacity(32);
		let mut roots = Vec::with_capacity(1);
		let root = Self::__new_from_expr(&mut processed, &mut nodes, expr, &mut roots);
		roots.push(root);
		roots.sort();
		roots.dedup();
		roots.retain(|&r| nodes[r].parents.is_empty());
		nodes.roots = roots;
		nodes
	}

	pub fn root_cnt(&self) -> usize {
		self.roots.len()
	}

	pub fn root(&self, i: usize) -> NodeIndex {
		self.roots[i]
	}

	pub fn __new_from_expr(
		processed: &mut std::collections::HashMap<ExprRef<'a>, NodeIndex>,
		nodes: &mut NodeVec<'a>,
		expr: &'a Expr,
		roots: &mut Vec<NodeIndex>,
		uf: &mut UnionFind,
	) -> NodeIndex {
		fn add_child<'a>(nodes: &mut NodeVec<'a>, parent: NodeIndex, child: NodeIndex) {
			nodes[parent].children.push(child);
			nodes[child].parents.push(parent);
		}

		let expr_ref = ExprRef { expr };
		if let Some(index) = processed.get(&expr_ref) {
			return *index;
		}

		match expr {
			Expr::Capture(capture) => {
				let child = Self::__new_from_expr(processed, nodes, capture.expr.as_ref(), roots);
				nodes[child].capture.push(capture.tensor_ref.clone());
				processed.insert(expr_ref, child);
				return child;
			},
			Expr::First(first) => {
				let first_child =
					Self::__new_from_expr(processed, nodes, first.lhs.as_ref(), roots);
				let second_child =
					Self::__new_from_expr(processed, nodes, first.rhs.as_ref(), roots);
				processed.insert(expr_ref, first_child);
				roots.push(second_child);
				return first_child;
			},
			_ => {},
		};

		let index = nodes.add(Node {
			expr,
			parents: Vec::new(),
			children: Vec::new(),
			capture: Vec::new(),
			scalar_used_by_nonscalar: false,
			shape_group: usize::MAX,
			out_shape: Vec::new(),
			out_is_scalar: false,
		});
		processed.insert(expr_ref, index);
		match expr {
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					nodes[index].shape_group = uf.add();
					nodes[index].out_shape.clone_from(&tensor_ref.shape);
				},
				ExprInput::Scalar(..) => {
					nodes[index].out_is_scalar = true;
				},
			},
			Expr::Capture(..) | Expr::First(..) => {
				unreachable!() // handled above
			},
			Expr::Cast(cast) => {
				let child = Self::__new_from_expr(processed, nodes, cast.expr.as_ref(), roots, uf);
				nodes[index].out_is_scalar = nodes[child].out_is_scalar;
				nodes[index].out_shape.clone_from(&nodes[child].out_shape);
				nodes[index].shape_group = nodes[child].shape_group;
				add_child(nodes, index, child);
			},
			Expr::Unary(unary) => {
				let child = Self::__new_from_expr(processed, nodes, unary.expr.as_ref(), roots, uf);
				nodes[index].out_is_scalar = nodes[child].out_is_scalar;
				nodes[index].out_shape.clone_from(&nodes[child].out_shape);
				nodes[index].shape_group = nodes[child].shape_group;
				add_child(nodes, index, child);
			},
			Expr::Binary(binary) => {
				let left_child =
					Self::__new_from_expr(processed, nodes, binary.lhs.as_ref(), roots, uf);
				add_child(nodes, index, left_child);
				let right_child =
					Self::__new_from_expr(processed, nodes, binary.rhs.as_ref(), roots, uf);
				add_child(nodes, index, right_child);
				let out_is_scalar =
					nodes[left_child].out_is_scalar && nodes[right_child].out_is_scalar;
				nodes[index].out_is_scalar = out_is_scalar;
				if !out_is_scalar {
					// TODO - out_shape, shape_group
				}
			},
			Expr::Reduction(reduction) => {
				let child =
					Self::__new_from_expr(processed, nodes, reduction.expr.as_ref(), roots, uf);
				add_child(nodes, index, child);
			},
		}
		index
	}
}

impl<'a> std::ops::Index<NodeIndex> for NodeVec<'a> {
	type Output = Node<'a>;

	fn index(&self, index: NodeIndex) -> &Node<'a> {
		&self.vec[index.0]
	}
}

impl<'a> std::ops::IndexMut<NodeIndex> for NodeVec<'a> {
	fn index_mut(&mut self, index: NodeIndex) -> &mut Node<'a> {
		&mut self.vec[index.0]
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
	First(ExprFirst),
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
	pub shape: Vec<usize>,
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

pub struct ExprFirst {
	pub lhs: RcExpr,
	pub rhs: RcExpr,
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
	pub fn new(
		name: Option<Cow<'static, str>>,
		dtype: DType,
		shape: Vec<usize>,
	) -> Rc<ExprTensorRef> {
		Rc::new(ExprTensorRef {
			tensor: RefCell::new(None),
			dtype,
			shape,
			name,
		})
	}
}

impl ExprScalarRef {
	pub fn new(name: Option<Cow<'static, str>>) -> Rc<ExprScalarRef> {
		Rc::new(ExprScalarRef { value: RefCell::new(None), name })
	}
}

//--------------------------------------------------------------------------------------------------

impl RcExpr {
	pub fn new_tensor_input(tensor_ref: Rc<ExprTensorRef>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Input(ExprInput::Tensor(tensor_ref))),
		}
	}

	pub fn new_scalar_input(scalar_ref: Rc<ExprScalarRef>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Input(ExprInput::Scalar(scalar_ref))),
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
			rc_expr: Rc::new(Expr::First(ExprFirst { lhs: first, rhs: second })),
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

pub fn __calc_shape_groups(nodes: &mut NodeVec, node: NodeIndex, uf: &mut UnionFind) -> usize {
	let out_shape_group = nodes[node].out_shape_group;
	if out_shape_group != usize::MAX {
		return out_shape_group; // already visited
	}
	match &nodes[node].expr {
		Expr::Input(input) => match input {
			ExprInput::Tensor(..) => {
				unreachable!(); // shape group of tensor inputs should be set already
			},
			ExprInput::Scalar(..) => usize::MAX - 1,
		},
		Expr::Capture(..) | Expr::Cast(..) | Expr::Unary(..) => {
			assert!(nodes[node].children.len() == 1);
			let g = __calc_shape_groups(nodes, nodes[node].children[0], uf);
			nodes[node].op_shape_group = g;
			nodes[node].out_shape_group = g;
			if g < uf.size() {
				nodes[node].out_shape = nodes[nodes[node].children[0]].out_shape.clone();
			}
			g
		},
		Expr::Binary(..) => {
			assert!(nodes[node].children.len() == 2);
			let c1 = nodes[node].children[0];
			let c2 = nodes[node].children[1];
			let g1 = __calc_shape_groups(nodes, c1, uf);
			let g2 = __calc_shape_groups(nodes, c2, uf);
			let g = if g2 >= uf.size() {
				nodes[c2].scalar_used_by_nonscalar = true;
				if g1 < uf.size() {
					nodes[node].out_shape = nodes[c1].out_shape.clone();
				}
				g1
			} else if g1 >= uf.size() {
				nodes[c1].scalar_used_by_nonscalar = true;
				if g2 < uf.size() {
					nodes[node].out_shape = nodes[c2].out_shape.clone();
				}
				g2
			} else {
				let l1 = nodes[c1].out_shape.len();
				let l2 = nodes[c2].out_shape.len();
				for i in (1..=l1.min(l2)).rev() {
					let d1 = nodes[c1].out_shape[l1 - i];
					let d2 = nodes[c2].out_shape[l2 - i];
					if d1 != d2 {
						break;
					}
					nodes[node].out_shape.push(d1);
				}
				nodes[node].out_shape.reverse();
				uf.union(g1, g2)
			};
			nodes[node].op_shape_group = g;
			nodes[node].out_shape_group = g;
			g
		},
		Expr::First(..) => {
			unreachable!();
		},
		Expr::Reduction(..) => {
			assert!(nodes[node].children.len() == 1);
			let g1 = __calc_shape_groups(nodes, nodes[node].children[0], uf);
			let g = usize::MAX - 1;
			nodes[node].op_shape_group = g1;
			nodes[node].out_shape_group = g;
			g
		},
	}
}

pub fn calc_shape_groups(nodes: &mut NodeVec) {
	let mut tensor_ref_map = std::collections::HashMap::new(); // *ExprTensorRef -> Index
	let mut tensor_ref_vec = Vec::new(); // Index -> Rc<ExprTensorRef>
	for node in &mut nodes.vec {
		if let Expr::Input(ExprInput::Tensor(tensor_ref)) = &node.expr {
			node.out_shape.clone_from(&tensor_ref.shape);
			let tensor_ref = tensor_ref.clone();
			let index = match tensor_ref_map.entry(std::ptr::from_ref(tensor_ref.as_ref())) {
				std::collections::hash_map::Entry::Vacant(entry) => {
					let index = tensor_ref_vec.len();
					entry.insert(index);
					tensor_ref_vec.push(tensor_ref);
					index
				},
				std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
			};
			node.op_shape_group = index;
			node.out_shape_group = index;
		}
	}
	let mut uf = UnionFind::new(tensor_ref_vec.len());
	for root in 0..nodes.root_cnt() {
		__calc_shape_groups(nodes, nodes.root(root), &mut uf);
	}
	let (compact_ids, sets_cnt) = uf.compact_ids();
	nodes.merge_groups = vec![MergeGroup { tensors: Vec::new() }; sets_cnt];
	for (i, tensor_ref) in tensor_ref_vec.into_iter().enumerate() {
		let group_id = compact_ids[i];
		nodes.merge_groups[group_id].tensors.push(tensor_ref);
	}
	for node in &mut nodes.vec {
		if node.op_shape_group < compact_ids.len() {
			node.op_shape_group = compact_ids[node.op_shape_group];
		}
		if node.out_shape_group < compact_ids.len() {
			node.out_shape_group = compact_ids[node.out_shape_group];
		}
	}
}

pub fn compile(expr: &Expr) {
	let mut nodes = NodeVec::new_from_expr(expr);
	calc_shape_groups(&mut nodes);
}

pub fn print_graphviz<'a, W: std::fmt::Write>(w: &mut W, nodes: &NodeVec<'a>) -> std::fmt::Result {
	writeln!(w, "digraph G {{")?;
	writeln!(w, "\trankdir=BT;")?;
	for (i, node) in nodes.vec.iter().enumerate() {
		writeln!(w, "\t\t{} [label=\"{}\"];", i, node.graphviz_label())?;
		if node.is_fragment_head() {
			writeln!(w, "\t{} [style=filled, fillcolor=\"#ffcccc\"];", i)?;
		}
		for &child_index in &node.children {
			let label = nodes[child_index].out_shape_as_str();
			writeln!(w, "\t{} -> {} [label=\"{}\"];", child_index.0, i, label)?;
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
