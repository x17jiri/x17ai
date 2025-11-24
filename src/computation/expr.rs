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
#![allow(clippy::new_without_default)]

use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use crate::tensor::{DType, Tensor};
use crate::util::union_find::UnionFind;

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct ReductionBitmap {
	bitmap: Vec<u64>,
}

impl ReductionBitmap {
	pub fn new() -> Self {
		Self { bitmap: Vec::new() }
	}

	pub fn union<'a>(mut a: &'a ReductionBitmap, mut b: &'a ReductionBitmap) -> ReductionBitmap {
		if a.bitmap.len() > b.bitmap.len() {
			std::mem::swap(&mut a, &mut b);
		}
		let mut result = ReductionBitmap {
			bitmap: Vec::with_capacity(b.bitmap.len()),
		};
		let ptr = result.bitmap.as_mut_ptr();
		unsafe {
			for i in 0..a.bitmap.len() {
				ptr.add(i).write(*a.bitmap.get_unchecked(i) | *b.bitmap.get_unchecked(i));
			}
			for i in a.bitmap.len()..b.bitmap.len() {
				ptr.add(i).write(*b.bitmap.get_unchecked(i));
			}
			result.bitmap.set_len(b.bitmap.len());
		}
		result
	}

	pub fn clone_and_set(&self, index: usize) -> ReductionBitmap {
		let word_index = index / 64;
		let bit_index = index % 64;
		let min_len = word_index + 1;
		let len = self.bitmap.len().max(min_len);
		let mut result = ReductionBitmap { bitmap: Vec::with_capacity(len) };
		let ptr = result.bitmap.as_mut_ptr();
		unsafe {
			for i in 0..self.bitmap.len() {
				ptr.add(i).write(*self.bitmap.get_unchecked(i));
			}
			for i in self.bitmap.len()..len {
				ptr.add(i).write(0);
			}
			*ptr.add(word_index) |= 1 << bit_index;
			result.bitmap.set_len(len);
		}
		result
	}

	pub fn is_equal(&self, other: &ReductionBitmap) -> bool {
		let (mut a, mut b) = (&self.bitmap[..], &other.bitmap[..]);
		if a.len() > b.len() {
			std::mem::swap(&mut a, &mut b);
		}
		for i in 0..a.len() {
			if a[i] != b[i] {
				return false;
			}
		}
		for b in &b[a.len()..] {
			if *b != 0 {
				return false;
			}
		}
		true
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct DimConstraint {
	pub source: String,
	pub size: usize,
}

#[derive(Clone)]
pub struct ShapeConstraint {
	pub constraint: Vec<Option<DimConstraint>>,
}

impl ShapeConstraint {
	pub fn new() -> Self {
		Self { constraint: Vec::new() }
	}

	pub fn as_str(&self) -> String {
		let dims: Vec<String> = self
			.constraint
			.iter()
			.skip_while(|d| d.is_none())
			.map(|d| match d {
				Some(d) => d.size.to_string(),
				None => "-".to_string(),
			})
			.collect();
		if dims.is_empty() { "[..]".to_string() } else { format!("[..,{}]", dims.join(",")) }
	}

	pub fn last(&self) -> Option<usize> {
		match self.constraint.last() {
			Some(d) => d.as_ref().map(|d| d.size),
			None => None,
		}
	}

	pub fn set_last(&mut self, value: Option<DimConstraint>) {
		if let Some(last) = self.constraint.last_mut() {
			*last = value;
		} else if value.is_some() {
			self.constraint.push(value);
		}
	}

	// Result dimension will be constrained only if both `a` and `b` have the same constraint.
	pub fn intersection(a: &ShapeConstraint, b: &ShapeConstraint) -> ShapeConstraint {
		let mut result = ShapeConstraint::new();
		let (l1, l2) = (a.constraint.len(), b.constraint.len());
		for i in (1..=l1.min(l2)).rev() {
			let d1 = &a.constraint[l1 - i];
			let d2 = &b.constraint[l2 - i];
			result.constraint.push(
				//
				if let Some(d1) = d1 {
					if let Some(d2) = d2 {
						if d1.size == d2.size {
							Some(DimConstraint {
								source: format!("{} & {}", d1.source, d2.source),
								size: d1.size,
							})
						} else {
							None //
						}
					} else {
						Some(d1.clone())
					}
				} else {
					d2.clone()
				},
			);
		}
		result
	}

	// Result dimension will be constrained if either `a` or `b` has the constraint.
	pub fn union(a: &ShapeConstraint, b: &ShapeConstraint) -> ShapeConstraint {
		let mut result = ShapeConstraint::new();
		let (l1, l2) = (a.constraint.len(), b.constraint.len());
		for i in (1..=l1.max(l2)).rev() {
			let d1 = a.constraint.get(l1.wrapping_sub(i)).unwrap_or(&None);
			let d2 = b.constraint.get(l2.wrapping_sub(i)).unwrap_or(&None);
			result.constraint.push(
				//
				if let Some(d1) = d1 {
					if let Some(d2) = d2 {
						if d1.size == d2.size {
							Some(DimConstraint {
								source: format!("{} & {}", d1.source, d2.source),
								size: d1.size,
							})
						} else {
							panic!("ShapeConstraint::union(): conflicting dimension constraints. Dimension -{}. Sizes: {} vs {}. The value {} comes from '{}'; the value {} comes from '{}'.", i, d1.size, d2.size, d1.size, d1.source, d2.size, d2.source);
						}
					} else {
						Some(d1.clone())
					}
				} else {
					d2.clone()
				},
			);
		}
		result
	}
}

//--------------------------------------------------------------------------------------------------

pub struct Node {
	pub expr: Rc<Expr>,
	pub parents: Vec<NodeIndex>,
	pub children: Vec<NodeIndex>,
	pub capture: Vec<Rc<ExprTensorRef>>,

	pub merge_group: usize,
	pub out_shape: ShapeConstraint,
	pub out_is_scalar: bool,
	pub reduction_head: bool,
	pub reduction_bitmap: ReductionBitmap,
	pub fragment: NodeIndex,
}

impl Node {
	pub fn graphviz_label(&self) -> String {
		match self.expr.as_ref() {
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					if let Some(name) = &tensor_ref.name {
						format!("<b>Tensor</b><br/><font color='blue'><b>{}</b></font>", name)
					} else {
						format!(
							"<b>Tensor</b><br/><font color='blue'><b>{:?}</b></font>",
							std::ptr::from_ref(tensor_ref.as_ref())
						)
					}
				},
				ExprInput::Scalar(scalar_ref) => {
					if let Some(name) = &scalar_ref.name {
						format!("<b>Scalar</b><br/><font color='blue'><b>{}</b></font>", name)
					} else {
						format!(
							"<b>Scalar</b><br/><font color='blue'><b>{:?}</b></font>",
							std::ptr::from_ref(scalar_ref.as_ref())
						)
					}
				},
			},
			Expr::Capture(..) | Expr::First(..) => {
				unreachable!() // Compilation removes these nodes nodes
			},
			Expr::Cast(cast) => format!("Cast to {:?}", cast.dtype),
			Expr::Unary(unary) => match unary.kind {
				ExprUnaryKind::Neg => "<b>Neg</b>".to_string(),
				ExprUnaryKind::Exp => "<b>Exp</b>".to_string(),
				ExprUnaryKind::Ln => "<b>Ln</b>".to_string(),
				ExprUnaryKind::Abs => "<b>Abs</b>".to_string(),
				ExprUnaryKind::Sqrt => "<b>Sqrt</b>".to_string(),
				ExprUnaryKind::Recip => "<b>Recip</b>".to_string(),
				ExprUnaryKind::Identity => "<b>Identity</b>".to_string(),
			},
			Expr::Binary(binary) => match binary.kind {
				ExprBinaryKind::Add => "<b>Add</b>".to_string(),
				ExprBinaryKind::Sub => "<b>Sub</b>".to_string(),
				ExprBinaryKind::Mul => "<b>Mul</b>".to_string(),
			},
			Expr::Reduction(reduction) => match reduction.kind {
				ExprReductionKind::Sum => "<b>Sum</b>".to_string(),
				ExprReductionKind::Max => "<b>Max</b>".to_string(),
			},
		}
	}

	pub fn is_input(&self) -> bool {
		let result = self.children.is_empty();
		debug_assert!(result == matches!(self.expr.as_ref(), Expr::Input(_)));
		result
	}

	pub fn is_scalar_input(&self) -> bool {
		matches!(self.expr.as_ref(), Expr::Input(ExprInput::Scalar(_)))
	}

	pub fn is_tensor_input(&self) -> bool {
		matches!(self.expr.as_ref(), Expr::Input(ExprInput::Tensor(_)))
	}

	pub fn is_reduction(&self) -> bool {
		matches!(self.expr.as_ref(), Expr::Reduction(_))
	}

	pub fn is_reduction_head(&self) -> bool {
		self.reduction_head
	}

	pub fn is_captured(&self) -> bool {
		!self.capture.is_empty()
	}

	pub fn is_fork(&self) -> bool {
		// TODO - should check for duplicated parents
		self.parents.len() != 1 && !self.is_input()
	}

	/// `fragment_head` is a node whose result we may have to store into a tensor.
	//	#[allow(clippy::nonminimal_bool)]
	#[rustfmt::skip]
	pub fn is_fragment_head(&self) -> bool {
		self.is_reduction_head()
		|| self.is_fork()
	}
}

#[derive(Clone)]
pub struct MergeGroup {
	pub tensors: Vec<Rc<ExprTensorRef>>,
}

pub struct MergeGroupBuilder {
	tensor_ref_map: std::collections::HashMap<*const ExprTensorRef, usize>, // *ExprTensorRef -> Index
	tensor_ref_vec: Vec<Rc<ExprTensorRef>>, // Index -> Rc<ExprTensorRef>
	union_find: UnionFind,
}

impl MergeGroupBuilder {
	pub fn new() -> Self {
		Self {
			tensor_ref_map: std::collections::HashMap::new(),
			tensor_ref_vec: Vec::new(),
			union_find: UnionFind::new(0),
		}
	}

	pub fn add(&mut self, tensor_ref: &Rc<ExprTensorRef>) -> usize {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		match self.tensor_ref_map.entry(key) {
			std::collections::hash_map::Entry::Vacant(entry) => {
				let index = self.tensor_ref_vec.len();
				entry.insert(index);
				self.tensor_ref_vec.push(tensor_ref.clone());
				self.union_find.add();
				index
			},
			std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
		}
	}

	pub fn union(&mut self, index0: usize, index1: usize) -> usize {
		self.union_find.union(index0, index1)
	}

	pub fn build(self, compilation: &mut Compilation) {
		let (compact_ids, sets_cnt) = self.union_find.compact_ids();
		compilation.merge_groups = vec![MergeGroup { tensors: Vec::new() }; sets_cnt];
		for (i, tensor_ref) in self.tensor_ref_vec.into_iter().enumerate() {
			let group_id = compact_ids[i];
			compilation.merge_groups[group_id].tensors.push(tensor_ref);
		}
		for node in &mut compilation.nodes {
			if node.merge_group < compact_ids.len() {
				node.merge_group = compact_ids[node.merge_group];
			}
		}
	}
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIndex(usize);

impl NodeIndex {
	pub fn invalid() -> Self {
		NodeIndex(usize::MAX)
	}

	pub fn is_valid(&self) -> bool {
		(self.0 as isize) >= 0
	}
}

pub struct NodeVec {
	vec: Vec<Node>,
}

impl NodeVec {
	pub fn add_node(&mut self, node: Node) -> NodeIndex {
		let index = NodeIndex(self.vec.len());
		self.vec.push(node);
		index
	}

	pub fn get(&self, index: NodeIndex) -> &Node {
		&self.vec[index.0]
	}

	pub fn indexes(&self) -> impl Iterator<Item = NodeIndex> + use<> {
		let len = self.vec.len();
		(0..len).map(NodeIndex)
	}
}

impl std::ops::Index<NodeIndex> for NodeVec {
	type Output = Node;

	fn index(&self, index: NodeIndex) -> &Node {
		&self.vec[index.0]
	}
}

impl std::ops::IndexMut<NodeIndex> for NodeVec {
	fn index_mut(&mut self, index: NodeIndex) -> &mut Node {
		&mut self.vec[index.0]
	}
}

impl<'a> IntoIterator for &'a NodeVec {
	type Item = &'a Node;
	type IntoIter = std::slice::Iter<'a, Node>;
	fn into_iter(self) -> Self::IntoIter {
		self.vec.iter()
	}
}

impl<'a> IntoIterator for &'a mut NodeVec {
	type Item = &'a mut Node;
	type IntoIter = std::slice::IterMut<'a, Node>;
	fn into_iter(self) -> Self::IntoIter {
		self.vec.iter_mut()
	}
}

pub struct Compilation {
	nodes: NodeVec,
	roots: Vec<NodeIndex>,
	merge_groups: Vec<MergeGroup>,
	fragments: Vec<NodeIndex>,

	processed: std::collections::HashMap<*const Expr, NodeIndex>,
	tensor_inputs: std::collections::HashMap<*const ExprTensorRef, NodeIndex>,
	scalar_inputs: std::collections::HashMap<*const ExprScalarRef, NodeIndex>,
	merge_group_builder: Option<MergeGroupBuilder>,
}

impl Compilation {
	pub fn new_from_expr(expr: RcExpr) -> Compilation {
		let mut compilation = Self {
			nodes: NodeVec { vec: Vec::with_capacity(32) },
			roots: Vec::with_capacity(1),
			merge_groups: Vec::new(),
			fragments: Vec::new(),

			processed: std::collections::HashMap::new(),
			tensor_inputs: std::collections::HashMap::new(),
			scalar_inputs: std::collections::HashMap::new(),
			merge_group_builder: Some(MergeGroupBuilder::new()),
		};
		let root = compilation.__new_from_expr(expr.rc_expr);
		compilation.merge_group_builder.take().unwrap().build(&mut compilation);
		compilation.move_reduction_heads();
		compilation.roots.push(root);
		compilation.roots.sort();
		compilation.roots.dedup();
		compilation.roots.retain(|&r| compilation.nodes[r].parents.is_empty());
		compilation.mark_fragments();
		compilation
	}

	fn move_reduction_heads(&mut self) {
		for mut i in self.nodes.indexes() {
			if !self.nodes[i].is_reduction() {
				continue;
			}
			self.nodes[i].reduction_head = false;
			while let [parent] = &self.nodes[i].parents[..] // has exactly one parent
			&& let parent = *parent
				&& self.nodes[parent].out_shape.last() == Some(1)
				&& self.nodes[i].reduction_bitmap.is_equal(&self.nodes[parent].reduction_bitmap)
			{
				i = parent;
			}
			self.nodes[i].reduction_head = true;
		}
	}

	fn __mark_fragments(&mut self, mut current: NodeIndex, node: NodeIndex) {
		if self.nodes[node].is_input() {
			return;
		}
		if self.nodes[node].is_fragment_head() {
			if self.nodes[node].fragment.is_valid() {
				return;
			}
			self.fragments.push(node);
			current = node;
		} else {
			assert!(!self.nodes[node].fragment.is_valid());
		}
		self.nodes[node].fragment = current;
		for i in 0..self.nodes[node].children.len() {
			self.__mark_fragments(current, self.nodes[node].children[i]);
		}
	}

	fn mark_fragments(&mut self) {
		for i in 0..self.roots.len() {
			self.__mark_fragments(NodeIndex::invalid(), self.roots[i]);
		}
	}

	pub fn root_cnt(&self) -> usize {
		self.roots.len()
	}

	pub fn root(&self, i: usize) -> NodeIndex {
		self.roots[i]
	}

	fn add_child(&mut self, parent: NodeIndex, child: NodeIndex) {
		self.nodes[parent].children.push(child);
		self.nodes[child].parents.push(parent);
	}

	#[allow(clippy::too_many_lines)]
	#[allow(clippy::cast_possible_wrap)]
	#[allow(clippy::collapsible_else_if)]
	#[allow(clippy::manual_assert)]
	#[allow(clippy::panic)]
	pub fn __new_from_expr(&mut self, expr: Rc<Expr>) -> NodeIndex {
		let expr_key = std::ptr::from_ref(expr.as_ref());
		if let Some(index) = self.processed.get(&expr_key) {
			return *index;
		}

		match expr.as_ref() {
			Expr::Capture(capture) => {
				let child = self.__new_from_expr(capture.expr.clone());
				let capture_shape_constraint = capture.tensor_ref.shape_constraint();
				self.nodes[child].out_shape =
					ShapeConstraint::union(&self.nodes[child].out_shape, &capture_shape_constraint);
				if self.nodes[child].is_input() {
					let id_expr = Rc::new(Expr::Unary(ExprUnary {
						kind: ExprUnaryKind::Identity,
						expr: self.nodes[child].expr.clone(),
					}));
					let id = self.nodes.add_node(Node {
						expr: id_expr,
						parents: Vec::new(),
						children: Vec::new(),
						capture: Vec::new(),
						merge_group: usize::MAX,
						out_shape: ShapeConstraint::new(),
						out_is_scalar: false,
						reduction_head: false,
						reduction_bitmap: ReductionBitmap::new(),
						fragment: NodeIndex::invalid(),
					});
					self.add_child(id, child);
					self.nodes[id].capture.push(capture.tensor_ref.clone());
					self.roots.push(id);
				} else {
					self.nodes[child].capture.push(capture.tensor_ref.clone());
				}
				self.processed.insert(expr_key, child);
				return child;
			},
			Expr::First(first) => {
				let first_child = self.__new_from_expr(first.lhs.clone());
				let second_child = self.__new_from_expr(first.rhs.clone());
				self.processed.insert(expr_key, first_child);
				self.roots.push(second_child);
				return first_child;
			},
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					if let Some(index) =
						self.tensor_inputs.get(&std::ptr::from_ref(tensor_ref.as_ref()))
					{
						self.processed.insert(expr_key, *index);
						return *index;
					}
				},
				ExprInput::Scalar(scalar_ref) => {
					if let Some(index) =
						self.scalar_inputs.get(&std::ptr::from_ref(scalar_ref.as_ref()))
					{
						self.processed.insert(expr_key, *index);
						return *index;
					}
				},
			},
			_ => {},
		};

		let index = self.nodes.add_node(Node {
			expr,
			parents: Vec::new(),
			children: Vec::new(),
			capture: Vec::new(),
			merge_group: usize::MAX,
			out_shape: ShapeConstraint::new(),
			out_is_scalar: false,
			reduction_head: false,
			reduction_bitmap: ReductionBitmap::new(),
			fragment: NodeIndex::invalid(),
		});
		self.processed.insert(expr_key, index);
		let expr_ref = self.nodes[index].expr.as_ref();
		match expr_ref {
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					self.tensor_inputs.insert(std::ptr::from_ref(tensor_ref.as_ref()), index);
					let merge_group = self.merge_group_builder.as_mut().unwrap().add(tensor_ref);
					let shape_constraint = tensor_ref.shape_constraint();
					self.nodes[index].merge_group = merge_group;
					self.nodes[index].out_shape = shape_constraint;
				},
				ExprInput::Scalar(scalar_ref) => {
					self.scalar_inputs.insert(std::ptr::from_ref(scalar_ref.as_ref()), index);
					self.nodes[index].out_is_scalar = true;
				},
			},
			Expr::Capture(..) | Expr::First(..) => {
				unreachable!() // handled above
			},
			Expr::Cast(cast) => {
				let child = self.__new_from_expr(cast.expr.clone());
				self.nodes[index].out_is_scalar = self.nodes[child].out_is_scalar;
				self.nodes[index].out_shape = self.nodes[child].out_shape.clone();
				self.nodes[index].merge_group = self.nodes[child].merge_group;
				self.nodes[index].reduction_bitmap = self.nodes[child].reduction_bitmap.clone();
				self.add_child(index, child);
			},
			Expr::Unary(unary) => {
				let child = self.__new_from_expr(unary.expr.clone());
				self.nodes[index].out_is_scalar = self.nodes[child].out_is_scalar;
				self.nodes[index].out_shape = self.nodes[child].out_shape.clone();
				self.nodes[index].merge_group = self.nodes[child].merge_group;
				self.nodes[index].reduction_bitmap = self.nodes[child].reduction_bitmap.clone();
				self.add_child(index, child);
			},
			Expr::Binary(binary) => {
				let lhs = binary.lhs.clone();
				let rhs = binary.rhs.clone();
				let left_child = self.__new_from_expr(lhs);
				let right_child = self.__new_from_expr(rhs);
				self.add_child(index, left_child);
				self.add_child(index, right_child);
				self.nodes[index].reduction_bitmap = ReductionBitmap::union(
					&self.nodes[left_child].reduction_bitmap,
					&self.nodes[right_child].reduction_bitmap,
				);
				if self.nodes[left_child].out_is_scalar {
					self.nodes[index].out_is_scalar = self.nodes[right_child].out_is_scalar;
					if !self.nodes[index].out_is_scalar {
						self.nodes[index].out_is_scalar = false;
						self.nodes[index].out_shape = self.nodes[right_child].out_shape.clone();
						self.nodes[index].merge_group = self.nodes[right_child].merge_group;
					}
				} else {
					self.nodes[index].out_is_scalar = false;
					if self.nodes[right_child].out_is_scalar {
						self.nodes[index].out_shape = self.nodes[left_child].out_shape.clone();
						self.nodes[index].merge_group = self.nodes[left_child].merge_group;
					} else {
						// shape group
						self.nodes[index].merge_group =
							if (self.nodes[right_child].merge_group as isize) < 0 {
								self.nodes[left_child].merge_group
							} else if (self.nodes[left_child].merge_group as isize) < 0 {
								self.nodes[right_child].merge_group
							} else {
								self.merge_group_builder.as_mut().unwrap().union(
									self.nodes[left_child].merge_group,
									self.nodes[right_child].merge_group,
								)
							};
						// shape
						self.nodes[index].out_shape = ShapeConstraint::intersection(
							&self.nodes[left_child].out_shape,
							&self.nodes[right_child].out_shape,
						);
					}
				}
			},
			Expr::Reduction(reduction) => {
				let reduction_expr = reduction.expr.clone();
				self.nodes[index].reduction_head = true;
				let child = self.__new_from_expr(reduction_expr);
				self.add_child(index, child);
				self.nodes[index].reduction_bitmap =
					self.nodes[child].reduction_bitmap.clone_and_set(index.0);
				self.nodes[index].out_is_scalar = self.nodes[child].out_is_scalar;
				if !self.nodes[index].out_is_scalar {
					self.nodes[index].merge_group = usize::MAX;
					self.nodes[index]
						.out_shape
						.set_last(Some(DimConstraint { source: "reduction".to_string(), size: 1 }));
				}
			},
		}
		index
	}

	pub fn print_graphviz<'a, W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		writeln!(w, "\trankdir=BT;")?;
		for (i, node) in self.nodes.vec.iter().enumerate() {
			let extra_label = if node.is_tensor_input() {
				format!("<br/>group: {}", node.merge_group)
			} else {
				String::new()
			};
			writeln!(w, "\t\t{} [label=<{}{}>];", i, node.graphviz_label(), extra_label)?;
			if node.is_input() {
				writeln!(w, "\t{} [shape=box];", i)?;
				if node.is_scalar_input() {
					writeln!(w, "\t{} [style=filled, fillcolor=\"#ffffc0\"];", i)?;
				} else {
					writeln!(w, "\t{} [style=filled, fillcolor=\"#a0f0ff\"];", i)?;
				}
			}
			if node.is_fork() {
				writeln!(w, "\t{} [style=filled, fillcolor=\"#ffcccc\"];", i)?;
			} else if node.is_reduction_head() {
				writeln!(w, "\t{} [style=filled, fillcolor=\"#ccccff\"];", i)?;
			} else if node.is_reduction() {
				writeln!(w, "\t{} [style=filled, fillcolor=\"#f0f0ff\"];", i)?;
			} else if node.is_captured() {
				writeln!(w, "\t{} [style=filled, fillcolor=\"#ccffcc\"];", i)?;
			}
			if node.fragment.is_valid() {
				writeln!(w, "subgraph cluster_{} {{ {} }}", node.fragment.0, i)?;
			}
			for &child_index in &node.children {
				let label = if self.nodes[child_index].out_is_scalar {
					String::new()
				} else {
					self.nodes[child_index].out_shape.as_str()
				};
				let extra_style = if node.fragment.is_valid()
					&& self.nodes[child_index].fragment.is_valid()
					&& node.fragment != self.nodes[child_index].fragment
				{
					", color=red, style=bold"
				} else {
					""
				};
				writeln!(w, "\t{} -> {} [label=\"{}\"{}];", child_index.0, i, label, extra_style)?;
			}
			for cap in &node.capture {
				let cap_label = if let Some(name) = &cap.name {
					format!("<b>Capture</b><br/><font color='blue'><b>{}</b></font>", name)
				} else {
					format!(
						"<b>Capture</b><br/><font color='blue'><b>{:?}</b></font>",
						std::ptr::from_ref(cap.as_ref())
					)
				};
				writeln!(
					w,
					"\tcap_{} [label=<{}>, shape=box, style=filled, fillcolor=\"#cceecc\"];",
					std::ptr::from_ref(cap.as_ref()) as usize,
					cap_label
				)?;
				writeln!(w, "\t{} -> cap_{};", i, std::ptr::from_ref(cap.as_ref()) as usize,)?;
				if node.fragment.is_valid() {
					writeln!(
						w,
						"subgraph cluster_{} {{ cap_{} }}",
						node.fragment.0,
						std::ptr::from_ref(cap.as_ref()) as usize
					)?;
				}
			}
		}
		writeln!(w, "}}")?;
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct RcExpr {
	pub rc_expr: Rc<Expr>,
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
	pub shape_constraint: Vec<usize>,
	pub name: Option<Cow<'static, str>>,
}

pub struct ExprScalarRef {
	pub value: RefCell<Option<f64>>,
	pub name: Option<Cow<'static, str>>,
}

pub struct ExprCapture {
	pub expr: Rc<Expr>,
	pub tensor_ref: Rc<ExprTensorRef>,
}

pub struct ExprCast {
	pub expr: Rc<Expr>,
	pub dtype: DType,
}

pub struct ExprUnary {
	pub kind: ExprUnaryKind,
	pub expr: Rc<Expr>,
}

pub enum ExprUnaryKind {
	Neg,
	Exp,
	Ln,
	Abs,
	Sqrt,
	Recip,
	Identity,
}

pub struct ExprFirst {
	pub lhs: Rc<Expr>,
	pub rhs: Rc<Expr>,
}

pub struct ExprBinary {
	pub kind: ExprBinaryKind,
	pub lhs: Rc<Expr>,
	pub rhs: Rc<Expr>,
}

pub enum ExprBinaryKind {
	Add,
	Sub,
	Mul,
}

pub struct ExprReduction {
	pub kind: ExprReductionKind,
	pub expr: Rc<Expr>,
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
			shape_constraint: shape,
			name,
		})
	}

	pub fn shape_constraint(&self) -> ShapeConstraint {
		let source = if let Some(name) = &self.name { name.as_ref() } else { "unnamed tensor" };
		ShapeConstraint {
			constraint: self
				.shape_constraint
				.iter()
				.map(|&d| Some(DimConstraint { source: source.to_string(), size: d }))
				.collect(),
		}
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
			rc_expr: Rc::new(Expr::Cast(ExprCast { expr: self.rc_expr, dtype })),
		}
	}

	pub fn exp(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Exp,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn ln(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Ln,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn abs(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Abs,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn sqrt(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Sqrt,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn recip(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Recip,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn sum(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Reduction(ExprReduction {
				kind: ExprReductionKind::Sum,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn max(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Reduction(ExprReduction {
				kind: ExprReductionKind::Max,
				expr: self.rc_expr,
			})),
		}
	}

	pub fn first(first: RcExpr, second: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::First(ExprFirst { lhs: first.rc_expr, rhs: second.rc_expr })),
		}
	}

	pub fn capture(self, tensor_ref: Rc<ExprTensorRef>) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Capture(ExprCapture { expr: self.rc_expr, tensor_ref })),
		}
	}
}

impl std::ops::Add for RcExpr {
	type Output = RcExpr;

	fn add(self, rhs: RcExpr) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Binary(ExprBinary {
				kind: ExprBinaryKind::Add,
				lhs: self.rc_expr,
				rhs: rhs.rc_expr,
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
				lhs: self.rc_expr,
				rhs: rhs.rc_expr,
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
				lhs: self.rc_expr,
				rhs: rhs.rc_expr,
			})),
		}
	}
}

impl std::ops::Neg for RcExpr {
	type Output = RcExpr;

	fn neg(self) -> RcExpr {
		RcExpr {
			rc_expr: Rc::new(Expr::Unary(ExprUnary {
				kind: ExprUnaryKind::Neg,
				expr: self.rc_expr,
			})),
		}
	}
}

//--------------------------------------------------------------------------------------------------
