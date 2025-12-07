//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#![allow(clippy::mutable_key_type)]
#![allow(clippy::panic)]
#![allow(clippy::unused_self)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::implicit_hasher)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::new_without_default)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::collapsible_else_if)]

use std::collections::{HashMap, HashSet, hash_map};
use std::hint::cold_path;
use std::rc::Rc;

use super::{
	Expr, ExprBinaryKind, ExprInput, ExprReductionKind, ExprScalarRef, ExprTensorRef, ExprUnary,
	ExprUnaryKind, RcExpr,
};
use crate::define_index_type;
use crate::new::expr::{ExprCapture, ExprCast};
use crate::tensor::HasDType;
use crate::util::index_vec::IndexVec;

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct ReductionBitmap {
	bitmap: Vec<u64>,
}

impl ReductionBitmap {
	pub fn new() -> Self {
		Self { bitmap: Vec::new() }
	}

	pub fn union<'a>(mut a: &'a Self, mut b: &'a Self) -> Self {
		if a.bitmap.len() > b.bitmap.len() {
			std::mem::swap(&mut a, &mut b);
		}
		let mut result = Self {
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

	pub fn clone_and_set(&self, index: usize) -> Self {
		let word_index = index / 64;
		let bit_index = index % 64;
		let min_len = word_index + 1;
		let len = self.bitmap.len().max(min_len);
		let mut result = Self { bitmap: Vec::with_capacity(len) };
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

	pub fn is_equal(&self, other: &Self) -> bool {
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

	// Calculates shape constraint for element-wise binary operation.
	pub fn bin_op(a: &Self, b: &Self) -> Self {
		let mut result = Self::new();
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
						} else if d1.size == 1 {
							Some(d2.clone())
						} else if d2.size == 1 {
							Some(d1.clone())
						} else {
							panic!("ShapeConstraint::bin_op(): conflicting dimension constraints. Dimension -{}. Sizes: {} vs {}. The value {} comes from '{}'; the value {} comes from '{}'.", i, d1.size, d2.size, d1.size, d1.source, d2.size, d2.source);
						}
					} else {
						if d1.size != 1 {
							Some(d1.clone())
						} else {
							None
						}
					}
				} else {
					if let Some(d2) = d2 && d2.size != 1 {
						Some(d2.clone())
					} else {
						None
					}
				},
			);
		}
		result
	}

	// As opposed to `bin_op()`, there is no broadcasting, so both shapes must match.
	pub fn capture(a: &Self, b: &Self) -> Self {
		let mut result = Self::new();
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

#[allow(clippy::struct_excessive_bools)]
pub struct Node {
	pub expr: Rc<Expr>,
	pub parents: Vec<NodeIndex>,
	pub children: Vec<NodeIndex>,
	pub capture: Vec<Rc<ExprTensorRef>>, // TODO: `Vec<TensorNodeIndex>`?

	pub out_is_scalar: bool,
	pub out_shape: ShapeConstraint,
	pub is_reduction_head: bool,
	pub is_reduction_input: bool,
	pub reduction_bitmap: ReductionBitmap,
	pub fragment: FragmentIndex,
	pub is_dead: bool,
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
		self.is_reduction_head
	}

	pub fn is_reduction_input(&self) -> bool {
		self.is_reduction_input
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
		|| self.is_reduction_input()
		|| self.is_fork()
	}
}

define_index_type!(NodeIndex);
type NodeVec = IndexVec<NodeIndex, Node>;

pub enum FragmentKind {
	ElementWise,
	Reduction,
}

pub struct Fragment {
	pub head: NodeIndex,
	pub kind: FragmentKind,
	pub reduction: NodeIndex,
	pub input_into: HashSet<FragmentIndex>,
	pub scalar_inputs_map: HashMap<*const ExprScalarRef, usize>,
	pub scalar_inputs_vec: Vec<Rc<ExprScalarRef>>,
	pub tensor_inputs_map: HashMap<*const ExprTensorRef, usize>,
	pub tensor_inputs_vec: Vec<Rc<ExprTensorRef>>,
	pub tensor_outputs_map: HashMap<*const ExprTensorRef, usize>,
	pub tensor_outputs_vec: Vec<Rc<ExprTensorRef>>,
}

impl Fragment {
	pub fn add_scalar_input(&mut self, scalar_ref: &Rc<ExprScalarRef>) {
		let key = std::ptr::from_ref(scalar_ref.as_ref());
		if let hash_map::Entry::Vacant(entry) = self.scalar_inputs_map.entry(key) {
			let index = self.scalar_inputs_vec.len();
			self.scalar_inputs_vec.push(scalar_ref.clone());
			entry.insert(index);
		}
	}

	pub fn add_tensor_input(&mut self, tensor_ref: &Rc<ExprTensorRef>) {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		if let hash_map::Entry::Vacant(entry) = self.tensor_inputs_map.entry(key) {
			let index = self.tensor_inputs_vec.len();
			self.tensor_inputs_vec.push(tensor_ref.clone());
			entry.insert(index);
		}
	}

	pub fn add_tensor_output(&mut self, tensor_ref: &Rc<ExprTensorRef>) {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		if let hash_map::Entry::Vacant(entry) = self.tensor_outputs_map.entry(key) {
			let index = self.tensor_outputs_vec.len();
			self.tensor_outputs_vec.push(tensor_ref.clone());
			entry.insert(index);
		}
	}
}

define_index_type!(FragmentIndex);
type FragmentVec = IndexVec<FragmentIndex, Fragment>;

pub struct TensorNode {
	pub tensor_ref: Rc<ExprTensorRef>,
	pub is_input: bool,
	pub is_output: bool,
}

define_index_type!(TensorNodeIndex);
type TensorNodeVec = IndexVec<TensorNodeIndex, TensorNode>;

pub struct Compilation {
	nodes_postorder: NodeVec,
	frag_preorder: FragmentVec,
	captures: HashSet<*const ExprTensorRef>,
	scalar_ref_map: HashMap<*const ExprScalarRef, Rc<ExprScalarRef>>,
	tensor_ref_map: HashMap<*const ExprTensorRef, TensorNodeIndex>,
	tensor_ref_vec: TensorNodeVec,
}

pub struct CompiledExpr {
	nodes_postorder: NodeVec,
	frag_preorder: FragmentVec,
}

impl CompiledExpr {
	pub fn new(comp: Compilation) -> Self {
		Self {
			nodes_postorder: comp.nodes_postorder,
			frag_preorder: comp.frag_preorder,
		}
	}

	pub fn nodes(&self) -> &NodeVec {
		&self.nodes_postorder
	}

	pub fn fragments(&self) -> &FragmentVec {
		&self.frag_preorder
	}

	pub fn fragments_postorder(&self) -> impl DoubleEndedIterator<Item = FragmentIndex> {
		self.frag_preorder.indexes().rev()
	}
}

impl Compilation {
	pub fn new(expr: RcExpr) -> Self {
		let mut comp = Compilation {
			nodes_postorder: NodeVec::with_capacity(32),
			frag_preorder: FragmentVec::new(),
			captures: HashSet::new(),
			scalar_ref_map: HashMap::new(),
			tensor_ref_map: HashMap::new(),
			tensor_ref_vec: TensorNodeVec::with_capacity(4),
		};
		let _root = comp.load_expr(expr.rc_expr, &mut HashMap::new());
		comp.remove_dead_code();
		comp.find_reduction_heads();
		comp.mark_fragments();
		comp.add_temp_captures();
		comp
	}

	#[allow(clippy::too_many_lines)]
	#[allow(clippy::collapsible_else_if)]
	#[allow(clippy::manual_assert)]
	#[allow(clippy::panic)]
	// TODO - refactor to make non recursive
	fn load_expr(
		&mut self,
		mut expr: Rc<Expr>,
		visited: &mut HashMap<*const Expr, NodeIndex>,
	) -> NodeIndex {
		let expr_key = std::ptr::from_ref(expr.as_ref());
		if let Some(index) = visited.get(&expr_key) {
			return *index;
		}

		let out_is_scalar: bool;
		let out_shape: ShapeConstraint;
		let reduction_bitmap: ReductionBitmap;
		let children: Vec<NodeIndex>;
		let mut capture: Vec<Rc<ExprTensorRef>> = Vec::new();
		match expr.as_ref() {
			Expr::Capture(ExprCapture { expr: x, tensor_ref }) => {
				let child = self.load_expr(x.clone(), visited);
				let tensor_ref = tensor_ref.clone();
				if !self.captures.insert(std::ptr::from_ref(tensor_ref.as_ref())) {
					panic!(
						"CompiledExpr::new(): Capturing multiple values into the same tensor '{}'.",
						tensor_ref.name.as_deref().unwrap_or("unnamed tensor")
					);
				}
				self.add_tensor_ref(tensor_ref.clone(), false, true);
				let child_shape_constraint = &self.nodes_postorder[child].out_shape;
				let capture_shape_constraint = tensor_ref.shape_constraint();
				self.nodes_postorder[child].out_shape =
					ShapeConstraint::capture(child_shape_constraint, &capture_shape_constraint);
				if !self.nodes_postorder[child].is_input() {
					self.nodes_postorder[child].capture.push(tensor_ref);
					visited.insert(expr_key, child);
					return child;
				}
				// Insert `Identity` node to perform the capture.
				// This node is also a root.
				expr = Rc::new(Expr::Unary(ExprUnary {
					kind: ExprUnaryKind::Identity,
					expr: self.nodes_postorder[child].expr.clone(),
				}));
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				out_shape = self.nodes_postorder[child].out_shape.clone();
				reduction_bitmap = ReductionBitmap::new();
				children = vec![child];
				capture.push(tensor_ref);
			},
			Expr::First(first) => {
				let first_child = self.load_expr(first.lhs.clone(), visited);
				let _second_child = self.load_expr(first.rhs.clone(), visited);
				visited.insert(expr_key, first_child);
				return first_child;
			},
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					out_is_scalar = false;
					out_shape = tensor_ref.shape_constraint();
					reduction_bitmap = ReductionBitmap::new();
					children = Vec::new();
					self.add_tensor_ref(tensor_ref.clone(), true, false);
				},
				ExprInput::Scalar(scalar_ref) => {
					out_is_scalar = true;
					out_shape = ShapeConstraint::new();
					reduction_bitmap = ReductionBitmap::new();
					children = Vec::new();
					self.add_scalar_ref(scalar_ref.clone());
				},
			},
			Expr::Cast(ExprCast { expr, .. }) | Expr::Unary(ExprUnary { expr, .. }) => {
				let child = self.load_expr(expr.clone(), visited);
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				out_shape = self.nodes_postorder[child].out_shape.clone();
				reduction_bitmap = self.nodes_postorder[child].reduction_bitmap.clone();
				children = vec![child];
			},
			Expr::Binary(binary) => {
				let lhs = binary.lhs.clone();
				let rhs = binary.rhs.clone();
				let left_child = self.load_expr(lhs, visited);
				let right_child = self.load_expr(rhs, visited);
				reduction_bitmap = ReductionBitmap::union(
					&self.nodes_postorder[left_child].reduction_bitmap,
					&self.nodes_postorder[right_child].reduction_bitmap,
				);
				let left_is_scalar = self.nodes_postorder[left_child].out_is_scalar;
				let right_is_scalar = self.nodes_postorder[right_child].out_is_scalar;
				if left_is_scalar {
					out_is_scalar = right_is_scalar;
					if right_is_scalar {
						out_shape = ShapeConstraint::new();
					} else {
						out_shape = self.nodes_postorder[right_child].out_shape.clone();
					}
				} else {
					out_is_scalar = false;
					if right_is_scalar {
						out_shape = self.nodes_postorder[left_child].out_shape.clone();
					} else {
						out_shape = ShapeConstraint::bin_op(
							&self.nodes_postorder[left_child].out_shape,
							&self.nodes_postorder[right_child].out_shape,
						);
					}
				}
				children = vec![left_child, right_child];
			},
			Expr::Reduction(reduction) => {
				let reduction_expr = reduction.expr.clone();
				let child = self.load_expr(reduction_expr, visited);
				self.nodes_postorder[child].is_reduction_input = true;
				reduction_bitmap = self.nodes_postorder[child]
					.reduction_bitmap
					.clone_and_set(self.nodes_postorder.next_index().raw);
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				if out_is_scalar {
					out_shape = ShapeConstraint::new();
				} else {
					let mut shape = self.nodes_postorder[child].out_shape.clone();
					shape
						.set_last(Some(DimConstraint { source: "reduction".to_string(), size: 1 }));
					out_shape = shape;
				}
				children = vec![child];
			},
		}

		let next_index = self.nodes_postorder.next_index();
		for &child in &children {
			self.nodes_postorder[child].parents.push(next_index);
		}

		let index = self.nodes_postorder.push(Node {
			expr,
			parents: Vec::new(),
			children,
			capture,

			out_is_scalar,
			out_shape,
			is_reduction_head: false,
			is_reduction_input: false,
			reduction_bitmap,
			fragment: FragmentIndex::new_invalid(),
			is_dead: false,
		});
		debug_assert!(index == next_index);
		visited.insert(expr_key, index);
		index
	}

	fn add_scalar_ref(&mut self, scalar_ref: Rc<ExprScalarRef>) {
		let key = std::ptr::from_ref(scalar_ref.as_ref());
		if let hash_map::Entry::Vacant(entry) = self.scalar_ref_map.entry(key) {
			entry.insert(scalar_ref);
		}
	}

	fn add_tensor_ref(
		&mut self,
		tensor_ref: Rc<ExprTensorRef>,
		is_input: bool,
		is_output: bool,
	) -> TensorNodeIndex {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		match self.tensor_ref_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index =
					self.tensor_ref_vec.push(TensorNode { tensor_ref, is_input, is_output });
				entry.insert(index);
				index
			},
			hash_map::Entry::Occupied(entry) => {
				let index = *entry.get();
				if is_input {
					self.tensor_ref_vec[index].is_input = true;
				}
				if is_output {
					assert!(!self.tensor_ref_vec[index].is_output);
					self.tensor_ref_vec[index].is_output = true;
				}
				index
			},
		}
	}

	#[allow(clippy::mem_replace_with_default)]
	fn remove_dead_code(&mut self) {
		for i in self.nodes_postorder.indexes().rev() {
			let mut parents = std::mem::replace(&mut self.nodes_postorder[i].parents, Vec::new());
			parents.retain(|&p| !self.nodes_postorder[p].is_dead);
			if parents.is_empty() {
				if self.nodes_postorder[i].capture.is_empty() {
					self.nodes_postorder[i].is_dead = true;
				}
			} else {
				self.nodes_postorder[i].parents = parents;
			}
		}
	}

	fn find_reduction_heads(&mut self) {
		'nodes: for parent in self.nodes_postorder.indexes() {
			if self.nodes_postorder[parent].is_reduction() {
				self.nodes_postorder[parent].is_reduction_head = true;
			} else {
				if self.nodes_postorder[parent].out_shape.last() != Some(1) {
					continue 'nodes;
				}

				// Check if we have exactly one child that is a reduction head
				let mut child = NodeIndex::new_invalid();
				for c in 0..self.nodes_postorder[parent].children.len() {
					let child_index = self.nodes_postorder[parent].children[c];
					if self.nodes_postorder[child_index].is_reduction_head {
						if child.is_valid() {
							continue 'nodes;
						}
						child = child_index;
					}
				}
				if !child.is_valid() {
					continue 'nodes;
				}

				if self.nodes_postorder[child].parents == [parent]
					&& self.nodes_postorder[child]
						.reduction_bitmap
						.is_equal(&self.nodes_postorder[parent].reduction_bitmap)
				{
					self.nodes_postorder[child].is_reduction_head = false;
					self.nodes_postorder[parent].is_reduction_head = true;
				}
			}
		}
	}

	fn mark_fragments(&mut self) {
		for i in self.nodes_postorder.indexes().rev() {
			if self.nodes_postorder[i].is_dead {
				cold_path();
				continue;
			}
			if let Expr::Input(input) = self.nodes_postorder[i].expr.as_ref() {
				match input {
					ExprInput::Scalar(scalar_ref) => {
						for &parent in &self.nodes_postorder[i].parents {
							let parent_frag = self.nodes_postorder[parent].fragment;
							self.frag_preorder[parent_frag].add_scalar_input(scalar_ref);
						}
					},
					ExprInput::Tensor(tensor_ref) => {
						for &parent in &self.nodes_postorder[i].parents {
							let parent_frag = self.nodes_postorder[parent].fragment;
							self.frag_preorder[parent_frag].add_tensor_input(tensor_ref);
						}
					},
				}
				continue;
			}
			let frag = if self.nodes_postorder[i].is_fragment_head() {
				let input_into = self.nodes_postorder[i]
					.parents
					.iter()
					.map(|&p| self.nodes_postorder[p].fragment)
					.collect();
				let kind = if self.nodes_postorder[i].is_reduction_head() {
					FragmentKind::Reduction
				} else {
					FragmentKind::ElementWise
				};
				self.frag_preorder.push(Fragment {
					head: i,
					kind,
					reduction: NodeIndex::new_invalid(),
					input_into,
					scalar_inputs_map: HashMap::new(),
					scalar_inputs_vec: Vec::new(),
					tensor_inputs_map: HashMap::new(),
					tensor_inputs_vec: Vec::new(),
					tensor_outputs_map: HashMap::new(),
					tensor_outputs_vec: Vec::new(),
				})
			} else {
				debug_assert!(self.nodes_postorder[i].parents.len() == 1);
				let parent = self.nodes_postorder[i].parents[0];
				debug_assert!(self.nodes_postorder[parent].fragment.is_valid());
				self.nodes_postorder[parent].fragment
			};
			self.nodes_postorder[i].fragment = frag;
			for cap in 0..self.nodes_postorder[i].capture.len() {
				self.frag_preorder[frag].add_tensor_output(&self.nodes_postorder[i].capture[cap]);
			}
			if self.nodes_postorder[i].is_reduction() {
				assert!(!self.frag_preorder[frag].reduction.is_valid());
				self.frag_preorder[frag].reduction = i;
			}
		}
	}

	fn add_temp_captures(&mut self) {
		let mut cnt = 0;
		for i in self.frag_preorder.indexes() {
			let f = &self.frag_preorder[i];
			if self.nodes_postorder[f.head].is_captured() {
				continue;
			}
			let tensor_ref = Rc::new(ExprTensorRef {
				tensor: std::cell::RefCell::new(None),
				dtype: f32::dtype,
				shape_constraint: Vec::new(),
				name: Some(format!("__tmp__[{cnt}]").into()),
			});
			cnt += 1;

			let key = std::ptr::from_ref(tensor_ref.as_ref());
			let index = self.tensor_ref_vec.push(TensorNode {
				tensor_ref: tensor_ref.clone(),
				is_input: false,
				is_output: true,
			});
			self.tensor_ref_map.insert(key, index);

			self.nodes_postorder[f.head].capture.push(tensor_ref);
		}
	}

	fn graphviz_tensor_id(&self, tensor_ref: &Rc<ExprTensorRef>) -> String {
		format!("ten_{}", std::ptr::from_ref(tensor_ref.as_ref()) as usize)
	}

	fn graphviz_scalar_id(&self, scalar_ref: &Rc<ExprScalarRef>) -> String {
		format!("sca_{}", std::ptr::from_ref(scalar_ref.as_ref()) as usize)
	}

	fn graphviz_node_id(&self, node_index: NodeIndex) -> String {
		match self.nodes_postorder[node_index].expr.as_ref() {
			Expr::Input(ExprInput::Tensor(tensor_ref)) => self.graphviz_tensor_id(tensor_ref),
			Expr::Input(ExprInput::Scalar(scalar_ref)) => self.graphviz_scalar_id(scalar_ref),
			_ => {
				format!("expr_{}", node_index.raw)
			},
		}
	}

	pub fn print_graphviz<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		//writeln!(w, "\tgraph [splines=line];")?;
		writeln!(w, "\trankdir=BT;")?;
		for i in self.nodes_postorder.indexes() {
			let node = &self.nodes_postorder[i];
			let node_id = self.graphviz_node_id(i);
			let extra_label = "";
			if node.is_input() {
				continue;
			}
			writeln!(w, "\t\t{node_id} [label=<{}{extra_label}>];", node.graphviz_label(),)?;
			if node.is_reduction_head() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccccff\"];")?;
			} else if node.is_fork() {
				if node.is_dead {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#cccccc\"];")?;
				} else {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffcccc\"];")?;
				}
			} else if node.is_reduction() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#f0f0ff\"];")?;
			} else if node.is_captured() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccffcc\"];")?;
			}
			if node.fragment.is_valid() {
				let fragment_kind = match self.frag_preorder[node.fragment].kind {
					FragmentKind::ElementWise => "Element-wise",
					FragmentKind::Reduction => "Reduction",
				};
				writeln!(
					w,
					"subgraph cluster_{} {{ label=\"{fragment_kind}\" labelloc=\"b\" labeljust=\"l\" {node_id} }}",
					node.fragment.raw
				)?;
			}
			for &child_index in &node.children {
				let child_id = self.graphviz_node_id(child_index);
				let label = if self.nodes_postorder[child_index].out_is_scalar {
					String::new()
				} else {
					self.nodes_postorder[child_index].out_shape.as_str()
				};
				let extra_style = if node.fragment.is_valid()
					&& self.nodes_postorder[child_index].fragment.is_valid()
					&& node.fragment != self.nodes_postorder[child_index].fragment
				{
					", color=red, style=bold"
				} else {
					""
				};
				writeln!(w, "\t{child_id} -> {node_id} [label=\"{}\"{}];", label, extra_style)?;
			}
			for cap in &node.capture {
				let label =
					if node.out_is_scalar { String::new() } else { node.out_shape.as_str() };
				let cap_id = self.graphviz_tensor_id(cap);
				let key = std::ptr::from_ref(cap.as_ref());
				let index = self.tensor_ref_map[&key];
				if self.tensor_ref_vec[index].is_input {
					writeln!(w, "\t{node_id} -> {cap_id} [label=\"{label}\", constraint=false];")?;
				} else {
					writeln!(w, "\t{node_id} -> {cap_id} [label=\"{label}\"];")?;
				}
			}
		}
		for tensor_ref in &self.tensor_ref_vec {
			let id = self.graphviz_tensor_id(&tensor_ref.tensor_ref);
			let tensor_name = if let Some(name) = &tensor_ref.tensor_ref.name {
				name.as_ref().to_string()
			} else {
				format!("{:?}", std::ptr::from_ref(tensor_ref.tensor_ref.as_ref()))
			};
			let color = if tensor_name.starts_with("__tmp__[") { "#00eeee" } else { "#cceecc" };
			writeln!(
				w,
				"\t{id} [label=<<b>Tensor</b><br/><font color='blue'><b>{tensor_name}</b></font>>, shape=box, style=filled, fillcolor=\"{color}\"];",
			)?;
		}
		for scalar_ref in self.scalar_ref_map.values() {
			let id = self.graphviz_scalar_id(scalar_ref);
			let label = if let Some(name) = &scalar_ref.name {
				name.to_string()
			} else {
				format!("{:?}", std::ptr::from_ref(scalar_ref.as_ref()))
			};
			writeln!(
				w,
				"\t\t{id} [label=<<b>Scalar</b><br/><font color='blue'><b>{label}</b></font>>, shape=box, style=filled, fillcolor=\"#ffffc0\"];"
			)?;
		}
		writeln!(w, "}}")?;
		Ok(())
	}

	pub fn fragments(&self) -> &FragmentVec {
		&self.frag_preorder
	}
}

//--------------------------------------------------------------------------------------------------
