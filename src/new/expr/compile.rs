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
use std::rc::Rc;

use super::{
	Expr, ExprBinaryKind, ExprInput, ExprReductionKind, ExprScalarRef, ExprTensorRef, ExprUnary,
	ExprUnaryKind, RcExpr,
};
use crate::define_index_type;
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
	pub is_root: bool,
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

pub struct CompiledExpr {
	nodes: NodeVec,
	roots: Vec<NodeIndex>,
	fragments: FragmentVec,
	captures: HashSet<*const ExprTensorRef>,
	processed: HashMap<*const Expr, NodeIndex>,
	scalar_ref_map: HashMap<*const ExprScalarRef, Rc<ExprScalarRef>>,
	tensor_ref_map: HashMap<*const ExprTensorRef, TensorNodeIndex>,
	tensor_ref_vec: TensorNodeVec,
}

impl CompiledExpr {
	pub fn new(expr: RcExpr) -> Self {
		let mut comp = Self {
			nodes: NodeVec::with_capacity(32),
			roots: Vec::with_capacity(1),
			fragments: FragmentVec::new(),
			captures: HashSet::new(),
			processed: HashMap::new(),
			scalar_ref_map: HashMap::new(),
			tensor_ref_map: HashMap::new(),
			tensor_ref_vec: TensorNodeVec::with_capacity(4),
		};
		let root = comp.__new(expr.rc_expr);
		comp.roots.push(root);
		comp.roots.sort();
		comp.roots.dedup();
		comp.roots.retain(|&r| comp.nodes[r].parents.is_empty());
		comp.remove_dead_code();
		comp.move_reduction_heads();
		comp.mark_fragments();
		comp.add_temp_captures();
		comp
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
		let old_roots = std::mem::replace(&mut self.roots, Vec::new());
		for root in old_roots {
			self.__find_live_code(root, NodeIndex::new_invalid());
		}
	}

	fn __find_live_code(&mut self, node: NodeIndex, parent: NodeIndex) {
		if let Some(pos) = self.nodes[node].parents.iter().position(|&p| p == parent) {
			self.nodes[node].parents.swap_remove(pos);
		}
		if !self.nodes[node].parents.is_empty() {
			return;
		}
		if self.nodes[node].capture.is_empty() {
			for child in 0..self.nodes[node].children.len() {
				self.__find_live_code(self.nodes[node].children[child], node);
			}
		} else {
			self.roots.push(node);
		}
	}

	fn move_reduction_heads(&mut self) {
		for mut i in self.nodes.indexes() {
			if !self.nodes[i].is_reduction() {
				continue;
			}
			self.nodes[i].is_reduction_head = false;
			while let [one_parent] = &self.nodes[i].parents[..]
				&& let parent = *one_parent
				&& !self.nodes[i].is_reduction_input()
				&& self.nodes[parent].out_shape.last() == Some(1)
				&& self.nodes[i].reduction_bitmap.is_equal(&self.nodes[parent].reduction_bitmap)
			{
				i = parent;
			}
			self.nodes[i].is_reduction_head = true;
		}
	}

	fn __mark_fragments(&mut self, mut frag: FragmentIndex, node: NodeIndex) {
		let is_root = !frag.is_valid();
		if let Expr::Input(input) = self.nodes[node].expr.as_ref() {
			assert!(!is_root);
			match input {
				ExprInput::Scalar(scalar_ref) => {
					self.fragments[frag].add_scalar_input(scalar_ref);
				},
				ExprInput::Tensor(tensor_ref) => {
					self.fragments[frag].add_tensor_input(tensor_ref);
				},
			}
			return;
		}
		if self.nodes[node].is_fragment_head() {
			if self.nodes[node].fragment.is_valid() {
				if !is_root {
					let child_frag = self.nodes[node].fragment;
					self.fragments[child_frag].input_into.insert(frag);
				}
				return;
			}
			let child_frag = self.fragments.push(Fragment {
				is_root,
				head: node,
				kind: if self.nodes[node].is_reduction_head() {
					FragmentKind::Reduction
				} else {
					FragmentKind::ElementWise
				},
				reduction: NodeIndex::new_invalid(),
				input_into: if is_root { HashSet::new() } else { HashSet::from([frag]) },
				scalar_inputs_map: HashMap::new(),
				scalar_inputs_vec: Vec::new(),
				tensor_inputs_map: HashMap::new(),
				tensor_inputs_vec: Vec::new(),
				tensor_outputs_map: HashMap::new(),
				tensor_outputs_vec: Vec::new(),
			});
			frag = child_frag;
		} else {
			assert!(!self.nodes[node].fragment.is_valid());
		}
		for i in 0..self.nodes[node].capture.len() {
			self.fragments[frag].add_tensor_output(&self.nodes[node].capture[i]);
		}
		if self.nodes[node].is_reduction() {
			assert!(!self.fragments[frag].reduction.is_valid());
			self.fragments[frag].reduction = node;
		}
		self.nodes[node].fragment = frag;
		for i in 0..self.nodes[node].children.len() {
			self.__mark_fragments(frag, self.nodes[node].children[i]);
		}
	}

	fn mark_fragments(&mut self) {
		for i in 0..self.roots.len() {
			self.__mark_fragments(FragmentIndex::new_invalid(), self.roots[i]);
		}
	}

	fn add_temp_captures(&mut self) {
		let mut cnt = 0;
		for i in self.fragments.indexes() {
			let f = &self.fragments[i];
			if self.nodes[f.head].is_captured() {
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

			self.nodes[f.head].capture.push(tensor_ref);
		}
	}

	fn add_child(&mut self, parent: NodeIndex, child: NodeIndex) {
		self.nodes[parent].children.push(child);
		self.nodes[child].parents.push(parent);
	}

	#[allow(clippy::too_many_lines)]
	#[allow(clippy::collapsible_else_if)]
	#[allow(clippy::manual_assert)]
	#[allow(clippy::panic)]
	pub fn __new(&mut self, expr: Rc<Expr>) -> NodeIndex {
		let expr_key = std::ptr::from_ref(expr.as_ref());
		if let Some(index) = self.processed.get(&expr_key) {
			return *index;
		}

		match expr.as_ref() {
			Expr::Capture(capture) => {
				let child = self.__new(capture.expr.clone());
				let tensor_ref = capture.tensor_ref.clone();
				if !self.captures.insert(std::ptr::from_ref(tensor_ref.as_ref())) {
					panic!(
						"CompiledExpr::new(): Capturing multiple values into the same tensor '{}'.",
						tensor_ref.name.as_deref().unwrap_or("unnamed tensor")
					);
				}
				self.add_tensor_ref(tensor_ref.clone(), false, true);
				let child_shape_constraint = &self.nodes[child].out_shape;
				let capture_shape_constraint = tensor_ref.shape_constraint();
				self.nodes[child].out_shape =
					ShapeConstraint::capture(child_shape_constraint, &capture_shape_constraint);
				if self.nodes[child].is_input() {
					let id_expr = Rc::new(Expr::Unary(ExprUnary {
						kind: ExprUnaryKind::Identity,
						expr: self.nodes[child].expr.clone(),
					}));
					let id = self.nodes.push(Node {
						expr: id_expr,
						parents: Vec::new(),
						children: Vec::new(),
						capture: Vec::new(),

						out_is_scalar: false,
						out_shape: ShapeConstraint::new(),
						is_reduction_head: false,
						is_reduction_input: false,
						reduction_bitmap: ReductionBitmap::new(),
						fragment: FragmentIndex::new_invalid(),
					});
					self.add_child(id, child);
					self.nodes[id].capture.push(tensor_ref);
					self.roots.push(id);
				} else {
					self.nodes[child].capture.push(tensor_ref);
				}
				self.processed.insert(expr_key, child);
				return child;
			},
			Expr::First(first) => {
				let first_child = self.__new(first.lhs.clone());
				let second_child = self.__new(first.rhs.clone());
				self.processed.insert(expr_key, first_child);
				self.roots.push(second_child);
				return first_child;
			},
			_ => {},
		}

		let index = self.nodes.push(Node {
			expr,
			parents: Vec::new(),
			children: Vec::new(),
			capture: Vec::new(),

			out_is_scalar: false,
			out_shape: ShapeConstraint::new(),
			is_reduction_head: false,
			is_reduction_input: false,
			reduction_bitmap: ReductionBitmap::new(),
			fragment: FragmentIndex::new_invalid(),
		});
		self.processed.insert(expr_key, index);
		let expr_ref = self.nodes[index].expr.as_ref();
		match expr_ref {
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					let tensor_ref = tensor_ref.clone();
					let shape_constraint = tensor_ref.shape_constraint();
					self.add_tensor_ref(tensor_ref, true, false);
					self.nodes[index].out_shape = shape_constraint;
				},
				ExprInput::Scalar(scalar_ref) => {
					self.add_scalar_ref(scalar_ref.clone());
					self.nodes[index].out_is_scalar = true;
				},
			},
			Expr::Capture(..) | Expr::First(..) => {
				unreachable!() // handled above
			},
			Expr::Cast(cast) => {
				let child = self.__new(cast.expr.clone());
				self.nodes[index].out_is_scalar = self.nodes[child].out_is_scalar;
				self.nodes[index].out_shape = self.nodes[child].out_shape.clone();
				self.nodes[index].reduction_bitmap = self.nodes[child].reduction_bitmap.clone();
				self.add_child(index, child);
			},
			Expr::Unary(unary) => {
				let child = self.__new(unary.expr.clone());
				self.nodes[index].out_is_scalar = self.nodes[child].out_is_scalar;
				self.nodes[index].out_shape = self.nodes[child].out_shape.clone();
				self.nodes[index].reduction_bitmap = self.nodes[child].reduction_bitmap.clone();
				self.add_child(index, child);
			},
			Expr::Binary(binary) => {
				let lhs = binary.lhs.clone();
				let rhs = binary.rhs.clone();
				let left_child = self.__new(lhs);
				let right_child = self.__new(rhs);
				self.add_child(index, left_child);
				self.add_child(index, right_child);
				self.nodes[index].reduction_bitmap = ReductionBitmap::union(
					&self.nodes[left_child].reduction_bitmap,
					&self.nodes[right_child].reduction_bitmap,
				);
				let left_is_scalar = self.nodes[left_child].out_is_scalar;
				let right_is_scalar = self.nodes[right_child].out_is_scalar;
				if left_is_scalar {
					self.nodes[index].out_is_scalar = right_is_scalar;
					if !right_is_scalar {
						self.nodes[index].out_shape = self.nodes[right_child].out_shape.clone();
					}
				} else {
					self.nodes[index].out_is_scalar = false;
					if right_is_scalar {
						self.nodes[index].out_shape = self.nodes[left_child].out_shape.clone();
					} else {
						self.nodes[index].out_shape = ShapeConstraint::bin_op(
							&self.nodes[left_child].out_shape,
							&self.nodes[right_child].out_shape,
						);
					}
				}
			},
			Expr::Reduction(reduction) => {
				let reduction_expr = reduction.expr.clone();
				self.nodes[index].is_reduction_head = true;
				let child = self.__new(reduction_expr);
				self.nodes[child].is_reduction_input = true;
				self.add_child(index, child);
				self.nodes[index].reduction_bitmap =
					self.nodes[child].reduction_bitmap.clone_and_set(index.raw);
				self.nodes[index].out_is_scalar = self.nodes[child].out_is_scalar;
				if !self.nodes[index].out_is_scalar {
					self.nodes[index].out_shape = self.nodes[child].out_shape.clone();
					self.nodes[index]
						.out_shape
						.set_last(Some(DimConstraint { source: "reduction".to_string(), size: 1 }));
				}
			},
		}
		index
	}

	fn graphviz_tensor_id(&self, tensor_ref: &Rc<ExprTensorRef>) -> String {
		format!("ten_{}", std::ptr::from_ref(tensor_ref.as_ref()) as usize)
	}

	fn graphviz_scalar_id(&self, scalar_ref: &Rc<ExprScalarRef>) -> String {
		format!("sca_{}", std::ptr::from_ref(scalar_ref.as_ref()) as usize)
	}

	fn graphviz_node_id(&self, node_index: NodeIndex) -> String {
		match self.nodes[node_index].expr.as_ref() {
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
		for i in self.nodes.indexes() {
			let node = &self.nodes[i];
			let node_id = self.graphviz_node_id(i);
			let extra_label = "";
			if node.is_input() {
				continue;
			}
			writeln!(w, "\t\t{node_id} [label=<{}{extra_label}>];", node.graphviz_label(),)?;
			if node.is_reduction_head() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccccff\"];")?;
			} else if node.is_fork() {
				if node.parents.is_empty() && !self.roots.contains(&i) {
					// dead code
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
				let fragment_kind = match self.fragments[node.fragment].kind {
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
		&self.fragments
	}
}

//--------------------------------------------------------------------------------------------------
