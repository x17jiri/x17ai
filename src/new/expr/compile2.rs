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

use thin_vec::ThinVec;

use super::{
	Expr, ExprBinaryKind, ExprInput, ExprReductionKind, ExprScalarRef, ExprTensorRef, ExprUnary,
	ExprUnaryKind, RcExpr,
};
use crate::define_index_type;
use crate::new::expr::{ExprCapture, ExprCast};
use crate::tensor::error::ShapeMismatchError;
use crate::tensor::{HasDType, TensorOpError};
use crate::util::index_vec::IndexVec;
use crate::util::union_find::UnionFind;

//--------------------------------------------------------------------------------------------------

#[allow(clippy::struct_excessive_bools)]
pub struct Node {
	pub expr: Rc<Expr>,
	pub shape: ThinVec<usize>,
	pub parents: ThinVec<NodeIndex>,
	pub children: [NodeIndex; 2],
	pub capture: Vec<TensorRefIndex>,

	pub out_is_scalar: bool,
	pub is_dead: bool,
	pub is_reduction_head: bool,
	pub input_index_raw: usize, // Either ScalarRefIndex or TensorRefIndex, depending on out_is_scalar
}

impl Node {
	pub fn is_input(&self) -> bool {
		self.is_nullary()
	}

	pub fn is_scalar_input(&self) -> bool {
		self.is_input() && self.out_is_scalar
	}

	pub fn is_tensor_input(&self) -> bool {
		self.is_input() && !self.out_is_scalar
	}

	pub fn is_reduction(&self) -> bool {
		matches!(self.expr.as_ref(), Expr::Reduction(_))
	}

	pub fn is_captured(&self) -> bool {
		!self.capture.is_empty()
	}

	pub fn is_nullary(&self) -> bool {
		(self.children[0].raw as isize) < 0
	}

	pub fn is_unary(&self) -> bool {
		(self.children[0].raw | !self.children[1].raw) as isize >= 0
	}

	pub fn is_binary(&self) -> bool {
		(self.children[1].raw as isize) >= 0
	}

	pub fn is_root(&self) -> bool {
		self.parents.is_empty()
	}

	pub fn is_fork(&self) -> bool {
		self.parents.len() != 1 && !self.is_input()
	}
}

define_index_type!(NodeIndex);
type NodeVec = IndexVec<NodeIndex, Node>;

pub struct TensorRef {
	pub tensor_ref: Rc<ExprTensorRef>,
	pub is_input: bool,
	pub is_output: bool,
}

define_index_type!(TensorRefIndex);
type TensorRefVec = IndexVec<TensorRefIndex, TensorRef>;

type TensorShapeVec = IndexVec<TensorRefIndex, ThinVec<usize>>;

define_index_type!(ScalarRefIndex);
type ScalarRefVec = IndexVec<ScalarRefIndex, Rc<ExprScalarRef>>;

pub struct PreCompilation {
	nodes_postorder: NodeVec,
	tensor_shapes: TensorShapeVec,
	scalar_ref_map: HashMap<*const ExprScalarRef, ScalarRefIndex>,
	scalar_ref_vec: ScalarRefVec,
	tensor_ref_map: HashMap<*const ExprTensorRef, TensorRefIndex>,
	tensor_ref_vec: TensorRefVec,
}

impl PreCompilation {
	pub fn new(expr: RcExpr) -> Self {
		let mut comp = PreCompilation {
			nodes_postorder: NodeVec::with_capacity(32),
			tensor_shapes: TensorShapeVec::with_capacity(32),
			scalar_ref_map: HashMap::new(),
			scalar_ref_vec: ScalarRefVec::with_capacity(4),
			tensor_ref_map: HashMap::new(),
			tensor_ref_vec: TensorRefVec::with_capacity(4),
		};
		let _root = comp.load_expr(expr.rc_expr, &mut HashMap::new(), &mut HashSet::new());
		comp.remove_dead_code();
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
		captures: &mut HashSet<*const ExprTensorRef>,
	) -> NodeIndex {
		let expr_key = std::ptr::from_ref(expr.as_ref());
		if let Some(index) = visited.get(&expr_key) {
			return *index;
		}

		let mut input_index_raw: usize = usize::MAX;
		let out_is_scalar: bool;
		let children: [NodeIndex; 2];
		let mut capture: Vec<TensorRefIndex> = Vec::new();
		match expr.as_ref() {
			Expr::Capture(ExprCapture { expr: x, tensor_ref }) => {
				let child = self.load_expr(x.clone(), visited, captures);
				let tensor_ref = tensor_ref.clone();
				if !captures.insert(std::ptr::from_ref(tensor_ref.as_ref())) {
					panic!(
						"CompiledExpr::new(): Capturing multiple values into the same tensor '{}'.",
						tensor_ref.name.as_deref().unwrap_or("unnamed tensor")
					);
				}
				let capture_shape_constraint = tensor_ref.shape_constraint();
				let tensor_ref_index = self.add_tensor_ref(tensor_ref, false, true);
				if !self.nodes_postorder[child].is_input() {
					self.nodes_postorder[child].capture.push(tensor_ref_index);
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
				children = [child, NodeIndex::new_invalid()];
				capture.push(tensor_ref_index);
			},
			Expr::First(first) => {
				let first_child = self.load_expr(first.lhs.clone(), visited, captures);
				let _second_child = self.load_expr(first.rhs.clone(), visited, captures);
				visited.insert(expr_key, first_child);
				return first_child;
			},
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					out_is_scalar = false;
					children = [NodeIndex::new_invalid(), NodeIndex::new_invalid()];
					input_index_raw = self.add_tensor_ref(tensor_ref.clone(), true, false).raw;
				},
				ExprInput::Scalar(scalar_ref) => {
					out_is_scalar = true;
					children = [NodeIndex::new_invalid(), NodeIndex::new_invalid()];
					input_index_raw = self.add_scalar_ref(scalar_ref.clone()).raw;
				},
			},
			Expr::Cast(ExprCast { expr, .. }) | Expr::Unary(ExprUnary { expr, .. }) => {
				let child = self.load_expr(expr.clone(), visited, captures);
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				children = [child, NodeIndex::new_invalid()];
			},
			Expr::Binary(binary) => {
				let lhs = binary.lhs.clone();
				let rhs = binary.rhs.clone();
				let left_child = self.load_expr(lhs, visited, captures);
				let right_child = self.load_expr(rhs, visited, captures);
				let left_is_scalar = self.nodes_postorder[left_child].out_is_scalar;
				let right_is_scalar = self.nodes_postorder[right_child].out_is_scalar;
				out_is_scalar = left_is_scalar && right_is_scalar;
				children = [left_child, right_child];
			},
			Expr::Reduction(reduction) => {
				let reduction_expr = reduction.expr.clone();
				let child = self.load_expr(reduction_expr, visited, captures);
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				children = [child, NodeIndex::new_invalid()];
			},
		}

		let next_index = self.nodes_postorder.next_index();
		for &child in &children {
			if !child.is_valid() {
				break;
			}
			self.nodes_postorder[child].parents.push(next_index);
		}

		let index = self.nodes_postorder.push(Node {
			expr,
			shape: ThinVec::new(),
			parents: ThinVec::new(),
			children,
			capture,
			out_is_scalar,
			is_dead: false,
			is_reduction_head: false,
			input_index_raw,
		});
		debug_assert!(index == next_index);
		visited.insert(expr_key, index);
		index
	}

	fn add_scalar_ref(&mut self, scalar_ref: Rc<ExprScalarRef>) -> ScalarRefIndex {
		let key = std::ptr::from_ref(scalar_ref.as_ref());
		match self.scalar_ref_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index = self.scalar_ref_vec.push(scalar_ref);
				entry.insert(index);
				index
			},
			hash_map::Entry::Occupied(entry) => {
				let index = *entry.get();
				index
			},
		}
	}

	fn add_tensor_ref(
		&mut self,
		tensor_ref: Rc<ExprTensorRef>,
		is_input: bool,
		is_output: bool,
	) -> TensorRefIndex {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		match self.tensor_ref_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index = self.tensor_ref_vec.push(TensorRef { tensor_ref, is_input, is_output });
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
			let mut parents =
				std::mem::replace(&mut self.nodes_postorder[i].parents, ThinVec::new());
			parents.retain(|&p| !self.nodes_postorder[p].is_dead);
			if parents.is_empty() {
				if self.nodes_postorder[i].capture.is_empty() {
					self.nodes_postorder[i].is_dead = true;
				}
			} else {
				parents.sort_unstable();
				parents.dedup();
				parents.shrink_to_fit();
				self.nodes_postorder[i].parents = parents;
			}
		}
	}

	pub fn calc_shapes(&mut self) -> Result<(), TensorOpError> {
		if self.tensor_shapes.len() != self.tensor_ref_vec.len() {
			self.tensor_shapes =
				IndexVec::from_vec(vec![ThinVec::new(); self.tensor_ref_vec.len()]);
		}

		for i in self.tensor_ref_vec.indexes() {
			self.tensor_shapes[i].clear();
			if !self.tensor_ref_vec[i].is_input {
				continue;
			}

			let tensor_borrow =
				unsafe { self.tensor_ref_vec[i].tensor_ref.tensor.try_borrow_unguarded() };
			let Ok(tensor) = tensor_borrow else {
				cold_path();
				return Err(TensorOpError::CannotBorrow);
			};
			let Some(tensor) = tensor else {
				cold_path();
				return Err(TensorOpError::MissingInput);
			};

			self.tensor_shapes[i].extend_from_slice(tensor.shape());
		}

		for i in self.nodes_postorder.indexes() {
			let (prev, me, _) = self.nodes_postorder.borrow_multiple(i);
			me.shape.clear();
			if me.is_tensor_input() {
				// For input nodes, get shape from tensor_ref
				let tensor_ref_index = TensorRefIndex::new(me.input_index_raw);
				me.shape.extend_from_slice(&self.tensor_shapes[tensor_ref_index]);
			} else if me.is_unary() {
				// For unary operations, shape is the same as input
				let child = me.children[0];
				let child = &prev[child.raw];
				me.shape.extend_from_slice(&child.shape);
				if me.is_reduction() {
					if let Some(last) = me.shape.last_mut() {
						*last = 1;
					} else {
						me.shape.push(1);
					}
				}
			} else if me.is_binary() {
				// For binary operations, use broadcast to get output shape
				let left = me.children[0];
				let right = me.children[1];
				let left = &prev[left.raw];
				let right = &prev[right.raw];
				let shape_len = left.shape.len().max(right.shape.len());
				let skip_left = shape_len - left.shape.len();
				let skip_right = shape_len - right.shape.len();
				for d in 0..shape_len {
					let dim_left = if d < skip_left { 1 } else { left.shape[d - skip_left] };
					let dim_right = if d < skip_right { 1 } else { right.shape[d - skip_right] };
					let dim = if dim_left == dim_right || dim_right == 1 {
						dim_left
					} else if dim_left == 1 {
						dim_right
					} else {
						return Err(TensorOpError::ShapeMismatch);
					};
					me.shape.push(dim);
				}
			} else {
				debug_assert!(self.nodes_postorder[i].is_scalar_input());
			}
		}

		Ok(())
	}

	fn find_reduction_heads(&mut self) {
		for parent_idx in self.nodes_postorder.indexes() {
			let (prev, parent, _) = self.nodes_postorder.borrow_multiple(parent_idx);
			if parent.is_reduction() {
				parent.is_reduction_head = true;
			} else {
				let shape: &[usize] = &parent.shape;
				for child in parent.children {
					if !child.is_valid() {
						break;
					}
					let child = &mut prev[child.raw];
					if child.is_reduction_head
						&& &child.parents == &[parent_idx]
						&& child.shape == shape
					{
						child.is_reduction_head = false;
						parent.is_reduction_head = true;
					}
				}
			}
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

	pub fn graphviz_node_label(&self, node: &Node) -> String {
		match node.expr.as_ref() {
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

	fn shape_to_str(&self, shape: &[usize]) -> String {
		let mut result = String::from("[");
		for (i, &dim) in shape.iter().enumerate() {
			if i > 0 {
				result.push(',');
			}
			result.push_str(&dim.to_string());
		}
		result.push(']');
		result
	}

	pub fn print_graphviz<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		writeln!(w, "\trankdir=BT;")?;
		for i in self.nodes_postorder.indexes() {
			let node = &self.nodes_postorder[i];
			let node_id = self.graphviz_node_id(i);
			let extra_label = "";
			if node.is_input() {
				continue;
			}
			writeln!(
				w,
				"\t\t{node_id} [label=<{}{extra_label}>];",
				self.graphviz_node_label(node),
			)?;
			if node.is_reduction_head {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccccff\"];")?;
			} else if node.is_fork() {
				if node.is_dead {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#cccccc\"];")?;
				} else {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffcccc\"];")?;
				}
			} else if node.is_reduction() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffccff\"];")?;
			} else if node.is_captured() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccffcc\"];")?;
			}
			/*
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
			*/
			for &child_index in &node.children {
				if !child_index.is_valid() {
					break;
				}
				let child_id = self.graphviz_node_id(child_index);
				let label = if self.nodes_postorder[child_index].out_is_scalar {
					String::new()
				} else {
					self.shape_to_str(&self.nodes_postorder[child_index].shape)
				};
				let extra_style = /*if node.fragment.is_valid()
					&& self.nodes_postorder[child_index].fragment.is_valid()
					&& node.fragment != self.nodes_postorder[child_index].fragment
				{
					", color=red, style=bold"
				} else*/ {
					""
				};
				writeln!(w, "\t{child_id} -> {node_id} [label=\"{}\"{}];", label, extra_style)?;
			}
			for &capt_idx in &node.capture {
				let label =
					if node.out_is_scalar { String::new() } else { self.shape_to_str(&node.shape) };
				let capt = &self.tensor_ref_vec[capt_idx].tensor_ref;
				let cap_id = self.graphviz_tensor_id(capt);
				let key = std::ptr::from_ref(capt.as_ref());
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
		for scalar_ref in &self.scalar_ref_vec {
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
}

//--------------------------------------------------------------------------------------------------
