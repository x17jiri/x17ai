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

use std::borrow::Cow;
use std::collections::{HashMap, hash_map};
use std::hint::{cold_path, unlikely};
use std::rc::Rc;

use thin_vec::ThinVec;

use super::{
	Expr, ExprBinaryKind, ExprInput, ExprReductionKind, ExprScalarRef, ExprTensorRef, ExprUnary,
	ExprUnaryKind, RcExpr,
};
use crate::new::expr::{ExprCapture, ExprCast};
use crate::tensor::TensorOpError;
use crate::util::bitmap::IndexBitmap;
use crate::util::index_vec::{IndexTrait, IndexVec, UntypedIndex32};
use crate::{ErrPack, define_index_type32};

//--------------------------------------------------------------------------------------------------

pub struct ShapeRef<'a> {
	pub shape: &'a [usize],
}

impl<'a> ShapeRef<'a> {
	pub fn new(shape: &'a [usize]) -> Self {
		Self { shape }
	}

	pub fn zip_with<'b>(
		&'a self,
		other: &'b ShapeRef,
	) -> impl ExactSizeIterator<Item = (usize, usize)> + DoubleEndedIterator {
		let shape_len = self.shape.len().max(other.shape.len());
		let skip_self = shape_len - self.shape.len();
		let skip_other = shape_len - other.shape.len();
		(0..shape_len).map(move |d| {
			let idx_self = d.wrapping_sub(skip_self);
			let dim_self = *self.shape.get(idx_self).unwrap_or(&1);
			let idx_other = d.wrapping_sub(skip_other);
			let dim_other = *other.shape.get(idx_other).unwrap_or(&1);
			(dim_self, dim_other)
		})
	}
}

impl<'a> std::cmp::PartialEq for ShapeRef<'a> {
	fn eq(&self, other: &Self) -> bool {
		self.zip_with(other).all(|(a, b)| a == b)
	}
}

impl<'a> std::cmp::Eq for ShapeRef<'a> {
}

#[allow(clippy::struct_excessive_bools)]
pub struct Node {
	pub expr: Rc<Expr>,
	pub shape: ThinVec<usize>,
	pub parents: ThinVec<NodeIndex32>,
	pub dominator: NodeIndex32,
	pub children: [NodeIndex32; 2],
	pub capture: ThinVec<TensorRefIndex32>,

	pub out_is_scalar: bool,
	pub is_dead: bool,
	pub reduction_fingerprint: u32,
	pub reduction_head_for: NodeIndex32,

	// This is one of:
	// - ScalarRefIndex32
	// - TensorRefIndex32
	// - FragmentIndex32
	// depending on the node type.
	pub x_index: UntypedIndex32,
}

impl Node {
	pub fn is_input(&self) -> bool {
		self.is_nullary()
	}

	pub fn scalar_index(&self) -> ScalarRefIndex32 {
		debug_assert!(self.is_scalar_input());
		ScalarRefIndex32::from(self.x_index)
	}

	pub fn tensor_index(&self) -> TensorRefIndex32 {
		debug_assert!(self.is_tensor_input());
		TensorRefIndex32::from(self.x_index)
	}

	pub fn fragment_index(&self) -> FragmentIndex32 {
		debug_assert!(!self.is_input());
		FragmentIndex32::from(self.x_index)
	}

	pub fn set_fragment_index(&mut self, fragment_index: FragmentIndex32) {
		debug_assert!(!self.is_input());
		self.x_index = fragment_index.to_untyped();
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

	pub fn is_reduction_head(&self) -> bool {
		!self.reduction_head_for.is_sentinel()
	}

	pub fn is_captured(&self) -> bool {
		!self.capture.is_empty()
	}

	pub fn is_nullary(&self) -> bool {
		self.children[0].is_sentinel()
	}

	pub fn is_unary(&self) -> bool {
		!self.children[0].is_sentinel() && self.children[1].is_sentinel()
	}

	pub fn is_binary(&self) -> bool {
		!self.children[1].is_sentinel()
	}

	pub fn is_root(&self) -> bool {
		self.parents.is_empty()
	}

	pub fn is_fork(&self) -> bool {
		self.parents.len() != 1 && !self.is_input()
	}
}

define_index_type32!(NodeIndex32);
type NodeVec = IndexVec<NodeIndex32, Node>;

pub struct TensorRef {
	pub tensor_ref: Rc<ExprTensorRef>,
	pub input_node: NodeIndex32,
	pub output_node: NodeIndex32,
	pub shape: ThinVec<usize>,
}

define_index_type32!(TensorRefIndex32);
type TensorVec = IndexVec<TensorRefIndex32, TensorRef>;

define_index_type32!(ScalarRefIndex32);
type ScalarVec = IndexVec<ScalarRefIndex32, Rc<ExprScalarRef>>;

pub struct Fragment {
	pub head: NodeIndex32,
}

define_index_type32!(FragmentIndex32);
type FragmentVec = IndexVec<FragmentIndex32, Fragment>;

pub struct PreCompilation {
	nodes_postorder: NodeVec,
	fragments_preorder: FragmentVec,
	scalar_map: HashMap<*const ExprScalarRef, ScalarRefIndex32>,
	scalar_vec: ScalarVec,
	tensor_map: HashMap<*const ExprTensorRef, TensorRefIndex32>,
	tensor_vec: TensorVec,
}

pub struct LoadExprState {
	visited: HashMap<*const Expr, NodeIndex32>,
	n_reductions: u32,
	bitmap: IndexBitmap<NodeIndex32>,
}

impl PreCompilation {
	pub fn new(expr: RcExpr) -> Result<Self, ErrPack<TensorOpError>> {
		let mut comp = PreCompilation {
			nodes_postorder: NodeVec::with_capacity(32),
			fragments_preorder: FragmentVec::with_capacity(8),
			scalar_map: HashMap::new(),
			scalar_vec: ScalarVec::with_capacity(4),
			tensor_map: HashMap::new(),
			tensor_vec: TensorVec::with_capacity(4),
		};
		let mut state = LoadExprState {
			visited: HashMap::new(),
			n_reductions: 0,
			bitmap: IndexBitmap::new(),
		};
		comp.load_expr(expr.rc_expr, &mut state);
		comp.remove_dead_code();
		comp.find_dominators();
		comp.find_reduction_fingerprints(&mut state);
		comp.find_races(&mut state)?;
		Ok(comp)
	}

	#[allow(clippy::collapsible_else_if)]
	#[allow(clippy::manual_assert)]
	#[allow(clippy::panic)]
	// TODO - refactor to make non recursive
	fn load_expr(&mut self, mut expr: Rc<Expr>, state: &mut LoadExprState) -> NodeIndex32 {
		let expr_key = std::ptr::from_ref(expr.as_ref());
		if let Some(index) = state.visited.get(&expr_key) {
			return *index;
		}

		let mut x_index = UntypedIndex32::new_sentinel();
		let out_is_scalar: bool;
		let children: [NodeIndex32; 2];
		let mut capture: ThinVec<TensorRefIndex32> = ThinVec::new();
		match expr.as_ref() {
			Expr::Capture(ExprCapture { expr: x, tensor_ref }) => {
				let child = self.load_expr(x.clone(), state);
				let tensor_ref_index =
					self.add_tensor_output(tensor_ref.clone(), self.nodes_postorder.next_index());
				if !self.nodes_postorder[child].is_input() {
					self.nodes_postorder[child].capture.push(tensor_ref_index);
					state.visited.insert(expr_key, child);
					return child;
				}
				// Insert `Identity` node to perform the capture.
				// This node is also a root.
				expr = Rc::new(Expr::Unary(ExprUnary {
					kind: ExprUnaryKind::Identity,
					expr: self.nodes_postorder[child].expr.clone(),
				}));
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				children = [child, NodeIndex32::new_sentinel()];
				capture.push(tensor_ref_index);
			},
			Expr::First(first) => {
				let first_child = self.load_expr(first.lhs.clone(), state);
				let _second_child = self.load_expr(first.rhs.clone(), state);
				state.visited.insert(expr_key, first_child);
				return first_child;
			},
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					out_is_scalar = false;
					children = [NodeIndex32::new_sentinel(), NodeIndex32::new_sentinel()];
					match self
						.add_tensor_input(tensor_ref.clone(), self.nodes_postorder.next_index())
					{
						Ok(tensor_index) => {
							x_index = tensor_index.to_untyped();
						},
						Err(existing_node) => {
							state.visited.insert(expr_key, existing_node);
							return existing_node;
						},
					}
				},
				ExprInput::Scalar(scalar_ref) => {
					out_is_scalar = true;
					children = [NodeIndex32::new_sentinel(), NodeIndex32::new_sentinel()];
					x_index = self.add_scalar_input(scalar_ref.clone()).to_untyped();
				},
			},
			Expr::Cast(ExprCast { expr, .. }) | Expr::Unary(ExprUnary { expr, .. }) => {
				let child = self.load_expr(expr.clone(), state);
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				children = [child, NodeIndex32::new_sentinel()];
			},
			Expr::Binary(binary) => {
				let lhs = binary.lhs.clone();
				let rhs = binary.rhs.clone();
				let left_child = self.load_expr(lhs, state);
				let right_child = self.load_expr(rhs, state);
				let left_is_scalar = self.nodes_postorder[left_child].out_is_scalar;
				let right_is_scalar = self.nodes_postorder[right_child].out_is_scalar;
				out_is_scalar = left_is_scalar && right_is_scalar;
				children = [left_child, right_child];
			},
			Expr::Reduction(reduction) => {
				let reduction_expr = reduction.expr.clone();
				let child = self.load_expr(reduction_expr, state);
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				children = [child, NodeIndex32::new_sentinel()];
				state.n_reductions += 1;
			},
		}

		let next_index = self.nodes_postorder.next_index();
		for &child in &children {
			if !self.nodes_postorder.is_valid(child) {
				break;
			}
			self.nodes_postorder[child].parents.push(next_index);
		}

		let index = self.nodes_postorder.push(Node {
			expr,
			shape: ThinVec::new(),
			parents: ThinVec::new(),
			dominator: NodeIndex32::new_sentinel(),
			children,
			capture,
			out_is_scalar,
			is_dead: false,
			reduction_fingerprint: 0,
			reduction_head_for: NodeIndex32::new_sentinel(),
			x_index,
		});
		debug_assert!(index == next_index);
		state.visited.insert(expr_key, index);
		index
	}

	fn add_scalar_input(&mut self, scalar_ref: Rc<ExprScalarRef>) -> ScalarRefIndex32 {
		let key = std::ptr::from_ref(scalar_ref.as_ref());
		match self.scalar_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index = self.scalar_vec.push(scalar_ref);
				entry.insert(index);
				index
			},
			hash_map::Entry::Occupied(entry) => {
				let index = *entry.get();
				index
			},
		}
	}

	fn add_tensor_input(
		&mut self,
		tensor_ref: Rc<ExprTensorRef>,
		node: NodeIndex32,
	) -> Result<TensorRefIndex32, NodeIndex32> {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		match self.tensor_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index = self.tensor_vec.push(TensorRef {
					tensor_ref,
					input_node: node,
					output_node: NodeIndex32::new_sentinel(),
					shape: ThinVec::new(),
				});
				entry.insert(index);
				Ok(index)
			},
			hash_map::Entry::Occupied(entry) => {
				let index = *entry.get();
				Err(self.tensor_vec[index].input_node)
			},
		}
	}

	fn add_tensor_output(
		&mut self,
		tensor_ref: Rc<ExprTensorRef>,
		node: NodeIndex32,
	) -> TensorRefIndex32 {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		match self.tensor_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index = self.tensor_vec.push(TensorRef {
					tensor_ref,
					input_node: NodeIndex32::new_sentinel(),
					output_node: node,
					shape: ThinVec::new(),
				});
				entry.insert(index);
				index
			},
			hash_map::Entry::Occupied(entry) => *entry.get(),
		}
	}

	fn find_reduction_fingerprints(&mut self, state: &mut LoadExprState) {
		let bitmap = &mut state.bitmap;
		bitmap.clear_and_resize(&self.nodes_postorder, state.n_reductions as usize);
		let mut n_reductions = 0;
		for idx in self.nodes_postorder.indexes() {
			let me = &mut self.nodes_postorder[idx];
			if unlikely(me.is_dead) {
				continue;
			}
			if me.is_binary() {
				let left_child = me.children[0];
				let right_child = me.children[1];
				bitmap.union(idx, left_child, right_child);
			} else if me.is_unary() {
				let child = me.children[0];
				bitmap.copy_row(idx, child);
				if me.is_reduction() {
					bitmap.set_bit(idx, n_reductions);
					n_reductions += 1;
				}
			}
		}
		debug_assert!(n_reductions == (state.n_reductions as usize));
		let mut fingerprints: HashMap<&[usize], u32> = HashMap::new();
		for idx in self.nodes_postorder.indexes() {
			let row: &[usize] = bitmap.row(idx);
			#[allow(clippy::cast_possible_truncation)]
			let next_fingerprint = fingerprints.len() as NodeIndex32::RawType;
			let fingerprint = match fingerprints.entry(row) {
				hash_map::Entry::Vacant(entry) => {
					entry.insert(next_fingerprint);
					next_fingerprint
				},
				hash_map::Entry::Occupied(entry) => *entry.get(),
			};
			self.nodes_postorder[idx].reduction_fingerprint = fingerprint;
		}
	}

	#[allow(clippy::manual_assert)]
	fn find_races(&self, state: &mut LoadExprState) -> Result<(), ErrPack<TensorOpError>> {
		let kills = NodeIndex32::from_raw(self.nodes_postorder.len());
		let rows = self.nodes_postorder.len() + 1;
		let cols = self.tensor_vec.len();
		let bitmap = &mut state.bitmap;
		bitmap.raw.clear_and_resize(rows, cols);
		for i in self.nodes_postorder.indexes() {
			let me = &self.nodes_postorder[i];
			if me.is_nullary() {
				if me.is_tensor_input() {
					let tensor_index = me.tensor_index();
					bitmap.set_bit(i, tensor_index.to_raw());
				}
			} else {
				if me.is_binary() {
					let left_child = me.children[0];
					let right_child = me.children[1];
					bitmap.union(i, left_child, right_child);
				} else if me.is_unary() {
					let child = me.children[0];
					bitmap.copy_row(i, child);
				}
				if bitmap.have_common_bits(i, kills) {
					cold_path();
					let mut message = String::new();
					let w: &mut dyn std::fmt::Write = &mut message;
					for c in 0..cols {
						if bitmap.get_bit(i, c) && bitmap.get_bit(kills, c) {
							let _ = writeln!(
								w,
								"Ambiguous use of tensor {}. Not clear whether to use the version before or after write.",
								self.tensor_vec[TensorRefIndex32::from_raw(c)]
									.tensor_ref
									.name
									.as_deref()
									.unwrap_or("<unnamed>")
							);
						}
					}
					return Err(ErrPack {
						code: TensorOpError::WriteReadRace,
						extra: Some(Box::new(crate::ErrExtra {
							message: Cow::from(message),
							nested: None,
						})),
					});
				}
				for &tensor_index in &me.capture {
					let was_killed = bitmap.set_bit(kills, tensor_index.to_raw());
					if was_killed {
						cold_path();
						return Err(ErrPack {
							code: TensorOpError::DoubleWrite,
							extra: Some(Box::new(crate::ErrExtra {
								message: Cow::from(format!(
									"Double write to tensor {}",
									self.tensor_vec[tensor_index]
										.tensor_ref
										.name
										.as_deref()
										.unwrap_or("<unnamed>")
								)),
								nested: None,
							})),
						});
					}
				}
				bitmap.and_not(i, i, kills);
			}
		}
		Ok(())
	}

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

	fn find_dominators(&mut self) {
		let mut changed = true;
		while changed {
			changed = false;
			for idx in self.nodes_postorder.indexes().rev() {
				match self.nodes_postorder[idx].parents.len() {
					0 => {
						// root
						debug_assert!(
							self.nodes_postorder[idx].dominator == NodeIndex32::new_sentinel()
						);
					},
					1 => {
						self.nodes_postorder[idx].dominator = self.nodes_postorder[idx].parents[0];
					},
					_ => {
						let dominator = self.nodes_postorder[idx].parents[1..]
							.iter()
							.copied()
							.fold(self.nodes_postorder[idx].parents[0], |mut d1, mut d2| {
								while d1 != d2 {
									if d1.to_raw() < d2.to_raw() {
										d1 = self.nodes_postorder[d1].dominator;
									} else {
										d2 = self.nodes_postorder[d2].dominator;
									}
								}
								d1
							});
						if self.nodes_postorder[idx].dominator != dominator {
							self.nodes_postorder[idx].dominator = dominator;
							changed = true;
						}
					},
				}
			}
		}
	}

	fn calc_shapes(&mut self) -> Result<(), TensorOpError> {
		for t in &mut self.tensor_vec {
			t.shape.clear();
			if t.input_node.is_sentinel() {
				continue;
			}

			let borrow = unsafe { t.tensor_ref.tensor.try_borrow_unguarded() };
			let Ok(tensor) = borrow else {
				cold_path();
				return Err(TensorOpError::CannotBorrow);
			};
			let Some(tensor) = tensor else {
				cold_path();
				return Err(TensorOpError::MissingInput);
			};

			t.shape.extend_from_slice(tensor.shape());
		}

		for i in self.nodes_postorder.indexes() {
			let (all_children, me, _) = self.nodes_postorder.borrow_multiple(i);
			me.shape.clear();
			if me.is_input() {
				if me.is_tensor_input() {
					// For input nodes, get shape from tensor_ref
					me.shape.extend_from_slice(&self.tensor_vec[me.tensor_index()].shape);
				}
				continue;
			}

			if me.is_unary() {
				// For unary operations, shape is the same as input
				let child = me.children[0];
				let child = &all_children[child];
				me.shape.extend_from_slice(&child.shape);
				if me.is_reduction() {
					if let Some(last) = me.shape.last_mut() {
						*last = 1;
					} else {
						me.shape.push(1);
					}
				}
			} else {
				debug_assert!(me.is_binary());

				// For binary operations, use broadcast to get output shape
				let left = me.children[0];
				let right = me.children[1];
				let left = &all_children[left];
				let right = &all_children[right];
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
						cold_path();
						return Err(TensorOpError::ShapeMismatch);
					};
					me.shape.push(dim);
				}
			}

			// If we store back into inputs, make sure captures have correct shape
			let my_shape = me.shape.as_slice();
			for &idx in &me.capture {
				if !self.tensor_vec[idx].input_node.is_sentinel() {
					let tensor_shape = self.tensor_vec[idx].shape.as_slice();
					if tensor_shape != my_shape {
						cold_path();
						return Err(TensorOpError::ShapeMismatch);
					}
				}
			}
		}

		Ok(())
	}

	fn find_reduction_heads(&mut self) {
		for idx in self.nodes_postorder.indexes() {
			if !self.nodes_postorder[idx].is_reduction() {
				continue;
			}

			let start = &self.nodes_postorder[idx];
			let start_shape = ShapeRef::new(&start.shape);
			let start_fingerprint = start.reduction_fingerprint;

			let mut head_idx = idx;
			let mut head = start;
			loop {
				let parent_idx = head.dominator;
				if !self.nodes_postorder.is_valid(parent_idx) {
					break;
				}
				let parent = &self.nodes_postorder[parent_idx];
				let parent_shape = ShapeRef::new(&parent.shape);
				let parent_fingerprint = parent.reduction_fingerprint;
				if parent_shape != start_shape || parent_fingerprint != start_fingerprint {
					break;
				}
				head_idx = parent_idx;
				head = parent;
			}

			self.nodes_postorder[head_idx].reduction_head_for = idx;
		}
	}

	pub fn find_fragments(&mut self) -> Result<(), TensorOpError> {
		self.calc_shapes()?;
		self.find_reduction_heads();
		self.fragments_preorder.raw.clear();
		for idx in self.nodes_postorder.indexes().rev() {
			let (_, item, all_parents) = self.nodes_postorder.borrow_multiple(idx);
			if item.is_input() || unlikely(item.is_dead) {
				continue;
			}
			if let Some((&first_parent, other_parents)) = item.parents.split_first()
				&& let parent_frag = all_parents[first_parent].fragment_index()
				&& !item.is_reduction_head()
				&& other_parents.iter().all(|&p| all_parents[p].fragment_index() == parent_frag)
			{
				item.set_fragment_index(parent_frag);
			} else {
				let new_frag = self.fragments_preorder.push(Fragment { head: idx });
				item.set_fragment_index(new_frag);
			}
		}
		Ok(())
	}

	pub fn graphviz_node_label(&self, node: &Node) -> String {
		match node.expr.as_ref() {
			Expr::Input(input) => match input {
				ExprInput::Tensor(tensor_ref) => {
					let name = tensor_ref.name.as_deref().unwrap_or("<unnamed>");
					format!("<b>Tensor</b><br/><font color='blue'><b>{name}</b></font>")
				},
				ExprInput::Scalar(scalar_ref) => {
					let name = scalar_ref.name.as_deref().unwrap_or("<unnamed>");
					format!("<b>Scalar</b><br/><font color='blue'><b>{name}</b></font>")
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

	#[allow(clippy::too_many_lines)]
	pub fn print_graphviz<W: std::fmt::Write>(
		&self,
		w: &mut W,
		mut state: Option<&mut LoadExprState>,
	) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		writeln!(w, "\trankdir=BT;")?;
		//writeln!(w, "\tsplines=polyline;")?;
		for i in self.nodes_postorder.indexes() {
			let node = &self.nodes_postorder[i];
			let node_id = format!("node_{}", i.raw);
			let extra_label = if let Some(state) = &mut state {
				let mut names = Vec::new();
				for (idx, ten) in self.tensor_vec.iter().enumerate() {
					if state.bitmap.get_bit(i, idx) {
						let ten_name = if let Some(name) = &ten.tensor_ref.name {
							name.as_ref().to_string()
						} else {
							format!("{:?}", std::ptr::from_ref(ten.tensor_ref.as_ref()))
						};
						names.push(ten_name);
					}
				}
				format!("<br/><font color='red'>In use: {}</font>", names.join(", "))
			} else {
				String::new()
			};
			writeln!(w, "\t{node_id} [label=<{}{extra_label}>];", self.graphviz_node_label(node),)?;
			if node.is_input() {
				if node.is_tensor_input() {
					writeln!(w, "\t{node_id} [shape=box, style=filled, fillcolor=\"#cceecc\"];")?;
				} else {
					writeln!(w, "\t{node_id} [shape=box, style=filled, fillcolor=\"#ffffc0\"];")?;
				}
			} else if node.is_reduction_head() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccccff\"];")?;
			} else if node.is_fork() {
				if unlikely(node.is_dead) {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#cccccc\"];")?;
				} else {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffcccc\"];")?;
				}
			} else if node.is_reduction() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffccff\"];")?;
			} else if node.is_captured() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccffcc\"];")?;
			}
			if !node.dominator.is_sentinel() {
				let dom_id = format!("node_{}", node.dominator.raw);
				writeln!(
					w,
					"\t{node_id} -> {dom_id} [label=< > style=dashed, color=\"#8080ff\", constraint=true];"
				)?;
			}
			if node.is_input() {
				//writeln!(w, "\t{{ rank = min; {node_id} }}")?;
			} else {
				let frag_index = node.fragment_index();
				if self.fragments_preorder.is_valid(frag_index) {
					let frag_head = self.fragments_preorder[frag_index].head;
					let fragment_kind = if self.nodes_postorder[frag_head].is_reduction_head() {
						"Reduction"
					} else {
						"Element-wise"
					};
					writeln!(
						w,
						"\tsubgraph cluster_{} {{ label=\"{fragment_kind}\" labelloc=\"b\" labeljust=\"l\" {node_id} }}",
						frag_head.raw
					)?;
				}
				for &child_index in &node.children {
					if !self.nodes_postorder.is_valid(child_index) {
						break;
					}
					let child_id = format!("node_{}", child_index.raw);
					let child = &self.nodes_postorder[child_index];
					let label = if child.out_is_scalar {
						String::new()
					} else {
						self.shape_to_str(&child.shape)
					};
					let extra_style = if self.fragments_preorder.is_valid(frag_index)
						&& !child.is_input()
						&& frag_index != child.fragment_index()
					{
						// Edge crosses fragment boundary
						", color=red, style=bold"
					} else {
						""
					};
					writeln!(
						w,
						"\t{child_id} -> {node_id} [label=\"{}\"{}, constraint=true];",
						label, extra_style
					)?;
				}
			}
			for &capt_idx in &node.capture {
				let label =
					if node.out_is_scalar { String::new() } else { self.shape_to_str(&node.shape) };
				let input_node = self.tensor_vec[capt_idx].input_node;
				if input_node.is_sentinel() {
					let cap_id = format!("ten_{}", capt_idx.raw);
					writeln!(w, "\t{node_id} -> {cap_id} [label=\"{label}\", constraint=true];")?;
					let tensor_ref = &self.tensor_vec[capt_idx].tensor_ref;
					let name = tensor_ref.name.as_deref().unwrap_or("<unnamed>");
					writeln!(
						w,
						"{cap_id} [label=<<b>Tensor</b><br/><font color='blue'><b>{name}</b></font>>, shape=box, style=filled, fillcolor=\"#cceeff\"];"
					)?;
				} else {
					let cap_id = format!("node_{}", input_node.raw);
					writeln!(w, "\t{node_id} -> {cap_id} [label=\"{label}\", constraint=true];")?;
				}
			}
		}
		writeln!(w, "}}")?;
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
