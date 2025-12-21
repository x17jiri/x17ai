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
	pub reduction_head_for: NodeIndex32,
	pub has_all_reductions: [bool; 2],

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
		self.reduction_head_for.is_valid()
	}

	pub fn is_captured(&self) -> bool {
		!self.capture.is_empty()
	}

	pub fn is_nullary(&self) -> bool {
		self.children[0].to_raw_isize() < 0
	}

	pub fn is_unary(&self) -> bool {
		(self.children[0].to_raw_isize() | !self.children[1].to_raw_isize()) >= 0
	}

	pub fn is_binary(&self) -> bool {
		self.children[1].to_raw_isize() >= 0
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
	pub is_input: bool,
	pub is_output: bool,
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
		comp.find_reduction_uses(&mut state);
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

		let mut x_index = UntypedIndex32::new_invalid();
		let out_is_scalar: bool;
		let children: [NodeIndex32; 2];
		let mut capture: ThinVec<TensorRefIndex32> = ThinVec::new();
		match expr.as_ref() {
			Expr::Capture(ExprCapture { expr: x, tensor_ref }) => {
				let child = self.load_expr(x.clone(), state);
				let tensor_ref = tensor_ref.clone();
				let tensor_ref_index = self.add_tensor_ref(tensor_ref, false, true);
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
				children = [child, NodeIndex32::new_invalid()];
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
					children = [NodeIndex32::new_invalid(), NodeIndex32::new_invalid()];
					x_index = self.add_tensor_ref(tensor_ref.clone(), true, false).to_untyped();
				},
				ExprInput::Scalar(scalar_ref) => {
					out_is_scalar = true;
					children = [NodeIndex32::new_invalid(), NodeIndex32::new_invalid()];
					x_index = self.add_scalar_ref(scalar_ref.clone()).to_untyped();
				},
			},
			Expr::Cast(ExprCast { expr, .. }) | Expr::Unary(ExprUnary { expr, .. }) => {
				let child = self.load_expr(expr.clone(), state);
				out_is_scalar = self.nodes_postorder[child].out_is_scalar;
				children = [child, NodeIndex32::new_invalid()];
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
				children = [child, NodeIndex32::new_invalid()];
				state.n_reductions += 1;
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
			dominator: NodeIndex32::new_invalid(),
			children,
			capture,
			out_is_scalar,
			is_dead: false,
			reduction_head_for: NodeIndex32::new_invalid(),
			has_all_reductions: [true; 2],
			x_index,
		});
		debug_assert!(index == next_index);
		state.visited.insert(expr_key, index);
		index
	}

	fn add_scalar_ref(&mut self, scalar_ref: Rc<ExprScalarRef>) -> ScalarRefIndex32 {
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

	fn add_tensor_ref(
		&mut self,
		tensor_ref: Rc<ExprTensorRef>,
		is_input: bool,
		is_output: bool,
	) -> TensorRefIndex32 {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		match self.tensor_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index = self.tensor_vec.push(TensorRef {
					tensor_ref,
					is_input,
					is_output,
					shape: ThinVec::new(),
				});
				entry.insert(index);
				index
			},
			hash_map::Entry::Occupied(entry) => {
				let index = *entry.get();
				if is_input {
					self.tensor_vec[index].is_input = true;
				}
				if is_output {
					self.tensor_vec[index].is_output = true;
				}
				index
			},
		}
	}

	fn find_reduction_uses(&mut self, state: &mut LoadExprState) {
		let bitmap = &mut state.bitmap;
		bitmap.clear_and_resize(&self.nodes_postorder, state.n_reductions as usize);
		let mut n_reductions = 0;
		for i in self.nodes_postorder.indexes() {
			let me = &mut self.nodes_postorder[i];
			if unlikely(me.is_dead) {
				continue;
			}
			if me.is_binary() {
				let left_child = me.children[0];
				let right_child = me.children[1];
				me.has_all_reductions = bitmap.check_inclusion(left_child, right_child);
				bitmap.union(i, left_child, right_child);
			} else if me.is_unary() {
				let child = me.children[0];
				bitmap.copy_row(i, child);
				if me.is_reduction() {
					bitmap.set_bit(i, n_reductions);
					n_reductions += 1;
				}
			}
		}
		debug_assert!(n_reductions == (state.n_reductions as usize));
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
		while (changed) {
			changed = false;
			for idx in self.nodes_postorder.indexes().rev() {
				match self.nodes_postorder[idx].parents.len() {
					0 => {
						// root
						debug_assert!(
							self.nodes_postorder[idx].dominator.to_raw() == u32::MAX as usize
						);
					},
					1 => {
						self.nodes_postorder[idx].dominator = self.nodes_postorder[idx].parents[0];
					},
					_ => {
						let dominator = self.nodes_postorder[idx].parents[1..].iter().fold(
							self.nodes_postorder[self.nodes_postorder[idx].parents[0]].dominator,
							|mut d1, &p| {
								let mut d2 = self.nodes_postorder[p].dominator;
								while d1 != d2 {
									if d1.to_raw() < d2.to_raw() {
										d1 = self.nodes_postorder[d1].dominator;
									} else {
										d2 = self.nodes_postorder[d2].dominator;
									}
								}
								d1
							},
						);
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
			if !t.is_input {
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
				if self.tensor_vec[idx].is_input {
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
		#[derive(Clone, Copy)]
		struct Item {
			token: NodeIndex32,
			count: usize,
			head: NodeIndex32,
		}
		let mut t: IndexVec<NodeIndex32, Item> = IndexVec::from_vec(vec![
			Item {
				token: NodeIndex32::new_invalid(),
				count: 0,
				head: NodeIndex32::new_invalid()
			};
			self.nodes_postorder.len()
		]);
		for idx in self.nodes_postorder.indexes() {
			let (mut prev, child, all_parents) = self.nodes_postorder.borrow_multiple(idx);
			child.reduction_head_for = NodeIndex32::new_invalid();
			if child.is_reduction() {
				t[idx].token = idx;
				t[idx].count = 1;
			}
			let token = t[idx].token;
			if !token.is_valid() {
				continue;
			}
			if t[idx].count == t[token].count {
				let head = t[token].head;
				if head.is_valid() {
					prev[head].reduction_head_for = NodeIndex32::new_invalid();
				}
				t[token].head = idx;
				child.reduction_head_for = token;
			}

			// The current node has a token; try to propagate it to parents
			let mut eligible_parents = 0;
			for &p in &child.parents {
				let parent = &all_parents[p];
				if !parent.is_reduction()
					&& parent
						.children
						.iter()
						.zip(parent.has_all_reductions)
						.any(|(&c, has_all)| c == idx && has_all)
					&& ShapeRef::new(&parent.shape) == ShapeRef::new(&child.shape)
				{
					eligible_parents += 1;
				}
			}

			if eligible_parents > 0 && eligible_parents == child.parents.len() {
				t[token].count = t[token].count - t[idx].count + eligible_parents;
				t[idx].token = NodeIndex32::new_invalid();
				for &p in &child.parents {
					t[p].token = token;
					t[p].count += 1;
				}
			}
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

	fn graphviz_tensor_id(&self, tensor_ref: &Rc<ExprTensorRef>) -> String {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		if let Some(&index) = self.tensor_map.get(&key) {
			format!("ten_{}", index.raw)
		} else {
			format!("ten_{}", key as usize)
		}
	}

	fn graphviz_scalar_id(&self, scalar_ref: &Rc<ExprScalarRef>) -> String {
		let key = std::ptr::from_ref(scalar_ref.as_ref());
		if let Some(&index) = self.scalar_map.get(&key) {
			format!("sca_{}", index.raw)
		} else {
			format!("sca_{}", key as usize)
		}
	}

	fn graphviz_node_id(&self, node_index: NodeIndex32) -> String {
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

	#[allow(clippy::too_many_lines)]
	pub fn print_graphviz<W: std::fmt::Write>(
		&self,
		w: &mut W,
		mut state: Option<&mut LoadExprState>,
	) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		writeln!(w, "\trankdir=BT;")?;
		for i in self.nodes_postorder.indexes() {
			let node = &self.nodes_postorder[i];
			let node_id = self.graphviz_node_id(i);
			let mut extra_label = String::new();
			if let Some(state) = &mut state {
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
				extra_label = format!("<br/><font color='red'>In use: {}</font>", names.join(", "));
			}
			if node.is_input() {
				continue;
			}
			writeln!(w, "\t{node_id} [label=<{}{extra_label}>];", self.graphviz_node_label(node),)?;
			if node.is_reduction_head() {
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
			let frag_index = node.fragment_index();
			if frag_index.is_valid() {
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
				if !child_index.is_valid() {
					break;
				}
				let child_id = self.graphviz_node_id(child_index);
				let child = &self.nodes_postorder[child_index];
				let label = if child.out_is_scalar {
					String::new()
				} else {
					self.shape_to_str(&child.shape)
				};
				let extra_style = if frag_index.is_valid()
					&& !child.is_input()
					&& frag_index != child.fragment_index()
				{
					// Edge crosses fragment boundary
					", color=red, style=bold"
				} else {
					""
				};
				writeln!(w, "\t{child_id} -> {node_id} [label=\"{}\"{}];", label, extra_style)?;
			}
			for &capt_idx in &node.capture {
				let label =
					if node.out_is_scalar { String::new() } else { self.shape_to_str(&node.shape) };
				let capt = &self.tensor_vec[capt_idx].tensor_ref;
				let cap_id = self.graphviz_tensor_id(capt);
				let key = std::ptr::from_ref(capt.as_ref());
				let index = self.tensor_map[&key];
				if self.tensor_vec[index].is_input {
					writeln!(w, "\t{node_id} -> {cap_id} [label=\"{label}\", constraint=false];")?;
				} else {
					writeln!(w, "\t{node_id} -> {cap_id} [label=\"{label}\"];")?;
				}
			}
		}
		for tensor_ref in &self.tensor_vec {
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
		for scalar_ref in &self.scalar_vec {
			let id = self.graphviz_scalar_id(scalar_ref);
			let label = if let Some(name) = &scalar_ref.name {
				name.to_string()
			} else {
				format!("{:?}", std::ptr::from_ref(scalar_ref.as_ref()))
			};
			writeln!(
				w,
				"\t{id} [label=<<b>Scalar</b><br/><font color='blue'><b>{label}</b></font>>, shape=box, style=filled, fillcolor=\"#ffffc0\"];"
			)?;
		}
		writeln!(w, "}}")?;
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
