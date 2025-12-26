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
#![allow(clippy::panic_in_result_fn)]

use std::collections::{HashMap, hash_map};
use std::hint::{cold_path, unlikely};
use std::rc::Rc;

use thin_vec::{ThinVec, thin_vec};

use super::super::expr;
use super::{ExprBinaryKind, ExprInput, ExprNode, ExprUnaryKind};
use crate::define_index_type32;
use crate::new::expr::{ExprBinary, ExprCapture, ExprCast, ExprReshape, ExprUnary};
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::error::ShapeMismatchError;
use crate::tensor::{DType, TensorOpError};
use crate::util::LossyFrom;
use crate::util::bitmap::IndexBitmap;
use crate::util::index_vec::{IndexTrait, IndexVec, UntypedIndex32};

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryKind {
	Neg,
	Exp,
	Ln,
	Abs,
	Sqrt,
	Recip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionKind {
	Sum,
	Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectKind {
	Even,
	Odd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryKind {
	Add,
	Sub,
	Mul,
}

impl BinaryKind {
	pub fn is_commutative(&self) -> bool {
		match self {
			BinaryKind::Add | BinaryKind::Mul => true,
			BinaryKind::Sub => false,
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatMulKind {
	RowTimesMat,
}

pub enum NodeKind {
	Input,
	Select(SelectKind),
	View,
	Unary(UnaryKind),
	Binary(BinaryKind),
	MatMul(MatMulKind),
	Attention,
	Reduction(ReductionKind),
}

pub struct Node {
	pub node_kind: NodeKind,

	/// When `None`, the type is ambiguous. This can happen for example with Scalar inputs.
	pub dtype: Option<DType>,

	/// When `None`, the output is scalar.
	pub shape: Option<ThinVec<usize>>,

	pub can_be_batched: bool,
	pub children: [NodeIndex32; 2],

	pub parents: ThinVec<NodeIndex32>,
	pub dominator: NodeIndex32,
	pub capture: ThinVec<TensorRefIndex32>,

	pub is_dead: bool,
	pub reduction_fingerprint: u32,
	pub config_head_for: NodeIndex32,
	// `reshape_n` and `reshape_to` are used only if is_view()
	pub reshape_n: u8,
	pub reshape_to: ThinVec<usize>,

	// This is one of:
	// - ScalarRefIndex32
	// - TensorRefIndex32
	// - FragmentIndex32
	// depending on the node type.
	pub x_index: UntypedIndex32,
}

impl Default for Node {
	fn default() -> Self {
		Self {
			node_kind: NodeKind::Input,
			shape: None,
			can_be_batched: false,
			dtype: None,
			parents: ThinVec::new(),
			dominator: NodeIndex32::new_sentinel(),
			children: [NodeIndex32::new_sentinel(), NodeIndex32::new_sentinel()],
			capture: ThinVec::new(),
			is_dead: false,
			reduction_fingerprint: 0,
			config_head_for: NodeIndex32::new_sentinel(),
			reshape_n: 0,
			reshape_to: ThinVec::new(),
			x_index: UntypedIndex32::new_sentinel(),
		}
	}
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
		self.is_input() && self.shape.is_none()
	}

	pub fn is_tensor_input(&self) -> bool {
		self.is_input() && self.shape.is_some()
	}

	pub fn is_reduction(&self) -> bool {
		matches!(self.node_kind, NodeKind::Reduction(_))
	}

	pub fn is_matmul(&self) -> bool {
		matches!(self.node_kind, NodeKind::MatMul(_))
	}

	pub fn is_attention(&self) -> bool {
		matches!(self.node_kind, NodeKind::Attention)
	}

	pub fn is_config_head(&self) -> bool {
		!self.config_head_for.is_sentinel()
	}

	pub fn is_view(&self) -> bool {
		matches!(self.node_kind, NodeKind::View)
	}

	pub fn is_reshape(&self) -> bool {
		self.reshape_n != 0 || !self.reshape_to.is_empty()
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
	pub tensor_ref: Rc<expr::TensorRef>,
	pub input_node: NodeIndex32,
	pub output_node: NodeIndex32,
	pub shape: ThinVec<usize>,
	pub can_be_batched: bool,
}

impl TensorRef {
	pub fn is_input(&self) -> bool {
		!self.input_node.is_sentinel()
	}

	pub fn is_output(&self) -> bool {
		!self.output_node.is_sentinel()
	}
}

pub struct ScalarRef {
	pub scalar_ref: Rc<expr::ScalarRef>,
	pub input_node: NodeIndex32,
}

define_index_type32!(TensorRefIndex32);
type TensorVec = IndexVec<TensorRefIndex32, TensorRef>;

define_index_type32!(ScalarRefIndex32);
type ScalarVec = IndexVec<ScalarRefIndex32, ScalarRef>;

pub struct Fragment {
	pub head: NodeIndex32,
}

define_index_type32!(FragmentIndex32);
type FragmentVec = IndexVec<FragmentIndex32, Fragment>;

pub struct SumToMean {
	pub node: NodeIndex32,
	pub scalar: ScalarRefIndex32,
}

pub struct PreCompilation {
	nodes_postorder: NodeVec,
	fragments_preorder: FragmentVec,
	scalar_map: HashMap<*const expr::ScalarRef, ScalarRefIndex32>,
	scalar_vec: ScalarVec,
	tensor_map: HashMap<*const expr::TensorRef, TensorRefIndex32>,
	tensor_vec: TensorVec,
	sum_to_mean: Vec<SumToMean>,

	err_map: HashMap<NodeIndex32, usize>,
	err_vec: Vec<(NodeIndex32, Vec<String>)>,
}

struct LoadedNode {
	node: Node,
	complex: bool,
	cache_key: String,
	err: Vec<String>,
}

pub struct LoadExprState {
	visited: HashMap<*const ExprNode, NodeIndex32>,
	node_cache: HashMap<String, NodeIndex32>,
	n_complex: u32,
	bitmap: IndexBitmap<NodeIndex32>,
}

impl PreCompilation {
	pub fn new(expr: &ExprNode) -> Self {
		let mut comp = PreCompilation {
			nodes_postorder: NodeVec::with_capacity(32),
			fragments_preorder: FragmentVec::with_capacity(8),
			scalar_map: HashMap::new(),
			scalar_vec: ScalarVec::with_capacity(4),
			tensor_map: HashMap::new(),
			tensor_vec: TensorVec::with_capacity(4),
			sum_to_mean: Vec::new(),
			err_map: HashMap::new(),
			err_vec: Vec::new(),
		};
		let mut state = LoadExprState {
			visited: HashMap::new(),
			node_cache: HashMap::new(),
			n_complex: 0,
			bitmap: IndexBitmap::new(),
		};
		comp.load_expr(expr, &mut state);
		if comp.have_errors() {
			return comp;
		}
		comp.remove_dead_code();
		if comp.have_errors() {
			return comp;
		}
		comp.find_dominators();
		if comp.have_errors() {
			return comp;
		}
		comp.find_reduction_fingerprints(&mut state);
		if comp.have_errors() {
			return comp;
		}
		//comp.find_races(&mut state);
		//comp.calc_shapes()?;
		//comp.find_heads();
		//comp.find_fragments()?;
		comp
	}

	pub fn have_errors(&self) -> bool {
		!self.err_map.is_empty()
	}

	fn log_error(&mut self, node_index: NodeIndex32, message: String) {
		let idx = match self.err_map.entry(node_index) {
			hash_map::Entry::Occupied(entry) => *entry.get(),
			hash_map::Entry::Vacant(entry) => {
				let next_idx = self.err_vec.len();
				entry.insert(next_idx);
				self.err_vec.push((node_index, Vec::new()));
				next_idx
			},
		};
		self.err_vec[idx].1.push(message);
	}

	// TODO - refactor to make non recursive
	fn load_expr(&mut self, expr: &ExprNode, state: &mut LoadExprState) -> NodeIndex32 {
		let expr_key = std::ptr::from_ref(expr);
		if let Some(index) = state.visited.get(&expr_key) {
			return *index;
		}

		let t = match expr {
			ExprNode::Input(input) => {
				self.load_input(input) //
			},
			ExprNode::Capture(capture) => {
				let child_idx = self.load_expr(&capture.expr, state);
				self.load_capture(capture, child_idx)
			},
			ExprNode::Cast(cast) => {
				let child_idx = self.load_expr(&cast.expr, state);
				self.load_cast(cast, child_idx)
			},
			ExprNode::Reshape(reshape) => {
				let child_idx = self.load_expr(&reshape.expr, state);
				self.load_reshape(reshape, child_idx)
			},
			ExprNode::Unary(unary) => {
				let child_idx = self.load_expr(&unary.expr, state);
				self.load_unary(unary, child_idx)
			},
			ExprNode::Binary(binary) => {
				let a_idx = self.load_expr(&binary.lhs, state);
				let b_idx = self.load_expr(&binary.rhs, state);
				self.load_binary(binary, a_idx, b_idx)
			},
		};

		let loaded = match t {
			Ok(loaded) => loaded,
			Err(existing_node) => {
				state.visited.insert(expr_key, existing_node);
				return existing_node;
			},
		};

		let next_index = self.nodes_postorder.next_index();
		if !loaded.cache_key.is_empty() {
			match state.node_cache.entry(loaded.cache_key) {
				hash_map::Entry::Occupied(entry) => {
					let cached_index = *entry.get();
					state.visited.insert(expr_key, cached_index);
					return cached_index;
				},
				hash_map::Entry::Vacant(entry) => {
					entry.insert(next_index);
				},
			}
		}
		for err in loaded.err {
			self.log_error(next_index, err);
		}

		if loaded.complex {
			state.n_complex += 1;
		}

		for &child in &loaded.node.children {
			if !self.nodes_postorder.is_valid(child) {
				break;
			}
			self.nodes_postorder[child].parents.push(next_index);
		}

		let real_index = self.nodes_postorder.push(loaded.node);
		debug_assert!(next_index == real_index);
		state.visited.insert(expr_key, next_index);
		next_index
	}

	fn load_input(&mut self, input: &ExprInput) -> Result<LoadedNode, NodeIndex32> {
		let dtype;
		let x_index;
		let shape;
		let can_be_batched;
		match input {
			ExprInput::Tensor(tensor_ref) => {
				dtype = Some(tensor_ref.dtype);
				match self.add_tensor_input(tensor_ref.clone(), self.nodes_postorder.next_index()) {
					Ok(tensor_index) => {
						x_index = tensor_index.to_untyped();
						let t = &self.tensor_vec[tensor_index];
						shape = Some(t.shape.clone());
						can_be_batched = t.can_be_batched;
					},
					Err(existing_node) => {
						return Err(existing_node);
					},
				}
			},
			ExprInput::Scalar(scalar_ref) => {
				dtype = None;
				can_be_batched = false;
				match self.add_scalar_input(scalar_ref.clone(), self.nodes_postorder.next_index()) {
					Ok(scalar_index) => {
						x_index = scalar_index.to_untyped();
						shape = None;
					},
					Err(existing_node) => {
						return Err(existing_node);
					},
				}
			},
		}
		Ok(LoadedNode {
			node: Node {
				node_kind: NodeKind::Input,
				dtype,
				shape,
				can_be_batched,
				children: [NodeIndex32::new_sentinel(), NodeIndex32::new_sentinel()],
				x_index,
				..Default::default()
			},
			complex: false,
			cache_key: String::new(),
			err: Vec::new(),
		})
	}

	fn load_capture(
		&mut self,
		capture: &ExprCapture,
		child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		let tensor_idx =
			self.add_tensor_output(capture.tensor_ref.clone(), self.nodes_postorder.next_index());
		let child = &mut self.nodes_postorder[child_idx];
		if !child.is_input() {
			child.capture.push(tensor_idx);
			return Err(child_idx);
		}
		// Insert identity node to perform the capture.
		// This node is also a root.
		Ok(LoadedNode {
			node: Node {
				node_kind: NodeKind::View,
				dtype: child.dtype,
				shape: child.shape.clone(),
				can_be_batched: child.can_be_batched,
				children: [child_idx, NodeIndex32::new_sentinel()],
				capture: thin_vec![tensor_idx],
				..Default::default()
			},
			complex: false,
			cache_key: format!("identity:{:?}", child_idx.raw),
			err: Vec::new(),
		})
	}

	fn load_reshape(
		&self,
		reshape: &ExprReshape,
		mut child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		let child = &self.nodes_postorder[child_idx];
		if child.is_view() && child.reshape_n == 0 && child.reshape_to.is_empty() {
			child_idx = child.children[0];
		}

		let mut err = Vec::new();
		let old_shape: &[usize] = child.shape.as_ref().map_or(&[], |vec| &vec);
		let mut shape = ThinVec::from(old_shape);
		let keep = shape.len().saturating_sub(reshape.reshape_n as usize);
		let old_elems = old_shape.iter().skip(keep).product::<usize>();
		let new_elems = reshape.reshape_to.iter().product::<usize>();
		if old_elems != new_elems {
			cold_path();
			err.push(format!(
				"Reshape: element count mismatch (got {new_elems}, expected {old_elems})",
			));
		}
		shape.truncate(keep);
		shape.extend_from_slice(&reshape.reshape_to);

		Ok(LoadedNode {
			node: Node {
				node_kind: NodeKind::View,
				dtype: child.dtype,
				shape: Some(shape),
				can_be_batched: child.can_be_batched,
				children: [child_idx, NodeIndex32::new_sentinel()],
				reshape_n: reshape.reshape_n,
				reshape_to: reshape.reshape_to.clone(),
				..Default::default()
			},
			complex: false,
			cache_key: format!(
				"reshape:{:?}:{:?}:{:?}",
				reshape.reshape_n, reshape.reshape_to, child_idx.raw
			),
			err,
		})
	}

	fn load_cast(
		&self,
		cast: &ExprCast,
		mut child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		let child = &self.nodes_postorder[child_idx];
		let reshape_n;
		let reshape_to;
		if child.is_view() {
			child_idx = child.children[0];
			reshape_n = child.reshape_n;
			reshape_to = child.reshape_to.clone();
		} else {
			reshape_n = 0;
			reshape_to = ThinVec::new();
		}
		Ok(LoadedNode {
			node: Node {
				node_kind: NodeKind::View,
				dtype: Some(cast.dtype),
				shape: child.shape.clone(),
				can_be_batched: child.can_be_batched,
				children: [child_idx, NodeIndex32::new_sentinel()],
				reshape_n,
				reshape_to,
				..Default::default()
			},
			complex: false,
			cache_key: format!("cast:{:?}:{:?}", cast.dtype, child_idx.raw),
			err: Vec::new(),
		})
	}

	fn load_sum_to_mean(&mut self, child_idx: NodeIndex32) -> Result<LoadedNode, NodeIndex32> {
		let mut err = Vec::new();
		let child = &self.nodes_postorder[child_idx];
		let width = if let Some(&w) = child.shape.as_ref().and_then(|vec| vec.last()) {
			w
		} else {
			cold_path();
			err.push("SumToMean: missing reduce dimension".into());
			1
		};
		// Insert `ScalarInput` node.
		// TODO - should create `Const` node
		let scalar_ref = expr::ScalarRef::new(Some("__sum_to_mean__".into()));
		*scalar_ref.value.borrow_mut() = Some(1.0 / f64::lossy_from(width));
		let x_index = match self.add_scalar_input(scalar_ref, self.nodes_postorder.next_index()) {
			Ok(scalar_idx) => {
				self.sum_to_mean.push(SumToMean { node: child_idx, scalar: scalar_idx });
				scalar_idx.to_untyped()
			},
			Err(_) => {
				unreachable!();
			},
		};
		Ok(LoadedNode {
			node: Node {
				node_kind: NodeKind::Input,
				dtype: None,
				can_be_batched: false,
				x_index,
				..Default::default()
			},
			complex: false,
			cache_key: String::new(),
			err,
		})
	}

	fn load_unary(
		&mut self,
		unary: &ExprUnary,
		child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		match unary.kind {
			ExprUnaryKind::Neg
			| ExprUnaryKind::Exp
			| ExprUnaryKind::Ln
			| ExprUnaryKind::Abs
			| ExprUnaryKind::Sqrt
			| ExprUnaryKind::Recip => {
				let unary_kind = match unary.kind {
					ExprUnaryKind::Neg => UnaryKind::Neg,
					ExprUnaryKind::Exp => UnaryKind::Exp,
					ExprUnaryKind::Ln => UnaryKind::Ln,
					ExprUnaryKind::Abs => UnaryKind::Abs,
					ExprUnaryKind::Sqrt => UnaryKind::Sqrt,
					ExprUnaryKind::Recip => UnaryKind::Recip,
					_ => unreachable!(),
				};
				let child = &self.nodes_postorder[child_idx];
				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Unary(unary_kind),
						dtype: child.dtype,
						shape: child.shape.clone(),
						can_be_batched: child.can_be_batched,
						children: [child_idx, NodeIndex32::new_sentinel()],
						..Default::default()
					},
					complex: false,
					cache_key: format!("unary:{:?}:{:?}", unary_kind, child_idx.raw),
					err: Vec::new(),
				})
			},
			ExprUnaryKind::Sum | ExprUnaryKind::Max => {
				let reduction_kind = match unary.kind {
					ExprUnaryKind::Sum => ReductionKind::Sum,
					ExprUnaryKind::Max => ReductionKind::Max,
					_ => unreachable!(),
				};
				let mut err = Vec::new();
				let child = &self.nodes_postorder[child_idx];
				let mut shape = child.shape.clone();
				if let Some(last_dim) = shape.as_mut().and_then(|vec| vec.last_mut()) {
					*last_dim = 1;
				} else {
					cold_path();
					err.push("missing reduce dimension".into());
					shape = Some(thin_vec![1]);
				};
				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Reduction(reduction_kind),
						dtype: child.dtype,
						shape,
						can_be_batched: child.can_be_batched,
						children: [child_idx, NodeIndex32::new_sentinel()],
						..Default::default()
					},
					complex: true,
					cache_key: format!("reduction:{:?}:{:?}", reduction_kind, child_idx.raw),
					err,
				})
			},
			ExprUnaryKind::SelectEven | ExprUnaryKind::SelectOdd => {
				let select_kind = match unary.kind {
					ExprUnaryKind::SelectEven => SelectKind::Even,
					ExprUnaryKind::SelectOdd => SelectKind::Odd,
					_ => unreachable!(),
				};
				let mut err = Vec::new();
				let child = &self.nodes_postorder[child_idx];
				let mut shape = child.shape.clone();
				if let Some(last_dim) = shape.as_mut().and_then(|vec| vec.last_mut()) {
					if *last_dim < 2 || (*last_dim % 2 != 0) {
						cold_path();
						err.push("select dimension not even".into());
					}
					*last_dim /= 2;
				} else {
					cold_path();
					err.push("missing select dimension".into());
					shape = Some(thin_vec![0]);
				};
				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Select(select_kind),
						dtype: child.dtype,
						shape,
						can_be_batched: child.can_be_batched,
						children: [child_idx, NodeIndex32::new_sentinel()],
						..Default::default()
					},
					complex: false,
					cache_key: format!("select:{:?}:{:?}", select_kind, child_idx.raw),
					err,
				})
			},
			ExprUnaryKind::SumToMean => self.load_sum_to_mean(child_idx),
		}
	}

	fn common_dtype(
		a_dtype: Option<DType>,
		b_dtype: Option<DType>,
		err: &mut Vec<String>,
	) -> Option<DType> {
		match (a_dtype, b_dtype) {
			(None, None) => None,
			(Some(a_dt), None) => Some(a_dt),
			(None, Some(b_dt)) => Some(b_dt),
			(Some(a_dt), Some(b_dt)) => {
				if a_dt == b_dt {
					Some(a_dt)
				} else {
					cold_path();
					err.push(format!("dtype mismatch: {} vs {}", a_dt, b_dt));
					Some(common_dtype(a_dt, b_dt))
				}
			},
		}
	}

	fn common_shape(mut a: &[usize], mut b: &[usize], err: &mut Vec<String>) -> ThinVec<usize> {
		if a.len() != b.len() {
			cold_path();
			err.push(format!("shape length mismatch: {:?} vs {:?}", a, b));
			let len = a.len().min(b.len()); // TODO - shoud use larger of the 2, not the smaller
			let skip_a = a.len() - len;
			let skip_b = b.len() - len;
			a = &a[skip_a..];
			b = &b[skip_b..];
		}
		let mut shape = ThinVec::with_capacity(a.len());
		for (a_dim, b_dim) in a.iter().zip(b.iter()) {
			if *a_dim == *b_dim {
				shape.push(*a_dim);
			} else if *a_dim == 1 {
				shape.push(*b_dim);
			} else if *b_dim == 1 {
				shape.push(*a_dim);
			} else {
				cold_path();
				err.push(format!("shape dimension mismatch: {:?} vs {:?}", a_dim, b_dim));
			}
		}
		shape
	}

	fn common_shape_opt(
		a: Option<&[usize]>,
		b: Option<&[usize]>,
		err: &mut Vec<String>,
	) -> Option<ThinVec<usize>> {
		match (a, b) {
			(None, None) => None,
			(None, Some(sh)) => Some(ThinVec::from(sh)),
			(Some(sh), None) => Some(ThinVec::from(sh)),
			(Some(a), Some(b)) => Some(Self::common_shape(a, b, err)),
		}
	}

	fn load_binary(
		&self,
		binary: &ExprBinary,
		a_idx: NodeIndex32,
		b_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		match binary.kind {
			ExprBinaryKind::Add | ExprBinaryKind::Sub | ExprBinaryKind::Mul => {
				let (binary_kind, commutative) = match binary.kind {
					ExprBinaryKind::Add => (BinaryKind::Add, true),
					ExprBinaryKind::Sub => (BinaryKind::Sub, false),
					ExprBinaryKind::Mul => (BinaryKind::Mul, true),
					_ => unreachable!(),
				};
				let (a_idx, b_idx) =
					if commutative && a_idx > b_idx { (b_idx, a_idx) } else { (a_idx, b_idx) };
				let a = &self.nodes_postorder[a_idx];
				let b = &self.nodes_postorder[b_idx];
				let mut err = Vec::new();
				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Binary(binary_kind),
						dtype: Self::common_dtype(a.dtype, b.dtype, &mut err),
						shape: Self::common_shape_opt(
							a.shape.as_ref().map(|sh| &sh[..]),
							b.shape.as_ref().map(|sh| &sh[..]),
							&mut err,
						),
						can_be_batched: a.can_be_batched || b.can_be_batched,
						children: [a_idx, b_idx],
						..Default::default()
					},
					complex: false,
					cache_key: format!("binary:{:?}:{:?}:{:?}", binary_kind, a_idx.raw, b_idx.raw),
					err,
				})
			},
			ExprBinaryKind::First => Err(a_idx),
			ExprBinaryKind::RowTimesMat => {
				let a = &self.nodes_postorder[a_idx];
				let b = &self.nodes_postorder[b_idx];
				let matmul_kind = MatMulKind::RowTimesMat;

				let mut err = Vec::new();
				let a_shape: &[usize] = a.shape.as_ref().map_or(&[], |vec| &vec[..]);
				let b_shape: &[usize] = b.shape.as_ref().map_or(&[], |vec| &vec[..]);
				if a_shape.len() < 1 || b_shape.len() < 2 {
					cold_path();
					err.push(format!("matmul: not enough dimensions"));
				}
				let (a_rest, [a_len]) = Self::split_shape::<1>(a_shape);
				let (b_rest, [b_row, b_col]) = Self::split_shape::<2>(b_shape);
				if a_len != b_row {
					cold_path();
					err.push(format!("matmul: shape mismatch"));
				}
				let mut shape = ThinVec::new();
				if Self::broadcast_shapes(&mut shape, a_rest, b_rest).is_err() {
					cold_path();
					err.push(format!("matmul: batch shape mismatch"));
				}
				shape.push(b_col);

				if b.can_be_batched {
					cold_path();
					err.push("row times mat: mat cannot be batched".into());
				}

				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::MatMul(matmul_kind),
						dtype: Self::common_dtype(a.dtype, b.dtype, &mut err),
						shape: Some(shape),
						can_be_batched: a.can_be_batched,
						children: [a_idx, b_idx],
						..Default::default()
					},
					complex: true,
					cache_key: format!("matmul:{:?}:{:?}:{:?}", matmul_kind, a_idx.raw, b_idx.raw),
					err,
				})
			},
			ExprBinaryKind::Attention => {
				let a = &self.nodes_postorder[a_idx];
				let b = &self.nodes_postorder[b_idx];

				let mut err = Vec::new();
				let a_shape: &[usize] = a.shape.as_ref().map_or(&[], |vec| &vec[..]);
				let b_shape: &[usize] = b.shape.as_ref().map_or(&[], |vec| &vec[..]);
				if a_shape.len() < 3 || b_shape.len() < 3 {
					cold_path();
					err.push(format!("attention: not enough dimensions"));
				}
				let (q_rest, [q1, q2, q3]) = Self::split_shape::<3>(a_shape);
				let (kv_rest, [kv1, kv2, kv3]) = Self::split_shape::<3>(b_shape);
				if kv1 != 1 || q2 != kv2 || q3 >= kv3 {
					cold_path();
					err.push(format!("attention: shape mismatch"));
				}
				let mut shape = ThinVec::new();
				if Self::broadcast_shapes(&mut shape, q_rest, kv_rest).is_err() {
					cold_path();
					err.push(format!("attention: batch shape mismatch"));
				}
				shape.push(q1);
				shape.push(q2);
				shape.push(kv3.saturating_sub(q3));

				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Attention,
						dtype: Self::common_dtype(a.dtype, b.dtype, &mut err),
						shape: Some(shape),
						can_be_batched: a.can_be_batched || b.can_be_batched,
						children: [a_idx, b_idx],
						..Default::default()
					},
					complex: true,
					cache_key: format!("attention:{:?}:{:?}", a_idx.raw, b_idx.raw),
					err,
				})
			},
		}
	}

	fn add_scalar_input(
		&mut self,
		scalar_ref: Rc<expr::ScalarRef>,
		node: NodeIndex32,
	) -> Result<ScalarRefIndex32, NodeIndex32> {
		let key = std::ptr::from_ref(scalar_ref.as_ref());
		match self.scalar_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index = self.scalar_vec.push(ScalarRef { scalar_ref, input_node: node });
				entry.insert(index);
				Ok(index)
			},
			hash_map::Entry::Occupied(entry) => {
				let index = *entry.get();
				Err(self.scalar_vec[index].input_node)
			},
		}
	}

	fn add_tensor_input(
		&mut self,
		tensor_ref: Rc<expr::TensorRef>,
		node: NodeIndex32,
	) -> Result<TensorRefIndex32, NodeIndex32> {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		match self.tensor_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let shape = ThinVec::from(&tensor_ref.shape[..]);
				let can_be_batched = tensor_ref.batched != expr::CanBeBatched::No;
				let index = self.tensor_vec.push(TensorRef {
					tensor_ref,
					input_node: node,
					output_node: NodeIndex32::new_sentinel(),
					shape,
					can_be_batched,
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
		tensor_ref: Rc<expr::TensorRef>,
		node: NodeIndex32,
	) -> TensorRefIndex32 {
		let key = std::ptr::from_ref(tensor_ref.as_ref());
		match self.tensor_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let shape = ThinVec::from(&tensor_ref.shape[..]);
				let can_be_batched = tensor_ref.batched != expr::CanBeBatched::No;
				let index = self.tensor_vec.push(TensorRef {
					tensor_ref,
					input_node: NodeIndex32::new_sentinel(),
					output_node: node,
					shape,
					can_be_batched,
				});
				entry.insert(index);
				index
			},
			hash_map::Entry::Occupied(entry) => *entry.get(),
		}
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

	fn find_reduction_fingerprints(&mut self, state: &mut LoadExprState) {
		let bitmap = &mut state.bitmap;
		bitmap.clear_and_resize(&self.nodes_postorder, state.n_complex as usize);
		let mut n_config_driving = 0;
		let mut n_dead_config_driving = 0;
		for idx in self.nodes_postorder.indexes() {
			let me = &mut self.nodes_postorder[idx];
			if unlikely(me.is_dead) {
				if me.is_reduction() || me.is_matmul() || me.is_attention() {
					n_dead_config_driving += 1;
				}
				continue;
			}
			if me.is_binary() {
				let left_child = me.children[0];
				let right_child = me.children[1];
				bitmap.union(idx, left_child, right_child);
				if me.is_matmul() || me.is_attention() {
					bitmap.set_bit(idx, n_config_driving);
					n_config_driving += 1;
				}
			} else if me.is_unary() {
				let child = me.children[0];
				bitmap.copy_row(idx, child);
				if me.is_reduction() {
					bitmap.set_bit(idx, n_config_driving);
					n_config_driving += 1;
				}
			}
		}
		debug_assert_eq!(n_config_driving + n_dead_config_driving, (state.n_complex as usize));
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

	/*
	#[allow(clippy::manual_assert)]
	fn find_races(&self, state: &mut LoadExprState) {
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
	}*/

	fn split_shape<const N: usize>(shape: &[usize]) -> (&[usize], [usize; N]) {
		let len = shape.len();
		let cnt = len.min(N);
		let rest = len - cnt;
		let mut a = [1; N];
		for i in 0..N {
			if i < cnt {
				a[N - 1 - i] = shape[len - 1 - i];
			}
		}
		(&shape[..rest], a)
	}

	fn broadcast_shapes(
		result: &mut ThinVec<usize>,
		a: &[usize],
		b: &[usize],
	) -> Result<(), ShapeMismatchError> {
		let mut err = Ok(());
		let len = a.len().max(b.len());
		let skip_a = len - a.len();
		let skip_b = len - b.len();
		for d in 0..len {
			let dim_a = if d < skip_a { 1 } else { a[d - skip_a] };
			let dim_b = if d < skip_b { 1 } else { b[d - skip_b] };
			let dim = if dim_a == dim_b || dim_b == 1 {
				dim_a
			} else if dim_a == 1 {
				dim_b
			} else {
				cold_path();
				err = Err(ShapeMismatchError);
				dim_a.max(dim_b)
			};
			result.push(dim);
		}
		err
	}

	/*
	#[allow(clippy::too_many_lines)]
	fn calc_shapes(&mut self) -> Result<(), TensorOpError> {
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
				match me.node_kind {
					NodeKind::Reduction(_) => {
						if let Some(last) = me.shape.last_mut() {
							*last = 1;
						} else {
							me.shape.push(1);
						}
					},
					NodeKind::Select(_) => {
						if let Some(last) = me.shape.last_mut() {
							*last /= 2;
						} else {
							me.shape.push(0);
						}
					},
					NodeKind::View if me.is_reshape() => {
						let keep = me.shape.len().saturating_sub(me.reshape_n as usize);
						let elems = me.shape.iter().skip(keep).product::<usize>();
						let new_elems = me.reshape_to.iter().product::<usize>();
						if elems != new_elems {
							cold_path();
							return Err(TensorOpError::InvalidReshape);
						}
						me.shape.truncate(keep);
						me.shape.extend_from_slice(&me.reshape_to);
					},
					_ => continue,
				}
			} else {
				debug_assert!(me.is_binary());

				// For binary operations, use broadcast to get output shape
				let a = me.children[0];
				let b = me.children[1];
				let a_shape: &[usize] = &all_children[a].shape;
				let b_shape: &[usize] = &all_children[b].shape;
				#[allow(clippy::single_match_else)]
				#[allow(clippy::len_zero)]
				match me.node_kind {
					NodeKind::MatMul(MatMulKind::RowTimesMat) => {
						let (a_rest, [a_len]) = Self::split_shape::<1>(a_shape);
						let (b_rest, [b_row, b_col]) = Self::split_shape::<2>(b_shape);
						if a_len != b_row {
							cold_path();
							return Err(TensorOpError::ShapeMismatch);
						}
						Self::broadcast_shapes(&mut me.shape, a_rest, b_rest)?;
						me.shape.push(b_col);
					},
					NodeKind::Attention => {
						let (q_rest, [q1, q2, q3]) = Self::split_shape::<3>(a_shape);
						let (kv_rest, [kv1, kv2, kv3]) = Self::split_shape::<3>(b_shape);
						if kv1 != 1 || q2 != kv2 || q3 >= kv3 {
							cold_path();
							return Err(TensorOpError::ShapeMismatch);
						}
						Self::broadcast_shapes(&mut me.shape, q_rest, kv_rest)?;
						me.shape.push(q1);
						me.shape.push(q2);
						me.shape.push(kv3 - q3);
					},
					_ => {
						Self::broadcast_shapes(&mut me.shape, a_shape, b_shape)?;
					},
				}
			}

			// If we store back into inputs, make sure captures have correct shape and dtype
			let my_shape = me.shape.as_slice();
			for &idx in &me.capture {
				let tensor = &self.tensor_vec[idx];
				if let Some(my_dtype) = me.dtype
					&& my_dtype != tensor.tensor_ref.dtype
				{
					cold_path();
					return Err(TensorOpError::DTypeMismatch);
				}
				if !tensor.input_node.is_sentinel() {
					let tensor_shape = tensor.shape.as_slice();
					if tensor_shape != my_shape {
						cold_path();
						return Err(TensorOpError::ShapeMismatch);
					}
				}
			}
		}

		Ok(())
	}
	*/

	fn find_heads(&mut self) {
		for idx in self.nodes_postorder.indexes() {
			if !self.nodes_postorder[idx].is_reduction()
				&& !self.nodes_postorder[idx].is_matmul()
				&& !self.nodes_postorder[idx].is_attention()
			{
				continue;
			}

			let start = &self.nodes_postorder[idx];
			let start_shape: Option<&[usize]> = start.shape.as_ref().map(|sh| &sh[..]);
			let start_fingerprint = start.reduction_fingerprint;

			let mut head_idx = idx;
			let mut head = start;
			loop {
				let parent_idx = head.dominator;
				if !self.nodes_postorder.is_valid(parent_idx) {
					break;
				}
				let parent = &self.nodes_postorder[parent_idx];
				let parent_shape: Option<&[usize]> = parent.shape.as_ref().map(|sh| &sh[..]);
				let parent_fingerprint = parent.reduction_fingerprint;
				if (!parent.is_view() && parent_shape != start_shape)
					|| parent_fingerprint != start_fingerprint
				{
					break;
				}
				head_idx = parent_idx;
				head = parent;
			}

			self.nodes_postorder[head_idx].config_head_for = idx;
		}
	}

	fn find_fragments(&mut self) -> Result<(), TensorOpError> {
		self.fragments_preorder.raw.clear();
		for idx in self.nodes_postorder.indexes().rev() {
			let (_, item, all_parents) = self.nodes_postorder.borrow_multiple(idx);
			if item.is_input() || unlikely(item.is_dead) {
				continue;
			}
			if let Some((&first_parent, other_parents)) = item.parents.split_first()
				&& !all_parents[first_parent].is_matmul()
				&& let parent_frag = all_parents[first_parent].fragment_index()
				&& !item.is_config_head()
				&& !item.is_reshape()
				&& other_parents.iter().all(|&p| {
					!all_parents[p].is_matmul() && all_parents[p].fragment_index() == parent_frag
				}) {
				item.set_fragment_index(parent_frag);
			} else {
				let new_frag = self.fragments_preorder.push(Fragment { head: idx });
				item.set_fragment_index(new_frag);
			}
		}
		Ok(())
	}

	pub fn graphviz_node_label(&self, node: &Node) -> String {
		match node.node_kind {
			NodeKind::Input => {
				if node.is_tensor_input() {
					let tensor_ref = &self.tensor_vec[node.tensor_index()].tensor_ref;
					let name = tensor_ref.name.as_deref().unwrap_or("<unnamed>");
					format!("<b>Tensor</b><br/><font color='blue'><b>{name}</b></font>")
				} else {
					let scalar_ref = &self.scalar_vec[node.scalar_index()].scalar_ref;
					let name = scalar_ref.name.as_deref().unwrap_or("<unnamed>");
					format!("<b>Scalar</b><br/><font color='blue'><b>{name}</b></font>")
				}
			},
			NodeKind::View => {
				let child = &self.nodes_postorder[node.children[0]];
				let mut result = format!("<b>View</b>");
				let mut from = String::new();
				let mut to = String::new();
				if child.dtype != node.dtype {
					from = self.dtype_to_str(child.dtype);
					to = self.dtype_to_str(node.dtype);
				}
				if node.is_reshape() {
					from =
						format!("{from}&#91;<font color='blue'>-{}</font>..&#93;", node.reshape_n);
					to = format!("{to}{}", self.shape_to_str(&node.reshape_to));
				}
				result = format!("{from} -&gt; {to}");
				result
			},
			NodeKind::Select(select) => match select {
				SelectKind::Even => "<b>Select Even</b>".to_string(),
				SelectKind::Odd => "<b>Select Odd</b>".to_string(),
			},
			NodeKind::Unary(unary) => match unary {
				UnaryKind::Neg => "<b>Neg</b>".to_string(),
				UnaryKind::Exp => "<b>Exp</b>".to_string(),
				UnaryKind::Ln => "<b>Ln</b>".to_string(),
				UnaryKind::Abs => "<b>Abs</b>".to_string(),
				UnaryKind::Sqrt => "<b>Sqrt</b>".to_string(),
				UnaryKind::Recip => "<b>Recip</b>".to_string(),
			},
			NodeKind::Binary(binary) => match binary {
				BinaryKind::Add => "<b>Add</b>".to_string(),
				BinaryKind::Sub => "<b>Sub</b>".to_string(),
				BinaryKind::Mul => "<b>Mul</b>".to_string(),
			},
			NodeKind::MatMul(matmul) => match matmul {
				MatMulKind::RowTimesMat => "<b>row * MAT</b>".to_string(),
			},
			NodeKind::Attention => "<b>ATTN</b>".to_string(),
			NodeKind::Reduction(reduction) => match reduction {
				ReductionKind::Sum => "<b>Sum</b>".to_string(),
				ReductionKind::Max => "<b>Max</b>".to_string(),
			},
		}
	}

	fn dtype_to_str(&self, dtype: Option<DType>) -> String {
		let dtype = if let Some(dt) = dtype { dt.to_string() } else { "T".to_string() };
		format!("<font color='teal'>{dtype}</font>")
	}

	fn shape_to_str(&self, shape: &[usize]) -> String {
		let mut result = format!("&#91;");
		for (i, &dim) in shape.iter().enumerate() {
			if i > 0 {
				result.push(',');
			}
			result.push_str("<font color='blue'>");
			result.push_str(&dim.to_string());
			result.push_str("</font>");
		}
		result.push_str("&#93;");
		result
	}

	pub fn sanitize_for_graphviz_html(s: &str) -> String {
		let mut result = String::with_capacity(s.len() * 2); // Pre-allocate

		for c in s.chars() {
			match c {
				'<' => result.push_str("&lt;"),
				'>' => result.push_str("&gt;"),
				'&' => result.push_str("&amp;"),
				'"' => result.push_str("&quot;"),
				'\n' => result.push_str("<BR/>"),
				'[' => result.push_str("&#91;"),
				']' => result.push_str("&#93;"),
				c => result.push(c),
			}
		}

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
			let label =
				self.graphviz_node_label(node) + &Self::sanitize_for_graphviz_html(&extra_label);
			writeln!(w, "\t{node_id} [label=<{label}>];")?;
			if node.is_input() {
				if node.is_tensor_input() {
					writeln!(w, "\t{node_id} [shape=box, style=filled, fillcolor=\"#cceecc\"];")?;
				} else {
					writeln!(w, "\t{node_id} [shape=box, style=filled, fillcolor=\"#ffffc0\"];")?;
				}
			} else if node.is_config_head() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccccff\"];")?;
			} else if node.is_fork() {
				if unlikely(node.is_dead) {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#cccccc\"];")?;
				} else {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffcccc\"];")?;
				}
			} else if node.is_reduction() || node.is_matmul() || node.is_attention() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffccff\"];")?;
			} else if node.is_captured() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccffcc\"];")?;
			}
			if !node.dominator.is_sentinel() {
				let dom_id = format!("node_{}", node.dominator.raw);
				writeln!(
					w,
					"\t{node_id} -> {dom_id} [label=< >, style=dashed, color=\"#808080\", constraint=true];"
				)?;
			}
			if node.is_input() {
				//writeln!(w, "\t{{ rank = min; {node_id} }}")?;
			} else {
				let frag_index = node.fragment_index();
				if self.fragments_preorder.is_valid(frag_index) {
					let frag_head = self.fragments_preorder[frag_index].head;
					let frag_node = &self.nodes_postorder[frag_head];
					let fragment_kind = if frag_node.is_config_head() {
						let frag_node = &self.nodes_postorder[frag_node.config_head_for];
						if frag_node.is_reduction() {
							"Reduction"
						} else if frag_node.is_matmul() {
							"MatMul"
						} else if frag_node.is_attention() {
							"Attention"
						} else {
							"<config head>"
						}
					} else {
						"Element-wise"
					};
					writeln!(
						w,
						"\tsubgraph cluster_{} {{ label=\"{fragment_kind}\" labelloc=\"b\" labeljust=\"l\" {node_id} }}",
						frag_head.raw
					)?;
				}
				let ordered = match node.node_kind {
					NodeKind::Binary(bin) => !bin.is_commutative(),
					NodeKind::MatMul(_) | NodeKind::Attention => true,
					_ => false,
				};
				for &child_index in &node.children {
					if !self.nodes_postorder.is_valid(child_index) {
						break;
					}
					let child_id = format!("node_{}", child_index.raw);
					let child = &self.nodes_postorder[child_index];
					let label = self.dtype_to_str(child.dtype);
					let label = if let Some(shape) = &child.shape {
						format!("{label}{}", self.shape_to_str(shape))
					} else {
						label
					};
					let extra_style = if ordered {
						if child_index == node.children[0] {
							", color=\"#0000aa\""
						} else if child_index == node.children[1] {
							", color=\"#aa0000\""
						} else {
							", color=\"#00aa00\""
						}
					} else {
						", color=\"#000000\""
					};
					let extra_style = if self.fragments_preorder.is_valid(frag_index)
						&& (child.is_input() || frag_index != child.fragment_index())
					{
						format!("{extra_style}, style=bold")
					} else {
						extra_style.to_string()
					};
					writeln!(
						w,
						"\t{child_id} -> {node_id} [label=<{label}>{extra_style}, constraint=true];",
					)?;
				}
			}
			for &capt_idx in &node.capture {
				let label = self.dtype_to_str(node.dtype);
				let label = if let Some(shape) = &node.shape {
					format!("{label}{}", self.shape_to_str(shape))
				} else {
					label
				};
				let input_node = self.tensor_vec[capt_idx].input_node;
				if input_node.is_sentinel() {
					let cap_id = format!("ten_{}", capt_idx.raw);
					writeln!(
						w,
						"\t{node_id} -> {cap_id} [label=<{label}>, style=bold, constraint=true];"
					)?;
					let tensor_ref = &self.tensor_vec[capt_idx].tensor_ref;
					let name = tensor_ref.name.as_deref().unwrap_or("<unnamed>");
					let name = Self::sanitize_for_graphviz_html(name);
					writeln!(
						w,
						"{cap_id} [label=<<b>Tensor</b><br/><font color='blue'><b>{name}</b></font>>, shape=box, style=filled, fillcolor=\"#cceeff\"];"
					)?;
				} else {
					let cap_id = format!("node_{}", input_node.raw);
					writeln!(w, "\t{node_id} -> {cap_id} [label=<{label}>, constraint=true];")?;
				}
			}
		}
		for sum_to_mean in &self.sum_to_mean {
			let node_id = format!("node_{}", sum_to_mean.node.raw);
			let scalar_id = format!("node_{}", self.scalar_vec[sum_to_mean.scalar].input_node.raw);
			writeln!(
				w,
				"\t{node_id} -> {scalar_id} [label=< >, style=dotted, color=\"#008000\", constraint=true];"
			)?;
		}
		for (node_idx, msgs) in &self.err_vec {
			let idx = node_idx.raw;
			writeln!(
				w,
				"\terr_{idx} [label=<<b><font color='yellow'>{idx}</font></b>>, shape=box, style=filled, fillcolor=\"#ff0000\"];"
			)?;
			writeln!(
				w,
				"\tnode_{idx} -> err_{idx} [style=bold, color=\"#ff0000\", constraint=true];",
			)?;
			for msg in msgs {
				eprintln!("Node {idx}: error: {msg}");
			}
		}
		writeln!(w, "}}")?;
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
