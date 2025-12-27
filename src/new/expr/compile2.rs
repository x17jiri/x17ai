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

use std::borrow::Cow;
use std::collections::{HashMap, hash_map};
use std::hint::{cold_path, unlikely};
use std::rc::Rc;

use ordered_float::OrderedFloat;
use thin_vec::{ThinVec, thin_vec};

use super::super::expr;
use super::{ExprBinaryKind, ExprInput, ExprNode, ExprUnaryKind};
use crate::define_index_type32;
use crate::new::expr::{ExprBinary, ExprCapture, ExprCast, ExprReshape, ExprUnary};
use crate::tensor::DType;
use crate::tensor::device::dtype::common_dtype;
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
	Const,
	Input,
	Select(SelectKind),
	Cast,
	Reshape,
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
	pub children_broadcast: [bool; 2],

	pub parents: ThinVec<NodeIndex32>,
	pub dominator: NodeIndex32,
	pub capture: ThinVec<TensorRefIndex32>,

	pub is_dead: bool,
	pub reduction_fingerprint: u32,
	pub head_for: NodeIndex32,

	// This is one of:
	// - ConstRefIndex32
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
			children_broadcast: [false, false],
			capture: ThinVec::new(),
			is_dead: false,
			reduction_fingerprint: 0,
			head_for: NodeIndex32::new_sentinel(),
			x_index: UntypedIndex32::new_sentinel(),
		}
	}
}

impl Node {
	pub fn is_input(&self) -> bool {
		self.is_nullary()
	}

	pub fn const_index(&self) -> ConstRefIndex32 {
		debug_assert!(self.is_const());
		ConstRefIndex32::from(self.x_index)
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

	pub fn is_const(&self) -> bool {
		matches!(self.node_kind, NodeKind::Const)
	}

	pub fn is_scalar_input(&self) -> bool {
		self.is_input() && self.shape.is_none() && !self.is_const()
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

	pub fn is_complex(&self) -> bool {
		self.is_reduction() || self.is_matmul() || self.is_attention()
	}

	pub fn is_head(&self) -> bool {
		!self.head_for.is_sentinel()
	}

	pub fn is_cast(&self) -> bool {
		matches!(self.node_kind, NodeKind::Cast)
	}

	pub fn is_reshape(&self) -> bool {
		matches!(self.node_kind, NodeKind::Reshape)
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

pub struct ConstRef {
	pub name: Cow<'static, str>,
	pub value: f64,
	pub input_node: NodeIndex32,
}

define_index_type32!(TensorRefIndex32);
type TensorVec = IndexVec<TensorRefIndex32, TensorRef>;

define_index_type32!(ScalarRefIndex32);
type ScalarVec = IndexVec<ScalarRefIndex32, ScalarRef>;

define_index_type32!(ConstRefIndex32);
type ConstVec = IndexVec<ConstRefIndex32, ConstRef>;

pub struct Fragment {
	pub head: NodeIndex32,
}

define_index_type32!(FragmentIndex32);
type FragmentVec = IndexVec<FragmentIndex32, Fragment>;

pub struct SumToMean {
	pub node: NodeIndex32,
	pub const_idx: ConstRefIndex32,
}

pub struct PreCompilation {
	nodes_postorder: NodeVec,
	fragments_preorder: FragmentVec,
	const_map: HashMap<OrderedFloat<f64>, ConstRefIndex32>,
	const_vec: ConstVec,
	scalar_map: HashMap<*const expr::ScalarRef, ScalarRefIndex32>,
	scalar_vec: ScalarVec,
	tensor_map: HashMap<*const expr::TensorRef, TensorRefIndex32>,
	tensor_vec: TensorVec,
	sum_to_mean: Vec<SumToMean>,

	err_log: ErrorLog,
}

struct LoadedNode {
	node: Node,
	cache_key: String,
	err: Vec<String>,
}

pub struct LoadExprState {
	visited: HashMap<*const ExprNode, NodeIndex32>,
	node_cache: HashMap<String, NodeIndex32>,
	n_complex: u32,
	bitmap: IndexBitmap<NodeIndex32>,
}

pub struct ErrorLog {
	err_map: HashMap<NodeIndex32, usize>,
	err_vec: Vec<(NodeIndex32, Vec<String>)>,
}

impl ErrorLog {
	pub fn new() -> Self {
		Self {
			err_map: HashMap::new(),
			err_vec: Vec::new(),
		}
	}

	pub fn is_empty(&self) -> bool {
		self.err_vec.is_empty()
	}

	pub fn check_errors(&self) -> Result<(), ()> {
		if self.is_empty() { Ok(()) } else { Err(()) }
	}

	fn log_error(&mut self, node_index: NodeIndex32, message: String) {
		let Self { err_map, err_vec } = self;
		let idx = match err_map.entry(node_index) {
			hash_map::Entry::Occupied(entry) => *entry.get(),
			hash_map::Entry::Vacant(entry) => {
				let next_idx = err_vec.len();
				entry.insert(next_idx);
				err_vec.push((node_index, Vec::new()));
				next_idx
			},
		};
		err_vec[idx].1.push(message);
	}
}

impl PreCompilation {
	pub fn new(expr: &ExprNode) -> Self {
		let mut comp = PreCompilation {
			nodes_postorder: NodeVec::with_capacity(32),
			fragments_preorder: FragmentVec::with_capacity(8),
			const_map: HashMap::new(),
			const_vec: ConstVec::with_capacity(4),
			scalar_map: HashMap::new(),
			scalar_vec: ScalarVec::with_capacity(4),
			tensor_map: HashMap::new(),
			tensor_vec: TensorVec::with_capacity(4),
			sum_to_mean: Vec::new(),
			err_log: ErrorLog::new(),
		};
		let _ = comp.analyze(expr);
		comp
	}

	pub fn analyze(&mut self, expr: &ExprNode) -> Result<(), ()> {
		let mut state = LoadExprState {
			visited: HashMap::new(),
			node_cache: HashMap::new(),
			n_complex: 0,
			bitmap: IndexBitmap::new(),
		};

		self.load_expr(expr, &mut state);
		self.remove_dead_code();
		self.find_dominators();
		self.find_complex_op_fingerprints(&mut state);
		self.find_races(&mut state);
		self.find_heads();
		self.find_fragments();

		self.err_log.check_errors()
	}

	// TODO - refactor to make non recursive
	fn load_expr(&mut self, expr: &ExprNode, state: &mut LoadExprState) -> NodeIndex32 {
		let expr_key = std::ptr::from_ref(expr);
		if let Some(index) = state.visited.get(&expr_key) {
			return *index;
		}

		let t = match expr {
			ExprNode::Const(cst) => {
				match self.add_const(cst.name.clone(), cst.value, self.nodes_postorder.next_index())
				{
					Ok(idx) => Ok(LoadedNode {
						node: Node {
							node_kind: NodeKind::Const,
							x_index: idx.to_untyped(),
							..Default::default()
						},
						cache_key: String::new(),
						err: Vec::new(),
					}),
					Err(existing_node) => Err(existing_node),
				}
			},
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
			self.err_log.log_error(next_index, err);
		}

		if loaded.node.is_complex() {
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
				node_kind: NodeKind::Cast,
				dtype: child.dtype,
				shape: child.shape.clone(),
				can_be_batched: child.can_be_batched,
				children: [child_idx, NodeIndex32::new_sentinel()],
				capture: thin_vec![tensor_idx],
				..Default::default()
			},
			cache_key: format!("identity:{:?}", child_idx.raw),
			err: Vec::new(),
		})
	}

	fn load_reshape(
		&self,
		reshape: &ExprReshape,
		child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		let child = &self.nodes_postorder[child_idx];
		let mut err = Vec::new();
		let old_shape: &[usize] = child.shape.as_ref().map_or(&[], |vec| vec);
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

		if shape == old_shape {
			return Err(child_idx);
		}

		Ok(LoadedNode {
			node: Node {
				node_kind: NodeKind::Reshape,
				dtype: child.dtype,
				shape: Some(shape),
				can_be_batched: child.can_be_batched,
				children: [child_idx, NodeIndex32::new_sentinel()],
				..Default::default()
			},
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
		child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		let child = &self.nodes_postorder[child_idx];
		if child.dtype == Some(cast.dtype) {
			return Err(child_idx);
		}
		Ok(LoadedNode {
			node: Node {
				node_kind: NodeKind::Cast,
				dtype: Some(cast.dtype),
				shape: child.shape.clone(),
				can_be_batched: child.can_be_batched,
				children: [child_idx, NodeIndex32::new_sentinel()],
				..Default::default()
			},
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
			err.push(format!("SumToMean: missing reduce dimension"));
			1
		};
		// Insert `Const` node.
		let c = 1.0 / f64::lossy_from(width);
		let x_index = match self.add_const(
			format!("1.0 / {width}").into(),
			c,
			self.nodes_postorder.next_index(),
		) {
			Ok(idx) => {
				self.sum_to_mean.push(SumToMean { node: child_idx, const_idx: idx });
				idx
			},
			Err(_) => {
				unreachable!();
			},
		};
		Ok(LoadedNode {
			node: Node {
				node_kind: NodeKind::Const,
				x_index: x_index.to_untyped(),
				..Default::default()
			},
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
					err.push(format!("missing reduce dimension"));
					shape = Some(thin_vec![1]);
				}
				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Reduction(reduction_kind),
						dtype: child.dtype,
						shape,
						can_be_batched: child.can_be_batched,
						children: [child_idx, NodeIndex32::new_sentinel()],
						..Default::default()
					},
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
						err.push(format!("select dimension not even"));
					}
					*last_dim /= 2;
				} else {
					cold_path();
					err.push(format!("missing select dimension"));
					shape = Some(thin_vec![0]);
				}
				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Select(select_kind),
						dtype: child.dtype,
						shape,
						can_be_batched: child.can_be_batched,
						children: [child_idx, NodeIndex32::new_sentinel()],
						..Default::default()
					},
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
				let dtype = Self::common_dtype(a.dtype, b.dtype, &mut err);
				let (shape, is_broadcasted) = Self::common_shape(
					a.shape.as_ref().map(|sh| &sh[..]),
					b.shape.as_ref().map(|sh| &sh[..]),
					&mut err,
				);
				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Binary(binary_kind),
						dtype,
						shape,
						can_be_batched: a.can_be_batched || b.can_be_batched,
						children: [a_idx, b_idx],
						children_broadcast: is_broadcasted,
						..Default::default()
					},
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
				let (_a_rest, [a_len]) = Self::split_shape::<1>(a_shape);
				let (b_rest, [b_row, b_col]) = Self::split_shape::<2>(b_shape);
				if a_len != b_row {
					cold_path();
					err.push(format!("matmul: shape mismatch"));
				}
				let mut shape = ThinVec::from(a_shape);
				*shape.last_mut().unwrap() = b_col;

				if !b_rest.is_empty() || b.can_be_batched {
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
				let (mut shape, is_broadcasted) = Self::broadcast_shapes(q_rest, kv_rest, &mut err);
				shape.push(q1);
				shape.push(q2);
				shape.push(kv3.saturating_sub(q3));
				if is_broadcasted[0] || is_broadcasted[1] {
					cold_path();
					err.push(format!("attention inputs cannot be broadcasted"));
				}

				Ok(LoadedNode {
					node: Node {
						node_kind: NodeKind::Attention,
						dtype: Self::common_dtype(a.dtype, b.dtype, &mut err),
						shape: Some(shape),
						can_be_batched: a.can_be_batched || b.can_be_batched,
						children: [a_idx, b_idx],
						..Default::default()
					},
					cache_key: format!("attention:{:?}:{:?}", a_idx.raw, b_idx.raw),
					err,
				})
			},
		}
	}

	fn add_const(
		&mut self,
		name: Cow<'static, str>,
		value: f64,
		node: NodeIndex32,
	) -> Result<ConstRefIndex32, NodeIndex32> {
		let key = OrderedFloat(value);
		match self.const_map.entry(key) {
			hash_map::Entry::Vacant(entry) => {
				let index = self.const_vec.push(ConstRef { name, value, input_node: node });
				entry.insert(index);
				Ok(index)
			},
			hash_map::Entry::Occupied(entry) => {
				let index = *entry.get();
				Err(self.const_vec[index].input_node)
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
				if self.nodes_postorder[idx].parents.is_empty() {
					// root
					debug_assert!(
						self.nodes_postorder[idx].dominator == NodeIndex32::new_sentinel()
					);
					continue;
				}

				let p0 = self.nodes_postorder[idx].parents[0];
				let dominator = self.nodes_postorder[idx].parents[1..].iter().copied().fold(
					p0,
					|mut d, mut p| {
						while d != p {
							if d.to_raw() < p.to_raw() {
								d = self.nodes_postorder[d].dominator;
							} else {
								p = self.nodes_postorder[p].dominator;
							}
						}
						d
					},
				);
				if self.nodes_postorder[idx].dominator != dominator {
					self.nodes_postorder[idx].dominator = dominator;
					changed = true;
				}
			}
		}
	}

	fn find_complex_op_fingerprints(&mut self, state: &mut LoadExprState) {
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

	#[allow(clippy::manual_assert)]
	fn find_races(&mut self, state: &mut LoadExprState) {
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
					for c in 0..cols {
						if bitmap.get_bit(i, c) && bitmap.get_bit(kills, c) {
							let name =
								&self.tensor_vec[TensorRefIndex32::from_raw(c)].tensor_ref.name;
							self.err_log.log_error(i, format!("Ambiguous use of tensor {name}. Not clear whether to use the version before or after write."));
						}
					}
				}
				for &tensor_index in &me.capture {
					let was_killed = bitmap.set_bit(kills, tensor_index.to_raw());
					if was_killed {
						cold_path();
						let name = &self.tensor_vec[tensor_index].tensor_ref.name;
						self.err_log.log_error(i, format!("Double write to tensor {name}."));
					}
				}
				bitmap.and_not(i, i, kills);
			}
		}
	}

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
		a: &[usize],
		b: &[usize],
		err: &mut Vec<String>,
	) -> (ThinVec<usize>, [bool; 2]) {
		let mut is_broadcasted = [false, false];
		let mut result = ThinVec::new();
		let len = a.len().max(b.len());
		let skip_a = len - a.len();
		let skip_b = len - b.len();
		for d in 0..len {
			let dim_a = if d < skip_a { 1 } else { a[d - skip_a] };
			let dim_b = if d < skip_b { 1 } else { b[d - skip_b] };
			let dim = if dim_a == dim_b {
				dim_a
			} else if dim_b == 1 {
				is_broadcasted[1] = true;
				dim_a
			} else if dim_a == 1 {
				is_broadcasted[0] = true;
				dim_b
			} else {
				cold_path();
				err.push(format!("broadcast dimension mismatch: {:?} vs {:?}", dim_a, dim_b));
				dim_a.max(dim_b)
			};
			result.push(dim);
		}
		(result, is_broadcasted)
	}

	fn common_shape(
		a: Option<&[usize]>,
		b: Option<&[usize]>,
		err: &mut Vec<String>,
	) -> (Option<ThinVec<usize>>, [bool; 2]) {
		match (a, b) {
			(None, None) => (None, [false, false]),
			(None, Some(sh)) => (Some(ThinVec::from(sh)), [false, false]),
			(Some(sh), None) => (Some(ThinVec::from(sh)), [false, false]),
			(Some(a), Some(b)) => {
				if a.len() != b.len() {
					cold_path();
					err.push(format!("shape length mismatch: {:?} vs {:?}", a, b));
				}
				let (shape, is_broadcasted) = Self::broadcast_shapes(a, b, err);
				(Some(shape), is_broadcasted)
			},
		}
	}

	fn find_heads(&mut self) {
		for idx in self.nodes_postorder.indexes() {
			if !self.nodes_postorder[idx].is_complex() {
				continue;
			}

			let start = &self.nodes_postorder[idx];
			let start_fingerprint = start.reduction_fingerprint;

			let mut head_idx = idx;
			let mut head = start;
			loop {
				let parent_idx = head.dominator;
				if !self.nodes_postorder.is_valid(parent_idx) {
					break;
				}
				let parent = &self.nodes_postorder[parent_idx];
				let head_shape: Option<&[usize]> = head.shape.as_ref().map(|sh| &sh[..]);
				let parent_shape: Option<&[usize]> = parent.shape.as_ref().map(|sh| &sh[..]);
				let parent_fingerprint = parent.reduction_fingerprint;
				if (!parent.is_reshape() && parent_shape != head_shape)
					|| parent_fingerprint != start_fingerprint
				{
					break;
				}
				head_idx = parent_idx;
				head = parent;
			}

			self.nodes_postorder[head_idx].head_for = idx;
		}
	}

	fn find_fragments(&mut self) {
		self.fragments_preorder.raw.clear();
		for idx in self.nodes_postorder.indexes().rev() {
			let (_, item, all_parents) = self.nodes_postorder.borrow_multiple(idx);
			if item.is_input() || unlikely(item.is_dead) {
				continue;
			}
			if let Some((&first_parent, other_parents)) = item.parents.split_first()
				&& !all_parents[first_parent].is_matmul()
				&& let parent_frag = all_parents[first_parent].fragment_index()
				&& !item.is_head()
				&& other_parents.iter().all(|&p| {
					!all_parents[p].is_matmul() && all_parents[p].fragment_index() == parent_frag
				}) {
				item.set_fragment_index(parent_frag);
			} else {
				let new_frag = self.fragments_preorder.push(Fragment { head: idx });
				item.set_fragment_index(new_frag);
			}
		}
	}

	pub fn graphviz_node_label(&self, node: &Node) -> String {
		match node.node_kind {
			NodeKind::Const => {
				let name = &self.const_vec[node.const_index()].name;
				let value = self.const_vec[node.const_index()].value;
				format!(
					"<b>Const</b><br/><font color='#800080'><b>{name}</b></font><br/><font color='blue'><b>{value}</b></font>"
				)
			},
			NodeKind::Input => {
				if node.is_tensor_input() {
					let name = &self.tensor_vec[node.tensor_index()].tensor_ref.name;
					format!("<b>Tensor</b><br/><font color='#800080'><b>{name}</b></font>")
				} else {
					let name = &self.scalar_vec[node.scalar_index()].scalar_ref.name;
					format!("<b>Scalar</b><br/><font color='#800080'><b>{name}</b></font>")
				}
			},
			NodeKind::Cast => {
				let child = &self.nodes_postorder[node.children[0]];
				let from = self.dtype_to_str(child.dtype);
				let to = self.dtype_to_str(node.dtype);
				format!("<b>Cast</b><br/>{from} -&gt; {to}")
			},
			NodeKind::Reshape => {
				let child = &self.nodes_postorder[node.children[0]];
				let from = self.shape_to_str(child.can_be_batched, &child.shape);
				let to = self.shape_to_str(node.can_be_batched, &node.shape);
				format!("<b>Reshape</b><br/>{from} -&gt; {to}")
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

	fn shape_to_str(&self, can_be_batched: bool, shape: &Option<ThinVec<usize>>) -> String {
		if let Some(shape) = shape {
			let mut result = format!("&#91;");
			if can_be_batched {
				result.push_str("*,");
			}
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
		} else {
			String::new()
		}
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
				let mut names = Vec::<String>::new();
				for (idx, ten) in self.tensor_vec.iter().enumerate() {
					if state.bitmap.get_bit(i, idx) {
						names.push(ten.tensor_ref.name.to_string());
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
				} else if node.is_scalar_input() {
					writeln!(w, "\t{node_id} [shape=box, style=filled, fillcolor=\"#ffffc0\"];")?;
				} else {
					debug_assert!(node.is_const());
					writeln!(w, "\t{node_id} [shape=box, style=filled, fillcolor=\"#ffff00\"];")?;
				}
			} else if node.is_head() {
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
					"\t{node_id} -> {dom_id} [label=< >, style=dashed, color=\"#808080\", constraint=true];",
				)?;
			}
			if node.is_input() {
				//writeln!(w, "\t{{ rank = min; {node_id} }}")?;
			} else {
				let frag_index = node.fragment_index();
				if self.fragments_preorder.is_valid(frag_index) {
					let frag_head = self.fragments_preorder[frag_index].head;
					let frag_node = &self.nodes_postorder[frag_head];
					let fragment_kind = if frag_node.is_head() {
						let frag_node = &self.nodes_postorder[frag_node.head_for];
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
				for (j, &child_index) in node.children.iter().enumerate() {
					if !self.nodes_postorder.is_valid(child_index) {
						break;
					}
					let child_id = format!("node_{}", child_index.raw);
					let child = &self.nodes_postorder[child_index];
					let label = format!(
						"{}{}{}",
						self.dtype_to_str(child.dtype),
						self.shape_to_str(child.can_be_batched, &child.shape),
						if node.children_broadcast[j] {
							" <font color='red'>(broadcasted)</font>"
						} else {
							""
						}
					);
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
				let label = format!(
					"{}{}",
					self.dtype_to_str(node.dtype),
					self.shape_to_str(node.can_be_batched, &node.shape)
				);
				let input_node = self.tensor_vec[capt_idx].input_node;
				if input_node.is_sentinel() {
					let cap_id = format!("ten_{}", capt_idx.raw);
					writeln!(
						w,
						"\t{node_id} -> {cap_id} [label=<{label}>, style=bold, constraint=true];"
					)?;
					let name = &self.tensor_vec[capt_idx].tensor_ref.name;
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
			let const_idx =
				format!("node_{}", self.const_vec[sum_to_mean.const_idx].input_node.raw);
			writeln!(
				w,
				"\t{node_id} -> {const_idx} [label=< >, style=dotted, color=\"#008000\", constraint=true];"
			)?;
		}
		for (node_idx, msgs) in &self.err_log.err_vec {
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
