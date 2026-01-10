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
use std::cell::RefCell;
use std::collections::{HashMap, hash_map};
use std::hint::{cold_path, unlikely};
use std::rc::Rc;

use smallvec::SmallVec;
use thin_vec::{ThinVec, thin_vec};

use crate::define_index_type32;
use crate::new::expr::{self, CanBeBatched};
use crate::tensor::DType;
use crate::util::ToBoxedSlice;
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
	MatTimesCol,
	ColTimesRowAcc,
}

pub enum NodeKind {
	Const,
	Input(InputKind),
	Select(SelectKind),
	EvenOdd,
	Cast,
	Reshape,
	Unary(UnaryKind),
	Binary(BinaryKind),
	MatMul(MatMulKind),
	Attention,
	Reduction(ReductionKind),
}

pub enum InputKind {
	Tensor,
	Scalar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XIndexState {
	Uninitialized,
	ConstRef,
	ScalarRef,
	TensorRef,
	FragmentHead,
}

pub struct Node {
	pub node_kind: NodeKind,

	/// When `None`, the type is ambiguous. This can happen for example with Scalar inputs.
	pub dtype: Option<DType>,

	/// When `None`, the output is scalar.
	pub shape: Rc<[usize]>,

	/// If true, the node can have different value for different batch indices.
	/// This is by definition false for example for scalars.
	pub can_be_batched: bool,
	pub children: [NodeIndex32; 2],
	pub children_broadcast: [bool; 2],

	pub parents: SmallVec<[NodeIndex32; 4]>,
	pub capture: ThinVec<TensorRefIndex32>,

	pub is_dead: bool,
	pub is_trivial_head: bool,

	/// This field is overloaded in order to save space.
	/// It can have different meanings depending on the node type:
	/// - For input nodes:
	///   - ConstRefIndex32 if is_const()
	///   - ScalarRefIndex32 if is_scalar_input()
	///   - TensorRefIndex32 if is_tensor_input()
	/// - For non-input nodes:
	///   - NodeIndex32 (head node of the current fragment)
	///
	/// The related field `x_index_state` is used in debug assertions
	/// to make sure we use `x_index` correctly.
	pub x_index: UntypedIndex32,
	pub x_index_state: XIndexState,

	pub labels: ThinVec<Cow<'static, str>>,
}

impl Node {
	pub fn is_input(&self) -> bool {
		self.is_nullary()
	}

	pub fn const_index(&self) -> ConstRefIndex32 {
		debug_assert!(self.is_const());
		debug_assert!(self.x_index_state == XIndexState::ConstRef);
		ConstRefIndex32::from(self.x_index)
	}

	pub fn scalar_index(&self) -> ScalarRefIndex32 {
		debug_assert!(self.is_scalar_input());
		debug_assert!(self.x_index_state == XIndexState::ScalarRef);
		ScalarRefIndex32::from(self.x_index)
	}

	pub fn tensor_index(&self) -> TensorRefIndex32 {
		debug_assert!(self.is_tensor_input());
		debug_assert!(self.x_index_state == XIndexState::TensorRef);
		TensorRefIndex32::from(self.x_index)
	}

	pub fn fragment_head_index(&self) -> NodeIndex32 {
		debug_assert!(!self.is_input());
		debug_assert!(
			self.x_index_state == XIndexState::FragmentHead
				|| self.x_index_state == XIndexState::Uninitialized
		);
		NodeIndex32::from(self.x_index)
	}

	pub fn set_fragment_head_index(&mut self, fragment_index: NodeIndex32) {
		debug_assert!(!self.is_input());
		debug_assert!(
			self.x_index_state == XIndexState::Uninitialized
				|| self.x_index_state == XIndexState::FragmentHead
		);
		self.x_index = fragment_index.to_untyped();
		self.x_index_state = XIndexState::FragmentHead;
	}

	pub fn is_const(&self) -> bool {
		matches!(self.node_kind, NodeKind::Const)
	}

	pub fn is_scalar_input(&self) -> bool {
		matches!(self.node_kind, NodeKind::Input(InputKind::Scalar))
	}

	pub fn is_tensor_input(&self) -> bool {
		matches!(self.node_kind, NodeKind::Input(InputKind::Tensor))
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

	pub fn is_trivial(&self) -> bool {
		matches!(
			self.node_kind,
			NodeKind::Cast | NodeKind::Reshape | NodeKind::Unary(UnaryKind::Neg)
		)
	}

	pub fn is_select(&self) -> bool {
		matches!(self.node_kind, NodeKind::Select(_))
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

	pub fn broadcasts_child(&self, node_idx: NodeIndex32) -> bool {
		self.children
			.iter()
			.zip(self.children_broadcast)
			.any(|(&child_idx, broadcast)| child_idx == node_idx && broadcast)
	}

	pub fn shape(&self) -> &[usize] {
		self.shape.as_ref()
	}
}

define_index_type32!(NodeIndex32);
type NodeVec = IndexVec<NodeIndex32, Node>;

pub struct TensorRef {
	pub tensor_ref: Rc<expr::TensorRef>,
	pub input_node: NodeIndex32,
	pub output_node: NodeIndex32,
	pub shape: Rc<[usize]>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FragmentKind {
	Trivial,
	Reduction,
	MatMul,
	Attention,
	Elementwise,
}

pub struct Fragment {
	pub head: NodeIndex32,
	pub kind: FragmentKind,
	pub reduced_children: SmallVec<[FragIndex32; 4]>,
	pub reduced_parent_count: u32,
	pub parents: SmallVec<[FragIndex32; 4]>,
	pub children: SmallVec<[FragIndex32; 4]>,
	pub kernel: KernelIndex32,
}

define_index_type32!(FragIndex32);
type FragVec = IndexVec<FragIndex32, Fragment>;

define_index_type32!(FragPreorderIndex32);

define_index_type32!(KernelIndex32);

pub struct KernelBuilderData {
	nodes_postorder: NodeVec,
	const_vec: ConstVec,
	scalar_map: HashMap<*const expr::ScalarRef, ScalarRefIndex32>,
	scalar_vec: ScalarVec,
	tensor_map: HashMap<*const expr::TensorRef, TensorRefIndex32>,
	tensor_vec: TensorVec,
	frags_postorder: FragVec,
	node_cache: HashMap<String, NodeIndex32>,
	err_log: ErrorLog,
}

pub struct KernelBuilder {
	data: RefCell<KernelBuilderData>,
}

pub struct Expr<'a> {
	kernel_builder: &'a KernelBuilder,
	node_index: NodeIndex32,
}

struct LoadedNode {
	node: Node,
	cache_key: String,
	err: ThinVec<String>,
}

pub struct LoadExprState {
	visited: HashMap<*const ExprNode, NodeIndex32>,
	n_complex: u32,
	bitmap: IndexBitmap<NodeIndex32>,
}

pub struct ErrorLog {
	err_map: HashMap<NodeIndex32, usize>,
	err_vec: ThinVec<(NodeIndex32, Vec<String>)>,
}

impl ErrorLog {
	pub fn new() -> Self {
		Self {
			err_map: HashMap::new(),
			err_vec: ThinVec::new(),
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

impl KernelBuilder {
	pub fn new() -> Self {
		KernelBuilder {
			data: RefCell::new(KernelBuilderData {
				nodes_postorder: NodeVec::with_capacity(32),
				const_vec: ConstVec::with_capacity(4),
				scalar_map: HashMap::new(),
				scalar_vec: ScalarVec::with_capacity(4),
				tensor_map: HashMap::new(),
				tensor_vec: TensorVec::with_capacity(4),
				frags_postorder: FragVec::with_capacity(16),
				node_cache: HashMap::new(),
				err_log: ErrorLog::new(),
			}),
		}
	}

	pub fn new_const<S: Into<Cow<'static, str>>>(&self, name: S, value: f64) -> Expr {
		let mut data = self.data.borrow_mut();
		let next_index = data.nodes_postorder.next_index();
		match data.add_const(name.into(), value, next_index) {
			Err(existing_node) => {
				//
				Expr {
					kernel_builder: self,
					node_index: existing_node,
				}
			},
			Ok(idx) => {
				let real_next_index =
					data.nodes_postorder.push(KernelBuilderData::new_nullary_node(
						None,
						Rc::from([]),
						false,
						NodeKind::Const,
						idx.to_untyped(),
						XIndexState::ConstRef,
					));
				debug_assert!(next_index == real_next_index);
				Expr {
					kernel_builder: self,
					node_index: next_index,
				}
			},
		}
	}

	pub fn new_tensor_input<S: Into<Cow<'static, str>>>(
		&self,
		name: S,
		dtype: DType,
		shape: &[usize],
		can_be_batched: CanBeBatched,
	) -> Expr {
		let mut data = self.data.borrow_mut();
		let next_index = data.nodes_postorder.next_index();
		match data.tensor_map.entry(name.into()) {
			hash_map::Entry::Vacant(entry) => {
				let idx = data.tensor_vec.push(TensorRef {
					input_node: next_index,
					output_node: NodeIndex32::new_sentinel(),
					shape,
					can_be_batched,
					tensor_ref,
				});
				entry.insert(idx);
				let real_next_index =
					data.nodes_postorder.push(KernelBuilderData::new_nullary_node(
						Some(dtype),
						shape,
						can_be_batched != CanBeBatched::No,
						NodeKind::Input(InputKind::Tensor),
						idx.to_untyped(),
						XIndexState::TensorRef,
					));
				debug_assert!(next_index == real_next_index);
				Expr {
					kernel_builder: self,
					node_index: next_index,
				}
			},
			hash_map::Entry::Occupied(entry) => {
				let index = *entry.get();
				let existing_node = self.tensor_vec[index].input_node;
				Expr {
					kernel_builder: self,
					node_index: existing_node,
				}
			},
		}
	}
}

impl KernelBuilderData {
	pub fn analyze(&mut self) -> Result<(), ()> {
		//self.load_expr(expr, &mut state);
		self.remove_dead_code();
		//self.find_races(&mut state);
		self.find_fragments();
		self.find_trivial_fragments();
		self.make_fragment_graph();
		self.find_reduced_children();
		self.find_kernels();

		self.err_log.check_errors()
	}

	pub fn new_nullary_node(
		dtype: Option<DType>,
		shape: Rc<[usize]>,
		can_be_batched: bool,
		node_kind: NodeKind,
		x_index: UntypedIndex32,
		x_index_state: XIndexState,
	) -> Node {
		Node {
			node_kind,
			dtype,
			shape,
			can_be_batched,
			parents: SmallVec::new(),
			children: [NodeIndex32::new_sentinel(), NodeIndex32::new_sentinel()],
			children_broadcast: [false, false],
			capture: ThinVec::new(),
			is_dead: false,
			is_trivial_head: false,
			x_index,
			x_index_state,
			labels: ThinVec::new(),
		}
	}

	pub fn new_unary_node(expr: &ExprNode, node_kind: NodeKind, child_idx: NodeIndex32) -> Node {
		Node {
			node_kind,
			dtype: expr.dtype,
			shape: expr.shape.clone(),
			can_be_batched: expr.can_be_batched,
			parents: SmallVec::new(),
			children: [child_idx, NodeIndex32::new_sentinel()],
			children_broadcast: [false, false],
			capture: ThinVec::new(),
			is_dead: false,
			is_trivial_head: false,
			x_index: UntypedIndex32::new_sentinel(),
			x_index_state: XIndexState::Uninitialized,
			labels: ThinVec::new(),
		}
	}

	pub fn new_binary_node(
		expr: &ExprNode,
		bin_expr: &ExprBinary,
		node_kind: NodeKind,
		lhs_idx: NodeIndex32,
		rhs_idx: NodeIndex32,
	) -> Node {
		Node {
			node_kind,
			dtype: expr.dtype,
			shape: expr.shape.clone(),
			can_be_batched: expr.can_be_batched,
			parents: SmallVec::new(),
			children: [lhs_idx, rhs_idx],
			children_broadcast: [bin_expr.lhs_broadcasted, bin_expr.rhs_broadcasted],
			capture: ThinVec::new(),
			is_dead: false,
			is_trivial_head: false,
			x_index: UntypedIndex32::new_sentinel(),
			x_index_state: XIndexState::Uninitialized,
			labels: ThinVec::new(),
		}
	}

	// TODO - refactor to make non recursive
	fn load_expr(&mut self, expr: &ExprNode, state: &mut LoadExprState) -> NodeIndex32 {
		let expr_key = std::ptr::from_ref(expr);
		if let Some(index) = state.visited.get(&expr_key) {
			return *index;
		}

		let t = match &expr.kind {
			ExprKind::Input(input) => {
				self.load_input(expr, input) //
			},
			ExprKind::Capture(capture) => {
				let child_idx = self.load_expr(&capture.expr, state);
				self.load_capture(capture, child_idx)
			},
			ExprKind::Cast(cast) => {
				let child_idx = self.load_expr(&cast.expr, state);
				self.load_cast(expr, child_idx)
			},
			ExprKind::Label(label) => {
				let child_idx = self.load_expr(&label.expr, state);
				self.load_label(label, child_idx)
			},
			ExprKind::Reshape(reshape) => {
				let child_idx = self.load_expr(&reshape.expr, state);
				self.load_reshape(expr, child_idx)
			},
			ExprKind::Unary(unary) => {
				let child_idx = self.load_expr(&unary.expr, state);
				self.load_unary(expr, unary, child_idx)
			},
			ExprKind::Binary(binary) => {
				let a_idx = self.load_expr(&binary.lhs, state);
				let b_idx = self.load_expr(&binary.rhs, state);
				self.load_binary(expr, binary, a_idx, b_idx)
			},
		};

		let loaded = match t {
			Ok(loaded) => loaded,
			Err(existing_node) => {
				state.visited.insert(expr_key, existing_node);
				for err in &expr.local_errors {
					self.err_log.log_error(existing_node, err.clone());
				}
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
		for err in &expr.local_errors {
			self.err_log.log_error(next_index, err.clone());
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

	fn load_input(
		&mut self,
		expr: &ExprNode,
		input: &ExprInput,
	) -> Result<LoadedNode, NodeIndex32> {
		let x_index;
		let x_index_state;
		let node_kind;
		match input {
			ExprInput::Tensor(tensor_ref) => {
				match self.add_tensor_input(tensor_ref.clone(), self.nodes_postorder.next_index()) {
					Ok(tensor_index) => {
						x_index = tensor_index.to_untyped();
						x_index_state = XIndexState::TensorRef;
						node_kind = NodeKind::Input(InputKind::Tensor);
					},
					Err(existing_node) => {
						return Err(existing_node);
					},
				}
			},
			ExprInput::Scalar(scalar_ref) => {
				match self.add_scalar_input(scalar_ref.clone(), self.nodes_postorder.next_index()) {
					Ok(scalar_index) => {
						x_index = scalar_index.to_untyped();
						x_index_state = XIndexState::ScalarRef;
						node_kind = NodeKind::Input(InputKind::Scalar);
					},
					Err(existing_node) => {
						return Err(existing_node);
					},
				}
			},
		}
		Ok(LoadedNode {
			node: Self::new_nullary_node(expr, node_kind, x_index, x_index_state),
			cache_key: String::new(),
			err: ThinVec::new(),
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
				parents: SmallVec::new(),
				children: [child_idx, NodeIndex32::new_sentinel()],
				children_broadcast: [false, false],
				capture: thin_vec![tensor_idx],
				is_dead: false,
				is_trivial_head: false,
				x_index: UntypedIndex32::new_sentinel(),
				x_index_state: XIndexState::Uninitialized,
				labels: ThinVec::new(),
			},
			cache_key: format!("identity:{:?}", child_idx.raw),
			err: ThinVec::new(),
		})
	}

	fn load_reshape(
		&self,
		expr: &ExprNode,
		child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		Ok(LoadedNode {
			node: Self::new_unary_node(expr, NodeKind::Reshape, child_idx),
			cache_key: format!("reshape:{:?}:{}", expr.shape(), child_idx.raw),
			err: ThinVec::new(),
		})
	}

	fn load_cast(
		&self,
		expr: &ExprNode,
		child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		Ok(LoadedNode {
			node: Self::new_unary_node(expr, NodeKind::Cast, child_idx),
			cache_key: format!("cast:{:?}:{:?}", expr.dtype, child_idx.raw),
			err: ThinVec::new(),
		})
	}

	fn load_label(
		&mut self,
		label: &ExprLabel,
		child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		let child = &mut self.nodes_postorder[child_idx];
		child.labels.push(label.label.clone());
		Err(child_idx)
	}

	fn load_unary(
		&self,
		expr: &ExprNode,
		unary: &ExprUnary,
		child_idx: NodeIndex32,
	) -> Result<LoadedNode, NodeIndex32> {
		match unary.kind {
			ExprUnaryKind::NoOp => Err(child_idx),
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
				Ok(LoadedNode {
					node: Self::new_unary_node(expr, NodeKind::Unary(unary_kind), child_idx),
					cache_key: format!("unary:{:?}:{:?}", unary_kind, child_idx.raw),
					err: ThinVec::new(),
				})
			},
			ExprUnaryKind::Sum | ExprUnaryKind::Max => {
				let reduction_kind = match unary.kind {
					ExprUnaryKind::Sum => ReductionKind::Sum,
					ExprUnaryKind::Max => ReductionKind::Max,
					_ => unreachable!(),
				};
				Ok(LoadedNode {
					node: Self::new_unary_node(
						expr,
						NodeKind::Reduction(reduction_kind),
						child_idx,
					),
					cache_key: format!("reduction:{:?}:{:?}", reduction_kind, child_idx.raw),
					err: ThinVec::new(),
				})
			},
			ExprUnaryKind::SelectEven | ExprUnaryKind::SelectOdd => {
				let select_kind = match unary.kind {
					ExprUnaryKind::SelectEven => SelectKind::Even,
					ExprUnaryKind::SelectOdd => SelectKind::Odd,
					_ => unreachable!(),
				};
				Ok(LoadedNode {
					node: Self::new_unary_node(expr, NodeKind::Select(select_kind), child_idx),
					cache_key: format!("select:{:?}:{:?}", select_kind, child_idx.raw),
					err: ThinVec::new(),
				})
			},
		}
	}

	fn load_binary(
		&self,
		expr: &ExprNode,
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
				Ok(LoadedNode {
					node: Self::new_binary_node(
						expr,
						binary,
						NodeKind::Binary(binary_kind),
						a_idx,
						b_idx,
					),
					cache_key: format!("binary:{:?}:{:?}:{:?}", binary_kind, a_idx.raw, b_idx.raw),
					err: ThinVec::new(),
				})
			},
			ExprBinaryKind::First => Err(a_idx),
			ExprBinaryKind::RowTimesMat => Ok(LoadedNode {
				node: Self::new_binary_node(
					expr,
					binary,
					NodeKind::MatMul(MatMulKind::RowTimesMat),
					a_idx,
					b_idx,
				),
				cache_key: format!("row_times_mat:{:?}:{:?}", a_idx.raw, b_idx.raw),
				err: ThinVec::new(),
			}),
			ExprBinaryKind::MatTimesCol => Ok(LoadedNode {
				node: Self::new_binary_node(
					expr,
					binary,
					NodeKind::MatMul(MatMulKind::MatTimesCol),
					b_idx,
					a_idx,
				),
				cache_key: format!("mat_times_col:{:?}:{:?}", b_idx.raw, a_idx.raw),
				err: ThinVec::new(),
			}),
			ExprBinaryKind::ColTimesRowAcc => Ok(LoadedNode {
				node: Self::new_binary_node(
					expr,
					binary,
					NodeKind::MatMul(MatMulKind::ColTimesRowAcc),
					a_idx,
					b_idx,
				),
				cache_key: format!("col_times_row_acc:{:?}:{:?}", a_idx.raw, b_idx.raw),
				err: ThinVec::new(),
			}),
			ExprBinaryKind::EvenOdd => Ok(LoadedNode {
				node: Self::new_binary_node(expr, binary, NodeKind::EvenOdd, a_idx, b_idx),
				cache_key: format!("even_odd:{:?}:{:?}", a_idx.raw, b_idx.raw),
				err: ThinVec::new(),
			}),
			ExprBinaryKind::Attention => Ok(LoadedNode {
				node: Self::new_binary_node(expr, binary, NodeKind::Attention, a_idx, b_idx),
				cache_key: format!("attention:{:?}:{:?}", a_idx.raw, b_idx.raw),
				err: ThinVec::new(),
			}),
		}
	}

	fn add_const(
		&mut self,
		name: Cow<'static, str>,
		value: f64,
		node: NodeIndex32,
	) -> Result<ConstRefIndex32, NodeIndex32> {
		Ok(self.const_vec.push(ConstRef { name, value, input_node: node }))
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
				let index = self.tensor_vec.push(TensorRef {
					input_node: node,
					output_node: NodeIndex32::new_sentinel(),
					shape: tensor_ref.shape.clone(),
					can_be_batched: tensor_ref.can_be_batched,
					tensor_ref,
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
				let index = self.tensor_vec.push(TensorRef {
					input_node: NodeIndex32::new_sentinel(),
					output_node: node,
					shape: tensor_ref.shape.clone(),
					can_be_batched: tensor_ref.can_be_batched,
					tensor_ref,
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
				std::mem::replace(&mut self.nodes_postorder[i].parents, SmallVec::new());
			parents.retain(|p| !self.nodes_postorder[*p].is_dead);
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

	fn find_fragments(&mut self) {
		for idx in self.nodes_postorder.indexes().rev() {
			let (_, item, all_parents) = self.nodes_postorder.borrow_multiple(idx);
			if item.is_input() || unlikely(item.is_dead) {
				continue;
			}

			if !item.is_complex()
				&& let Some((&first_parent, other_parents)) = item.parents.split_first()
				&& let first_parent = &all_parents[first_parent]
				&& (!first_parent.is_complex() || first_parent.is_reduction())
				&& !first_parent.is_select()
				&& !first_parent.broadcasts_child(idx)
				&& let parent_frag = first_parent.fragment_head_index()
				&& other_parents.iter().all(|&p| {
					let parent = &all_parents[p];
					parent.fragment_head_index() == parent_frag
						&& (!parent.is_complex() || parent.is_reduction())
						&& !parent.is_select()
						&& !parent.broadcasts_child(idx)
				}) {
				item.set_fragment_head_index(parent_frag);
			} else {
				item.set_fragment_head_index(idx);
			}
		}
	}

	fn find_trivial_fragments(&mut self) {
		let mut trails: Vec<Box<[NodeIndex32]>> = Vec::new();
		let mut trail = Vec::new();
		for idx in self.nodes_postorder.indexes().rev() {
			let item = &self.nodes_postorder[idx];
			if item.is_input() || unlikely(item.is_dead) {
				continue;
			}

			if item.fragment_head_index() == idx {
				trail.clear();
				let mut idx = idx;
				let mut item = item;
				let mut split = 0;
				let mut min_bits = usize::MAX;
				while item.is_trivial() && !item.is_captured() {
					if let Some(dtype) = item.dtype
						&& dtype.bits() < min_bits
					{
						min_bits = dtype.bits();
						split = trail.len();
					}
					trail.push(idx);
					idx = item.children[0];
					item = &self.nodes_postorder[idx];
				}
				if !item.is_input() && !item.is_captured() {
					unsafe { trail.set_len(split) }
				}
				if let Some(&last_idx) = trail.last() {
					// Note: This looks like a bug, but it's actually very important.
					// We need to fix the fragment_head_index of all nodes in the current
					// fragment. We will redirect them to the new head, which is `inp_idx`.
					// In the `else` branch, the head index of all nodes points to first_idx;
					// we do:
					//     node.fragment_head_index = nodes[first_idx].fragment_head_index
					let first_idx = trail[0];
					let inp_idx = self.nodes_postorder[last_idx].children[0];
					self.nodes_postorder[first_idx].set_fragment_head_index(inp_idx);

					let trail = &trail[..];
					trails.push(trail.to_boxed_slice());
				}
			} else {
				let frag = item.fragment_head_index();
				let new_frag = self.nodes_postorder[frag].fragment_head_index();
				if new_frag != frag {
					self.nodes_postorder[idx].set_fragment_head_index(new_frag);
				}
			}
		}

		std::mem::drop(trail);
		for trail in trails {
			let head = trail[0];
			self.nodes_postorder[head].is_trivial_head = true;
			for idx in trail {
				self.nodes_postorder[idx].set_fragment_head_index(head);
			}
		}
	}

	#[allow(clippy::expect_used)]
	fn make_fragment_graph(&mut self) {
		// First find all fragments and their children.
		struct FragData {
			head: NodeIndex32,
			postorder_idx: FragIndex32,
			children: SmallVec<[FragPreorderIndex32; 4]>,
		}
		let mut map: HashMap<NodeIndex32, FragPreorderIndex32> = HashMap::new();
		let mut preorder: IndexVec<FragPreorderIndex32, FragData> = IndexVec::new();
		let mut to_process: Vec<FragPreorderIndex32> = Vec::new();
		for idx in self.nodes_postorder.indexes().rev() {
			let item = &self.nodes_postorder[idx];
			if item.is_input() || unlikely(item.is_dead) || item.fragment_head_index() != idx {
				continue;
			}

			let preorder_idx = preorder.push(FragData {
				head: idx,
				postorder_idx: FragIndex32::new_sentinel(),
				children: SmallVec::new(),
			});
			map.insert(idx, preorder_idx);

			if item.is_root() {
				to_process.push(preorder_idx);
			} else {
				for &p in &item.parents {
					let parent_head = self.nodes_postorder[p].fragment_head_index();
					let parent_preorder_idx = *map
						.get(&parent_head)
						.expect("parent fragment must have been created already");
					preorder[parent_preorder_idx].children.push(preorder_idx);
				}
			}
		}
		to_process.reverse();
		let root_count = to_process.len();

		// Now use the children links to build the final graph.
		// Trivial fragments are intentionally not cached, which means they will be duplicated if
		// linked from multiple parents.
		let mut values: Vec<FragIndex32> = Vec::new();
		let mut phase2 = false;
		while let Some(item) = to_process.pop() {
			let Some(item_data) = preorder.get(item) else {
				// Sentinel item. Go to phase 2.
				debug_assert!(!phase2);
				phase2 = true;
				continue;
			};

			let children: &[FragPreorderIndex32] = &item_data.children;
			if !phase2 {
				if self.frags_postorder.is_valid(item_data.postorder_idx) {
					values.push(item_data.postorder_idx);
					continue;
				}
				if !children.is_empty() {
					to_process.push(item);
					to_process.push(FragPreorderIndex32::new_sentinel());
					let x = to_process.len();
					to_process.extend_from_slice(children);
					to_process[x..].reverse();
					continue;
				}
			}

			phase2 = false;
			let kind = self.fragment_kind(item_data.head).unwrap();
			let children = &values[values.len() - children.len()..];
			let postorder_idx = self.frags_postorder.push(Fragment {
				head: item_data.head,
				kind,
				reduced_children: SmallVec::new(),
				reduced_parent_count: 0,
				parents: SmallVec::new(),
				children: SmallVec::from_slice(children),
				kernel: KernelIndex32::new_sentinel(),
			});
			for &child in children {
				self.frags_postorder[child].parents.push(postorder_idx);
			}
			values.truncate(values.len() - children.len());
			values.push(postorder_idx);
			if kind != FragmentKind::Trivial {
				preorder[item].postorder_idx = postorder_idx;
			}
		}
		debug_assert!(!phase2);
		debug_assert!(values.len() == root_count);
	}

	fn find_reduced_children(&mut self) {
		for idx in self.frags_postorder.indexes() {
			self.frags_postorder[idx].children.sort_unstable();
			self.frags_postorder[idx].children.dedup();
			self.frags_postorder[idx].children.shrink_to_fit();

			self.frags_postorder[idx].parents.sort_unstable();
			self.frags_postorder[idx].parents.dedup();
			self.frags_postorder[idx].parents.shrink_to_fit();

			let mut reduced_children = SmallVec::new();
			for &child_idx in &self.frags_postorder[idx].children {
				let child_frag = &self.frags_postorder[child_idx];
				if child_frag.kind != FragmentKind::Trivial {
					reduced_children.push(child_idx);
				} else {
					reduced_children.extend_from_slice(&child_frag.reduced_children);
				}
			}

			reduced_children.sort_unstable();
			reduced_children.dedup();
			self.frags_postorder[idx].reduced_children = reduced_children;
		}

		self.transitive_reduction();
	}

	fn transitive_reduction(&mut self) {
		let mut reach: IndexVec<FragIndex32, bool> =
			IndexVec::from_vec(vec![false; self.frags_postorder.len()]);
		let mut kept_children = Vec::new();
		for src in self.frags_postorder.indexes().rev() {
			let src_frag = &mut self.frags_postorder[src];
			if src_frag.kind == FragmentKind::Trivial {
				continue;
			}
			let reduced_children = std::mem::take(&mut src_frag.reduced_children);
			kept_children.clear();
			for &dst in &reduced_children {
				reach.raw.fill(false);
				reach[src] = true;
				{
					for &j in &reduced_children {
						if j != dst {
							reach[j] = true;
						}
					}
				}
				for i in (dst.raw + 1..src.raw).rev() {
					let i = FragIndex32::new(i);
					if reach[i] {
						for &j in &self.frags_postorder[i].reduced_children {
							reach[j] = true;
						}
					}
				}
				if !reach[dst] {
					kept_children.push(dst);
					self.frags_postorder[dst].reduced_parent_count += 1;
				}
			}
			self.frags_postorder[src].reduced_children = SmallVec::from(&kept_children[..]);
		}
	}

	#[allow(clippy::too_many_lines)]
	fn find_kernels(&mut self) {
		#[derive(Copy, Clone)]
		struct KernelConfig {
			kind: FragmentKind,
			shape_node: NodeIndex32,
		}

		impl KernelConfig {
			fn merge(
				child: &KernelConfig,
				parent: &KernelConfig,
				nodes: &NodeVec,
			) -> Option<KernelConfig> {
				let child_node = &nodes[child.shape_node];
				let parent_node = &nodes[parent.shape_node];
				let child_shape = child_node.shape();
				let parent_shape = parent_node.shape();
				if child.kind == FragmentKind::Reduction {
					let (c_dims, [c_top]) = KernelBuilder::split_shape::<1>(child_shape);
					let (p_dims, [p_top]) = KernelBuilder::split_shape::<1>(parent_shape);
					let c_elems = c_dims.iter().product::<usize>();
					let p_elems = p_dims.iter().product::<usize>();
					let elems_eq = c_elems == p_elems
						&& child_node.can_be_batched == parent_node.can_be_batched;
					match parent.kind {
						FragmentKind::Reduction => {
							if elems_eq && c_top == p_top {
								Some(*child)
							} else {
								None
							}
						},
						FragmentKind::Elementwise | FragmentKind::Trivial => {
							if elems_eq && (c_top == p_top || p_top == 1) {
								Some(*child)
							} else {
								None
							}
						},
						_ => None,
					}
				} else {
					let child_elems = child_shape.iter().product::<usize>();
					let parent_elems = parent_shape.iter().product::<usize>();
					let elems_eq = child_elems == parent_elems
						&& child_node.can_be_batched == parent_node.can_be_batched;
					match child.kind {
						FragmentKind::Reduction => unreachable!(),
						FragmentKind::Elementwise | FragmentKind::Trivial => match parent.kind {
							FragmentKind::Elementwise
							| FragmentKind::Trivial
							| FragmentKind::Reduction => {
								if elems_eq {
									Some(*parent)
								} else {
									None
								}
							},
							_ => None,
						},
						FragmentKind::MatMul | FragmentKind::Attention => match parent.kind {
							FragmentKind::Elementwise | FragmentKind::Trivial => {
								if elems_eq {
									Some(*child)
								} else {
									None
								}
							},
							_ => None,
						},
					}
				}
			}
		}

		let mut kernels: IndexVec<KernelIndex32, KernelConfig> = IndexVec::new();
		for idx in self.frags_postorder.indexes() {
			let parent = &self.frags_postorder[idx];
			let shape_node = match parent.kind {
				FragmentKind::Reduction => {
					let head_node = &self.nodes_postorder[parent.head];
					head_node.children[0]
				},
				_ => parent.head,
			};
			let parent_config = KernelConfig { kind: parent.kind, shape_node };

			if let [child_idx] = &parent.reduced_children[..]
				&& let child_idx = *child_idx
				&& let child = &self.frags_postorder[child_idx]
				&& let child_kernel = child.kernel
				&& let child_config = &mut kernels[child_kernel]
				&& child.reduced_parent_count == 1
				&& let Some(merged_config) =
					KernelConfig::merge(child_config, &parent_config, &self.nodes_postorder)
			{
				*child_config = merged_config;
				let frag = &mut self.frags_postorder[idx];
				frag.kernel = child_kernel;
			} else {
				let frag = &mut self.frags_postorder[idx];
				frag.kernel = kernels.push(parent_config);
			}
		}
		for idx in self.frags_postorder.indexes() {
			if self.frags_postorder[idx].kind == FragmentKind::Trivial {
				debug_assert!(self.frags_postorder[idx].parents.len() == 1);
				let parent_idx = self.frags_postorder[idx].parents[0];
				//debug_assert!(!self.frags_postorder[parent_idx].kernel.is_sentinel());
				self.frags_postorder[idx].kernel = self.frags_postorder[parent_idx].kernel;
			}
		}
	}

	fn fragment_kind(&self, idx: NodeIndex32) -> Option<FragmentKind> {
		if let Some(item) = self.nodes_postorder.get(idx)
			&& let head = item.fragment_head_index()
			&& let Some(item) = self.nodes_postorder.get(head)
		{
			if item.is_trivial_head {
				Some(FragmentKind::Trivial)
			} else if item.is_reduction() {
				Some(FragmentKind::Reduction)
			} else if item.is_matmul() {
				Some(FragmentKind::MatMul)
			} else if item.is_attention() {
				Some(FragmentKind::Attention)
			} else {
				Some(FragmentKind::Elementwise)
			}
		} else {
			None
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
			NodeKind::Input(InputKind::Tensor) => {
				let name = &self.tensor_vec[node.tensor_index()].tensor_ref.name;
				format!("<b>Tensor</b><br/><font color='#800080'><b>{name}</b></font>")
			},
			NodeKind::Input(InputKind::Scalar) => {
				let name = &self.scalar_vec[node.scalar_index()].scalar_ref.name;
				format!("<b>Scalar</b><br/><font color='#800080'><b>{name}</b></font>")
			},
			NodeKind::Cast => {
				let child = &self.nodes_postorder[node.children[0]];
				let from = self.dtype_to_str(child.dtype);
				let to = self.dtype_to_str(node.dtype);
				format!("<b>Cast</b><br/>{from} -&gt; {to}")
			},
			NodeKind::Reshape => {
				let child = &self.nodes_postorder[node.children[0]];
				let from = self.shape_to_str(child.can_be_batched, Some(child.shape.as_ref()));
				let to = self.shape_to_str(node.can_be_batched, Some(node.shape.as_ref()));
				format!("<b>Reshape</b><br/>{from} -&gt; {to}")
			},
			NodeKind::Select(select) => match select {
				SelectKind::Even => "<b>Select Even</b>".to_string(),
				SelectKind::Odd => "<b>Select Odd</b>".to_string(),
			},
			NodeKind::EvenOdd => "<b>EvenOdd</b>".to_string(),
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
				MatMulKind::ColTimesRowAcc => "<b>ACC(col * row) → MAT</b>".to_string(),
				MatMulKind::MatTimesCol => "<b>MAT * col</b>".to_string(),
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

	fn shape_to_str(&self, can_be_batched: bool, shape: Option<&[usize]>) -> String {
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

	pub fn print_graphviz(&self) -> String {
		let mut s = String::new();
		let _ = self.__print_graphviz(&mut s, None);
		s
	}

	#[allow(clippy::too_many_lines)]
	pub fn __print_graphviz<W: std::fmt::Write>(
		&self,
		w: &mut W,
		mut state: Option<&mut LoadExprState>,
	) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		writeln!(w, "\trankdir=BT;")?;
		writeln!(w, "\tnewrank=true;")?;
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
			/*} else if node.is_head() {
			writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ccccff\"];")?;*/
			} else if node.is_captured() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#cceeff\"];")?;
			} else if node.is_fork() {
				if unlikely(node.is_dead) {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#cccccc\"];")?;
				} else {
					writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffcccc\"];")?;
				}
			} else if node.is_reduction() || node.is_matmul() || node.is_attention() {
				writeln!(w, "\t{node_id} [style=filled, fillcolor=\"#ffccff\"];")?;
			}
			if node.is_input() {
				//writeln!(w, "\t{{ rank = min; {node_id} }}")?;
			} else {
				let frag_idx = node.fragment_head_index();
				if let Some(frag_kind) = self.fragment_kind(frag_idx) {
					let (fragment_kind, color) = match frag_kind {
						FragmentKind::Trivial => ("Trivial", "#a0a0a0"),
						FragmentKind::Reduction => ("Reduction", "#c00000"),
						FragmentKind::MatMul => ("MatMul", "#c00000"),
						FragmentKind::Attention => ("Attention", "#c00000"),
						FragmentKind::Elementwise => ("Element-wise", "black"),
					};
					writeln!(
						w,
						"\tsubgraph cluster_{} {{ label=<<font color='{color}'>&#91;{}&#93; {fragment_kind}</font>> labelloc=\"b\" labeljust=\"l\" {node_id} color=\"{color}\"; }}",
						frag_idx.raw, frag_idx.raw
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
						self.shape_to_str(child.can_be_batched, Some(child.shape.as_ref())),
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
					let frag_index = node.x_index;
					let extra_style = if !frag_index.is_sentinel()
						&& (child.is_input() || frag_index != child.x_index)
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
			for (label_idx, label) in node.labels.iter().enumerate() {
				let label = Self::sanitize_for_graphviz_html(label);
				writeln!(w, "\t{{ rank=same; {node_id}; {node_id}_label_{label_idx}; }};")?;
				writeln!(
					w,
					"\t{node_id}_label_{label_idx} [label=<<font color='purple'><b>{label}</b></font>>, shape=box, style=filled, fillcolor=\"#eeeeff\"];",
				)?;
				writeln!(
					w,
					"\t{node_id} -> {node_id}_label_{} [style=dashed, color=\"#800080\"];",
					label_idx
				)?;
			}
			for &capt_idx in &node.capture {
				let label = format!(
					"{}{}",
					self.dtype_to_str(node.dtype),
					self.shape_to_str(node.can_be_batched, Some(node.shape.as_ref()))
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
					//writeln!(w, "\t{{ rank=same; {node_id}; {cap_id}; }};")?;
				} else {
					let cap_id = format!("node_{}", input_node.raw);
					writeln!(w, "\t{node_id} -> {cap_id} [label=<{label}>, constraint=true];")?;
				}
			}
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

	pub fn print_fragment_graphviz(&self) -> String {
		let mut s = String::new();
		let _ = self.__print_fragment_graphviz(&mut s);
		s
	}

	pub fn __print_fragment_graphviz<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		writeln!(w, "\trankdir=BT;")?;
		for i in self.frags_postorder.indexes() {
			let fragment = &self.frags_postorder[i];
			let node_id = format!("frag_{}", i.raw);
			let (fragment_kind, color) = match fragment.kind {
				FragmentKind::Trivial => ("Trivial", "#a0a0a0"),
				FragmentKind::Reduction => ("Reduction", "#c00000"),
				FragmentKind::MatMul => ("MatMul", "#c00000"),
				FragmentKind::Attention => ("Attention", "#c00000"),
				FragmentKind::Elementwise => ("Element-wise", "black"),
			};
			let label = format!("<b>&#91;{}&#93; {fragment_kind}</b>", fragment.head.raw);
			writeln!(
				w,
				"\t{node_id} [label=<<font color='{color}'>{label}</font>>, color=\"{color}\"];"
			)?;
			if !fragment.kernel.is_sentinel() {
				writeln!(
					w,
					"\tsubgraph cluster_kernel_{} {{ {node_id} color=\"#0000ff\"; style=rounded; }}",
					fragment.kernel.raw
				)?;
			}
			if fragment.kind != FragmentKind::Trivial {
				for &c in &fragment.reduced_children {
					let child_id = format!("frag_{}", c.raw);
					writeln!(
						w,
						"\t{child_id} -> {node_id} [label=< >, style=bold, color=\"#ff0000\", constraint=true];",
					)?;
				}
			}
			for &p in &fragment.parents {
				let parent_id = format!("frag_{}", p.raw);
				writeln!(
					w,
					"\t{node_id} -> {parent_id} [label=< >, color=\"#000000\", constraint=true];",
				)?;
			}
		}
		writeln!(w, "}}")?;
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------

pub fn dag_eval<'a, NodeHandle: Clone, Value>(
	mut to_process: Vec<NodeHandle>,
	sentinel: NodeHandle,
	is_valid_handle: &'a mut dyn FnMut(NodeHandle) -> bool,
	get_children: &'a mut dyn FnMut(NodeHandle) -> &'a [NodeHandle],
	get_cached: &mut dyn FnMut(NodeHandle) -> Option<Value>,
	make_node: &mut dyn FnMut(NodeHandle, &[Value]) -> Value,
) {
	let root_count = to_process.len();

	let mut values: Vec<Value> = Vec::new();
	let mut phase2 = false;
	while let Some(item) = to_process.pop() {
		if !is_valid_handle(item.clone()) {
			// Sentinel item. Go to phase 2.
			debug_assert!(!phase2);
			phase2 = true;
			continue;
		}

		let children = get_children(item.clone());
		if !phase2 {
			if let Some(value) = get_cached(item.clone()) {
				values.push(value);
				continue;
			}
			if !children.is_empty() {
				to_process.push(item);
				to_process.push(sentinel.clone());
				let x = to_process.len();
				to_process.extend_from_slice(children);
				to_process[x..].reverse();
				continue;
			}
		}

		phase2 = false;
		let children = &values[values.len() - children.len()..];
		let postorder_idx = make_node(item, children);
		values.truncate(values.len() - children.len());
		values.push(postorder_idx);
	}
	debug_assert!(!phase2);
	debug_assert!(values.len() == root_count);
}

//--------------------------------------------------------------------------------------------------
