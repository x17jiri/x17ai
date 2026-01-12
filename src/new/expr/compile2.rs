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
use thin_vec::ThinVec;

use crate::define_index_type32;
use crate::new::expr::{CanBeBatched, split_shape};
use crate::tensor::device::dtype::common_dtype;
use crate::tensor::{DType, HasDType};
use crate::util::bitmap::IndexBitmap;
use crate::util::index_vec::{IndexTrait, IndexVec, UntypedIndex32};
use crate::util::{LossyFrom, ToBoxedSlice};

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryKind {
	Identity,
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
			NodeKind::Cast
				| NodeKind::Reshape
				| NodeKind::Unary(UnaryKind::Identity | UnaryKind::Neg)
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

pub struct TensorPort {
	pub name: Cow<'static, str>,
	pub dtype: DType,
	pub input_node: NodeIndex32,
	pub output_node: NodeIndex32,
}

impl TensorPort {
	pub fn is_input(&self) -> bool {
		!self.input_node.is_sentinel()
	}

	pub fn is_output(&self) -> bool {
		!self.output_node.is_sentinel()
	}
}

pub struct ScalarPort {
	pub name: Cow<'static, str>,
	pub input_node: NodeIndex32,
}

pub struct ConstRef {
	pub name: Cow<'static, str>,
	pub value: f64,
	pub input_node: NodeIndex32,
}

define_index_type32!(TensorRefIndex32);
type TensorVec = IndexVec<TensorRefIndex32, TensorPort>;

define_index_type32!(ScalarRefIndex32);
type ScalarVec = IndexVec<ScalarRefIndex32, ScalarPort>;

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
	scalar_map: HashMap<Cow<'static, str>, ScalarRefIndex32>,
	scalar_vec: ScalarVec,
	tensor_map: HashMap<Cow<'static, str>, TensorRefIndex32>,
	tensor_vec: TensorVec,
	frags_postorder: FragVec,
	node_cache: HashMap<String, NodeIndex32>,
	err_log: ErrorLog,
}

pub struct KernelBuilder {
	pub data: RefCell<KernelBuilderData>,
}

#[derive(Clone, Copy)]
pub struct Expr<'a> {
	kernel_builder: &'a KernelBuilder,
	node_index: NodeIndex32,
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

	pub fn new_const<'a, S: Into<Cow<'static, str>>>(&'a self, name: S, value: f64) -> Expr<'a> {
		let mut data = self.data.borrow_mut();
		let node_idx = data.nodes_postorder.next_index();
		let const_idx = data.const_vec.push(ConstRef {
			name: name.into(),
			value,
			input_node: node_idx,
		});
		let real_node_index = data.nodes_postorder.push(KernelBuilderData::new_nullary_node(
			None,
			Rc::from([]),
			false,
			NodeKind::Const,
			const_idx.to_untyped(),
			XIndexState::ConstRef,
		));
		debug_assert!(node_idx == real_node_index);
		Expr {
			kernel_builder: self,
			node_index: node_idx,
		}
	}

	pub fn new_tensor_input<'a, S: Into<Cow<'static, str>>>(
		&'a self,
		name: S,
		dtype: DType,
		shape: &[usize],
		can_be_batched: CanBeBatched,
	) -> Expr<'a> {
		let mut data = self.data.borrow_mut();
		let KernelBuilderData {
			nodes_postorder,
			tensor_map,
			tensor_vec,
			err_log,
			..
		} = &mut *data;
		let node_index = nodes_postorder.next_index();
		let tensor_idx = tensor_vec.next_index();
		let name = name.into();
		match tensor_map.entry(name.clone()) {
			hash_map::Entry::Occupied(entry) => {
				let tensor_idx = *entry.get();
				let existing_node = tensor_vec[tensor_idx].input_node;
				err_log.log_error(existing_node, format!("Tensor port '{name}' redefined"));
				return Expr {
					kernel_builder: self,
					node_index: existing_node,
				};
			},
			hash_map::Entry::Vacant(entry) => {
				entry.insert(tensor_idx);
			},
		}
		let real_tensor_idx = tensor_vec.push(TensorPort {
			name,
			dtype,
			input_node: node_index,
			output_node: NodeIndex32::new_sentinel(),
		});
		debug_assert!(tensor_idx == real_tensor_idx);
		let real_node_index = nodes_postorder.push(KernelBuilderData::new_nullary_node(
			Some(dtype),
			Rc::from(shape),
			can_be_batched != CanBeBatched::No,
			NodeKind::Input(InputKind::Tensor),
			real_tensor_idx.to_untyped(),
			XIndexState::TensorRef,
		));
		debug_assert!(node_index == real_node_index);
		Expr { kernel_builder: self, node_index }
	}

	pub fn new_scalar_input<'a, S: Into<Cow<'static, str>>>(&'a self, name: S) -> Expr<'a> {
		let mut data = self.data.borrow_mut();
		let KernelBuilderData {
			nodes_postorder,
			scalar_map,
			scalar_vec,
			err_log,
			..
		} = &mut *data;
		let node_index = nodes_postorder.next_index();
		let scalar_idx = scalar_vec.next_index();
		let name = name.into();
		match scalar_map.entry(name.clone()) {
			hash_map::Entry::Occupied(entry) => {
				let scalar_idx = *entry.get();
				let existing_node = scalar_vec[scalar_idx].input_node;
				err_log.log_error(existing_node, format!("Scalar port '{name}' redefined"));
				return Expr {
					kernel_builder: self,
					node_index: existing_node,
				};
			},
			hash_map::Entry::Vacant(entry) => {
				entry.insert(scalar_idx);
			},
		}
		let real_scalar_idx = scalar_vec.push(ScalarPort { name, input_node: node_index });
		debug_assert!(scalar_idx == real_scalar_idx);
		let real_node_index = nodes_postorder.push(KernelBuilderData::new_nullary_node(
			None,
			Rc::from([]),
			false,
			NodeKind::Input(InputKind::Tensor),
			real_scalar_idx.to_untyped(),
			XIndexState::TensorRef,
		));
		debug_assert!(node_index == real_node_index);
		Expr { kernel_builder: self, node_index }
	}
}

impl<'a> Expr<'a> {
	fn cache_lookup(
		self,
		node_cache: &mut HashMap<String, NodeIndex32>,
		cache_key: String,
		new_node_index: NodeIndex32,
	) -> Option<Self> {
		if let Err(existing_node) = node_cache.try_insert(cache_key, new_node_index) {
			Some(Expr {
				kernel_builder: self.kernel_builder,
				node_index: *existing_node.entry.get(),
			})
		} else {
			None
		}
	}

	pub fn dtype(self) -> Option<DType> {
		let data = self.kernel_builder.data.borrow();
		let KernelBuilderData { nodes_postorder, .. } = &*data;
		nodes_postorder[self.node_index].dtype
	}

	pub fn get_dtype_or_log_error(self) -> DType {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, .. } = &mut *data;
		if let Some(my_dtype) = nodes_postorder[self.node_index].dtype {
			my_dtype
		} else {
			cold_path();
			err_log.log_error(self.node_index, format!("node has unknown dtype"));
			f64::dtype
		}
	}

	pub fn output<S: Into<Cow<'static, str>>>(self, port_name: S) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData {
			nodes_postorder,
			tensor_map,
			tensor_vec,
			err_log,
			..
		} = &mut *data;
		let node_index = self.node_index;
		let node = &nodes_postorder[node_index];
		if node.is_input() {
			cold_path();
			err_log.log_error(
				node_index,
				format!("cannot directly capture an input. Use `Identity` node first."),
			);
		}
		let dtype = if let Some(dt) = node.dtype {
			dt
		} else {
			cold_path();
			err_log.log_error(self.node_index, format!("captured value has unknown dtype"));
			f64::dtype
		};
		let tensor_idx;
		let port_name = port_name.into();
		match tensor_map.entry(port_name.clone()) {
			hash_map::Entry::Vacant(entry) => {
				tensor_idx = tensor_vec.push(TensorPort {
					name: port_name,
					dtype,
					input_node: NodeIndex32::new_sentinel(),
					output_node: node_index,
				});
				entry.insert(tensor_idx);
			},
			hash_map::Entry::Occupied(entry) => {
				tensor_idx = *entry.get();
				let existing_node = tensor_vec[tensor_idx].output_node;
				if !existing_node.is_sentinel() && existing_node != node_index {
					err_log.log_error(
						node_index,
						format!("Tensor port '{port_name}' written multiple times"),
					);
				}
				tensor_vec[tensor_idx].output_node = node_index;
			},
		}
		nodes_postorder[node_index].capture.push(tensor_idx);
		self
	}

	pub fn label<S: Into<Cow<'static, str>>>(self, label: S) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, .. } = &mut *data;
		let child = &mut nodes_postorder[self.node_index];
		child.labels.push(label.into());
		self
	}

	pub fn reshape(self, new_shape: &[usize]) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, node_cache, .. } = &mut *data;

		let node_index = nodes_postorder.next_index();
		let child_idx = self.node_index;

		let cache_key = format!("reshape:{:?}:{}", new_shape, child_idx.raw);
		if let Some(existing_expr) = self.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let mut log_error = |msg: String| err_log.log_error(node_index, msg);

		let child_node = &nodes_postorder[child_idx];
		let old_shape = child_node.shape();
		if old_shape == new_shape {
			return self;
		}

		let old_elems = old_shape.iter().product::<usize>();
		let new_elems = new_shape.iter().product::<usize>();
		if old_elems != new_elems {
			log_error(format!(
				"Reshape: element count mismatch (input has {old_elems} elements; replacement has {new_elems})",
			));
		}

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_unary_node(
			child_node.dtype,
			child_node.shape.clone(),
			child_node.can_be_batched,
			NodeKind::Reshape,
			child_idx,
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: self.kernel_builder,
			node_index,
		}
	}

	pub fn cast(self, dtype: DType) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, node_cache, .. } = &mut *data;

		let node_index = nodes_postorder.next_index();
		let child_idx = self.node_index;

		let cache_key = format!("cast:{dtype}:{:?}", child_idx.raw);
		if let Some(existing_expr) = self.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let child_node = &nodes_postorder[child_idx];
		if child_node.dtype == Some(dtype) {
			return self;
		}

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_unary_node(
			Some(dtype),
			child_node.shape.clone(),
			child_node.can_be_batched,
			NodeKind::Cast,
			child_idx,
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: self.kernel_builder,
			node_index,
		}
	}

	pub fn __unary(self, unary_kind: UnaryKind) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, node_cache, .. } = &mut *data;

		let node_index = nodes_postorder.next_index();
		let child_idx = self.node_index;

		let cache_key = format!("unary:{:?}:{:?}", unary_kind, child_idx.raw);
		if let Some(existing_expr) = self.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let child_node = &nodes_postorder[child_idx];

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_unary_node(
			child_node.dtype,
			child_node.shape.clone(),
			child_node.can_be_batched,
			NodeKind::Unary(unary_kind),
			child_idx,
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: self.kernel_builder,
			node_index,
		}
	}

	pub fn identity(self) -> Self {
		self.__unary(UnaryKind::Identity)
	}

	pub fn exp(self) -> Self {
		self.__unary(UnaryKind::Exp)
	}

	pub fn ln(self) -> Self {
		self.__unary(UnaryKind::Ln)
	}

	pub fn abs(self) -> Self {
		self.__unary(UnaryKind::Abs)
	}

	pub fn sqrt(self) -> Self {
		self.__unary(UnaryKind::Sqrt)
	}

	pub fn recip(self) -> Self {
		self.__unary(UnaryKind::Recip)
	}

	pub fn __reduction(self, reduction_kind: ReductionKind) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, node_cache, .. } = &mut *data;

		let node_index = nodes_postorder.next_index();
		let child_idx = self.node_index;

		let cache_key = format!("reduction:{:?}:{:?}", reduction_kind, child_idx.raw);
		if let Some(existing_expr) = self.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let mut log_error = |msg: String| err_log.log_error(node_index, msg);

		let child_node = &nodes_postorder[child_idx];

		let mut shape = child_node.shape.clone();
		if let Some(last_dim) = Rc::make_mut(&mut shape).last_mut() {
			*last_dim = 1;
		} else {
			cold_path();
			log_error(format!("missing reduce dimension"));
			let shape_slice: &[usize] = &[1];
			shape = Rc::from(shape_slice);
		}

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_unary_node(
			child_node.dtype,
			shape,
			child_node.can_be_batched,
			NodeKind::Reduction(reduction_kind),
			child_idx,
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: self.kernel_builder,
			node_index,
		}
	}

	pub fn sum(self) -> Self {
		self.__reduction(ReductionKind::Sum)
	}

	pub fn max(self) -> Self {
		self.__reduction(ReductionKind::Max)
	}

	pub fn __select(self, select_kind: SelectKind) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, node_cache, .. } = &mut *data;

		let node_index = nodes_postorder.next_index();
		let child_idx = self.node_index;

		let cache_key = format!("select:{:?}:{:?}", select_kind, child_idx.raw);
		if let Some(existing_expr) = self.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let mut log_error = |msg: String| err_log.log_error(node_index, msg);

		let child_node = &nodes_postorder[child_idx];

		let mut shape = child_node.shape.clone();
		if let Some(last_dim) = Rc::make_mut(&mut shape).last_mut() {
			if *last_dim % 2 != 0 {
				cold_path();
				log_error(format!("select dimension not even"));
			}
			*last_dim /= 2;
		} else {
			cold_path();
			log_error(format!("missing select dimension"));
			let shape_slice: &[usize] = &[0];
			shape = Rc::from(shape_slice);
		}

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_unary_node(
			child_node.dtype,
			shape,
			child_node.can_be_batched,
			NodeKind::Select(select_kind),
			child_idx,
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: self.kernel_builder,
			node_index,
		}
	}

	pub fn select_even(self) -> Self {
		self.__select(SelectKind::Even)
	}

	pub fn select_odd(self) -> Self {
		self.__select(SelectKind::Odd)
	}

	pub fn __binary(self, binary_kind: BinaryKind, rhs: Self) -> Self {
		// TODO - rhs may be from different kernel builder
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, node_cache, .. } = &mut *data;

		let node_index = nodes_postorder.next_index();
		let a_idx = self.node_index;
		let b_idx = rhs.node_index;

		let (a_idx, b_idx) = if binary_kind.is_commutative() && a_idx > b_idx {
			(b_idx, a_idx)
		} else {
			(a_idx, b_idx)
		};

		let cache_key = format!("binary:{:?}:{:?}:{:?}", binary_kind, a_idx.raw, b_idx.raw);
		if let Some(existing_expr) = self.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let mut log_error = |msg: String| err_log.log_error(node_index, msg);

		let a_node = &nodes_postorder[a_idx];
		let b_node = &nodes_postorder[b_idx];

		let dtype = same_dtype(a_node.dtype, b_node.dtype, &mut log_error);
		let (shape, is_broadcasted) =
			broadcast_shapes(a_node.shape(), b_node.shape(), &mut log_error);

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_binary_node(
			dtype,
			Rc::from(&shape[..]),
			a_node.can_be_batched || b_node.can_be_batched,
			NodeKind::Binary(binary_kind),
			[a_idx, b_idx],
			is_broadcasted,
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: self.kernel_builder,
			node_index,
		}
	}

	pub fn mat_times_col(self, col: Self) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, node_cache, .. } = &mut *data;

		let mat = self;
		let node_index = nodes_postorder.next_index();
		let m_idx = mat.node_index;
		let c_idx = col.node_index;

		let cache_key = format!("mat_times_col:{:?}:{:?}", m_idx.raw, c_idx.raw);
		if let Some(existing_expr) = mat.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let mut log_error = |msg: String| err_log.log_error(node_index, msg);

		let m_node = &nodes_postorder[m_idx];
		let c_node = &nodes_postorder[c_idx];

		let m_shape = m_node.shape();
		let c_shape = c_node.shape();
		if m_shape.len() < 2 || c_shape.len() < 1 {
			cold_path();
			log_error(format!("matmul: not enough dimensions"));
		}
		let (m_rest, [m_row, m_col]) = split_shape::<2>(m_shape);
		let (c_rest, [c_len]) = split_shape::<1>(c_shape);
		if m_col != c_len {
			cold_path();
			log_error(format!("matmul: shape mismatch"));
		}
		let mut shape = Rc::<[usize]>::from(c_shape);
		*Rc::make_mut(&mut shape).last_mut().unwrap() = m_row;

		if !m_rest.is_empty() || m_node.can_be_batched {
			cold_path();
			log_error("mat times col: mat cannot be batched".into());
		}
		let m_broadcasted = !c_rest.is_empty();

		let dtype = same_dtype(m_node.dtype, c_node.dtype, &mut log_error);

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_binary_node(
			dtype,
			Rc::from(&shape[..]),
			m_node.can_be_batched || c_node.can_be_batched,
			NodeKind::MatMul(MatMulKind::MatTimesCol),
			[m_idx, c_idx],
			[m_broadcasted, false],
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: mat.kernel_builder,
			node_index,
		}
	}

	pub fn col_times_row_acc(self, row: Self) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, node_cache, .. } = &mut *data;

		let col = self;
		let node_index = nodes_postorder.next_index();
		let c_idx = col.node_index;
		let r_idx = row.node_index;

		let cache_key = format!("col_times_row_acc:{:?}:{:?}", c_idx.raw, r_idx.raw);
		if let Some(existing_expr) = col.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let mut log_error = |msg: String| err_log.log_error(node_index, msg);

		let c_node = &nodes_postorder[c_idx];
		let r_node = &nodes_postorder[r_idx];

		let c_shape = c_node.shape();
		let r_shape = r_node.shape();
		if c_shape.len() < 1 || r_shape.len() < 1 {
			cold_path();
			log_error(format!("outer product: not enough dimensions"));
		}
		let (c_rest, [c_len]) = split_shape::<1>(c_shape);
		let (r_rest, [r_len]) = split_shape::<1>(r_shape);

		let (mut shape, is_broadcasted) = broadcast_shapes(c_rest, r_rest, &mut log_error);
		shape.push(c_len);
		shape.push(r_len);
		if is_broadcasted[0] || is_broadcasted[1] || c_node.can_be_batched != r_node.can_be_batched
		{
			cold_path();
			log_error(format!("outer product inputs cannot be broadcasted"));
		}

		let dtype = same_dtype(c_node.dtype, r_node.dtype, &mut log_error);

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_binary_node(
			dtype,
			Rc::from(&shape[..]),
			false, // we sum over the batch dimension
			NodeKind::MatMul(MatMulKind::ColTimesRowAcc),
			[c_idx, r_idx],
			is_broadcasted,
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: col.kernel_builder,
			node_index,
		}
	}

	pub fn even_odd(self, odd: Self) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, node_cache, .. } = &mut *data;

		let node_index = nodes_postorder.next_index();
		let a_idx = self.node_index;
		let b_idx = odd.node_index;

		let cache_key = format!("even_odd:{:?}:{:?}", a_idx.raw, b_idx.raw);
		if let Some(existing_expr) = self.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let mut log_error = |msg: String| err_log.log_error(node_index, msg);

		let a_node = &nodes_postorder[a_idx];
		let b_node = &nodes_postorder[b_idx];

		let (mut shape, is_broadcasted) =
			broadcast_shapes(a_node.shape(), b_node.shape(), &mut log_error);
		if let Some(last_dim) = shape.last_mut() {
			*last_dim *= 2;
		} else {
			shape.push(2);
		}
		if is_broadcasted[0] || is_broadcasted[1] || a_node.can_be_batched != b_node.can_be_batched
		{
			cold_path();
			log_error(format!("even_odd cannot broadcast inputs"));
		}

		let dtype = same_dtype(a_node.dtype, b_node.dtype, &mut log_error);

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_binary_node(
			dtype,
			Rc::from(&shape[..]),
			a_node.can_be_batched || b_node.can_be_batched,
			NodeKind::EvenOdd,
			[a_idx, b_idx],
			[false, false],
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: self.kernel_builder,
			node_index,
		}
	}

	pub fn attention(self, kv: Self) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, err_log, node_cache, .. } = &mut *data;

		let node_index = nodes_postorder.next_index();
		let q_idx = self.node_index;
		let kv_idx = kv.node_index;

		let cache_key = format!("attention:{:?}:{:?}", q_idx.raw, kv_idx.raw);
		if let Some(existing_expr) = self.cache_lookup(node_cache, cache_key, node_index) {
			return existing_expr;
		}

		let mut log_error = |msg: String| err_log.log_error(node_index, msg);

		let q_node = &nodes_postorder[q_idx];
		let kv_node = &nodes_postorder[kv_idx];

		let q_shape = q_node.shape();
		let kv_shape = kv_node.shape();
		if q_shape.len() < 3 || kv_shape.len() < 3 {
			cold_path();
			log_error(format!("attention: not enough dimensions"));
		}
		let (q_rest, [q1, q2, q3]) = split_shape::<3>(q_shape);
		let (kv_rest, [kv1, kv2, kv3]) = split_shape::<3>(kv_shape);
		if kv1 != 1 || q2 != kv2 || q3 >= kv3 {
			cold_path();
			log_error(format!("attention: shape mismatch"));
		}
		let (mut shape, is_broadcasted) = broadcast_shapes(q_rest, kv_rest, &mut log_error);
		shape.push(q1);
		shape.push(q2);
		shape.push(kv3.saturating_sub(q3));
		if is_broadcasted[0] || is_broadcasted[1] || q_node.can_be_batched != kv_node.can_be_batched
		{
			cold_path();
			log_error(format!("attention inputs cannot be broadcasted"));
		}

		let dtype = same_dtype(q_node.dtype, kv_node.dtype, &mut log_error);

		let real_node_index = nodes_postorder.push(KernelBuilderData::new_binary_node(
			dtype,
			Rc::from(&shape[..]),
			q_node.can_be_batched || kv_node.can_be_batched,
			NodeKind::Attention,
			[q_idx, kv_idx],
			[false, false],
		));
		debug_assert!(node_index == real_node_index);
		Expr {
			kernel_builder: self.kernel_builder,
			node_index,
		}
	}

	pub fn sum_to_mean(self) -> Self {
		let mut data = self.kernel_builder.data.borrow_mut();
		let KernelBuilderData { nodes_postorder, .. } = &mut *data;

		let node = &nodes_postorder[self.node_index];
		let shape = node.shape();
		let last_dim = shape.last().copied().unwrap_or(1);
		let c = 1.0 / f64::lossy_from(last_dim);

		std::mem::drop(data);

		self.kernel_builder.new_const(format!("1.0 / {last_dim}"), c)
	}

	pub fn mean(self) -> Self {
		self.sum() * self.sum_to_mean()
	}
}

impl<'a> std::ops::Add for Expr<'a> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		self.__binary(BinaryKind::Add, rhs)
	}
}

impl<'a> std::ops::Sub for Expr<'a> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		self.__binary(BinaryKind::Sub, rhs)
	}
}

impl<'a> std::ops::Mul for Expr<'a> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		self.__binary(BinaryKind::Mul, rhs)
	}
}

impl<'a> std::ops::Neg for Expr<'a> {
	type Output = Self;

	fn neg(self) -> Self {
		self.__unary(UnaryKind::Neg)
	}
}

impl KernelBuilderData {
	pub fn analyze(&mut self) -> Result<(), ()> {
		self.init_parents();
		self.remove_dead_code();
		self.find_races();
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

	pub fn new_unary_node(
		dtype: Option<DType>,
		shape: Rc<[usize]>,
		can_be_batched: bool,
		node_kind: NodeKind,
		child_idx: NodeIndex32,
	) -> Node {
		Node {
			node_kind,
			dtype,
			shape,
			can_be_batched,
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
		dtype: Option<DType>,
		shape: Rc<[usize]>,
		can_be_batched: bool,
		node_kind: NodeKind,
		children: [NodeIndex32; 2],
		is_broadcasted: [bool; 2],
	) -> Node {
		Node {
			node_kind,
			dtype,
			shape,
			can_be_batched,
			parents: SmallVec::new(),
			children,
			children_broadcast: is_broadcasted,
			capture: ThinVec::new(),
			is_dead: false,
			is_trivial_head: false,
			x_index: UntypedIndex32::new_sentinel(),
			x_index_state: XIndexState::Uninitialized,
			labels: ThinVec::new(),
		}
	}

	fn init_parents(&mut self) {
		for i in self.nodes_postorder.indexes().rev() {
			let children = self.nodes_postorder[i].children;
			for child in children {
				if !self.nodes_postorder.is_valid(child) {
					break;
				}
				self.nodes_postorder[child].parents.push(i);
			}
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
	fn find_races(&mut self) {
		let kills = NodeIndex32::from_raw(self.nodes_postorder.len());
		let rows = self.nodes_postorder.len() + 1;
		let cols = self.tensor_vec.len();
		let mut bitmap = IndexBitmap::<NodeIndex32>::new();
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
							let name = &self.tensor_vec[TensorRefIndex32::from_raw(c)].name;
							self.err_log.log_error(i, format!("Ambiguous use of tensor {name}. Not clear whether to use the version before or after write."));
						}
					}
				}
				for &tensor_index in &me.capture {
					let was_killed = bitmap.set_bit(kills, tensor_index.to_raw());
					if was_killed {
						cold_path();
						let name = &self.tensor_vec[tensor_index].name;
						self.err_log.log_error(i, format!("Double write to tensor {name}."));
					}
				}
				bitmap.and_not(i, i, kills);
			}
		}
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
					let (c_dims, [c_top]) = split_shape::<1>(child_shape);
					let (p_dims, [p_top]) = split_shape::<1>(parent_shape);
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
				let name = &self.tensor_vec[node.tensor_index()].name;
				format!("<b>Tensor</b><br/><font color='#800080'><b>{name}</b></font>")
			},
			NodeKind::Input(InputKind::Scalar) => {
				let name = &self.scalar_vec[node.scalar_index()].name;
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
				UnaryKind::Identity => "<b>Identity</b>".to_string(),
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
				MatMulKind::MatTimesCol => "<b>MAT * col</b>".to_string(),
				MatMulKind::ColTimesRowAcc => "<b>ACC(col * row) → MAT</b>".to_string(),
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
		let _ = self.__print_graphviz(&mut s);
		s
	}

	#[allow(clippy::too_many_lines)]
	pub fn __print_graphviz<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		writeln!(w, "\trankdir=BT;")?;
		writeln!(w, "\tnewrank=true;")?;
		for i in self.nodes_postorder.indexes() {
			let node = &self.nodes_postorder[i];
			let node_id = format!("node_{}", i.raw);
			let label = self.graphviz_node_label(node);
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
					let name = &self.tensor_vec[capt_idx].name;
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
			writeln!(w, "\t{{ rank=same; {node_id}; node_{}; }};", fragment.head.raw)?;
			/*writeln!(
				w,
				"\tnode_{} -> {node_id} [style=dashed, color=\"#808080\", constraint=false];",
				fragment.head.raw
			)?;*/
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

fn same_dtype(
	a_dtype: Option<DType>,
	b_dtype: Option<DType>,
	log_error: impl FnOnce(String),
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
				log_error(format!("dtype mismatch: {} vs {}", a_dt, b_dt));
				Some(common_dtype(a_dt, b_dt))
			}
		},
	}
}

//--------------------------------------------------------------------------------------------------

pub fn broadcast_shapes(
	a: &[usize],
	b: &[usize],
	mut log_error: impl FnMut(String),
) -> (Vec<usize>, [bool; 2]) {
	let mut is_broadcasted = [false, false];
	let mut result = Vec::new();
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
			log_error(format!("broadcast dimension mismatch: {:?} vs {:?}", dim_a, dim_b));
			dim_a.max(dim_b)
		};
		result.push(dim);
	}
	(result, is_broadcasted)
}

//--------------------------------------------------------------------------------------------------
