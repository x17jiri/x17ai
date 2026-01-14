//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

use crate::util::index_vec::{IndexTrait, IndexVec};
use crate::{ErrExtra, ErrPack, define_index_type};

//--------------------------------------------------------------------------------------------------

pub trait Node {
	fn n_inputs(&self) -> usize;
	fn n_outputs(&self) -> usize;

	/// If not found, returns usize::MAX
	fn input_by_name(&self, name: &str) -> usize;

	/// If not found, returns usize::MAX
	fn output_by_name(&self, name: &str) -> usize;
}

#[derive(Clone, Copy)]
pub struct InputPortReference {
	pub node: NodeIndex,
	pub port_index: usize,
}

#[derive(Clone, Copy)]
pub struct OutputPortReference {
	pub node: NodeIndex,
	pub port_index: usize,
}

struct NodeData {
	node: Rc<dyn Node>,
	inputs: Vec<Option<OutputPortReference>>,
	outputs: Vec<Vec<InputPortReference>>,
}

define_index_type!(NodeIndex);
type NodeVec = IndexVec<NodeIndex, NodeData>;

pub struct NodeReference {
	index: NodeIndex,
	node: Rc<dyn Node>,
}

impl NodeReference {
	pub fn input(&self) -> InputPortReference {
		InputPortReference { node: self.index, port_index: 0 }
	}

	pub fn output(&self) -> OutputPortReference {
		OutputPortReference { node: self.index, port_index: 0 }
	}

	pub fn named_input(&self, name: &str) -> InputPortReference {
		InputPortReference {
			node: self.index,
			port_index: self.node.input_by_name(name),
		}
	}

	pub fn named_output(&self, name: &str) -> OutputPortReference {
		OutputPortReference {
			node: self.index,
			port_index: self.node.output_by_name(name),
		}
	}
}

#[derive(Clone, Copy, Debug)]
pub struct GraphConnectError;

pub struct Graph {
	nodes_by_name: HashMap<Cow<'static, str>, NodeIndex>,
	nodes: NodeVec,
	inputs: Vec<Vec<InputPortReference>>,
	outputs: Vec<Option<OutputPortReference>>,
}

impl Graph {
	pub fn new(inputs: &[Cow<'static, str>], outputs: &[Cow<'static, str>]) -> Self {
		Self {
			nodes_by_name: std::collections::HashMap::new(),
			nodes: NodeVec::new(),
			inputs: vec![Vec::new(); outputs.len()],
			outputs: vec![None; inputs.len()],
		}
	}

	pub fn add_node<S: Into<Cow<'static, str>>>(
		&mut self,
		name: S,
		node: Rc<dyn Node>,
	) -> NodeReference {
		let index = self.nodes.push(NodeData {
			node: node.clone(),
			inputs: vec![None; node.n_inputs()],
			outputs: vec![Vec::new(); node.n_outputs()],
		});
		let name = name.into();
		self.nodes_by_name.insert(name, index);

		NodeReference { index, node }
	}

	pub fn connect(
		&mut self,
		from: OutputPortReference,
		to: InputPortReference,
	) -> Result<(), ErrPack<GraphConnectError>> {
		let Some(from_node) = self.nodes.get_mut(from.node) else {
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!("Source node {:?} not found", from.node).into(),
					nested: None,
				})),
			});
		};
		let to_node = self.nodes.get_mut(to.node).ok_or(ErrPack::new(GraphConnectError))?;

		if from.port_index >= from_node.outputs.len() || to.port_index >= to_node.inputs.len() {
			return Err(ErrPack::new(GraphConnectError));
		}

		to_node.inputs[to.port_index] = Some(from);
		from_node.outputs[from.port_index].push(to);

		Ok(())
	}

	pub fn input(&self) -> OutputPortReference {
		OutputPortReference {
			node: NodeIndex::new_sentinel(),
			port_index: 0,
		}
	}

	pub fn output(&self) -> InputPortReference {
		InputPortReference {
			node: NodeIndex::new_sentinel(),
			port_index: 0,
		}
	}
}

//--------------------------------------------------------------------------------------------------
