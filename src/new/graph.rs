//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

use crate::define_index_type;
use crate::util::index_vec::IndexVec;

//--------------------------------------------------------------------------------------------------

pub trait Node {
	fn n_inputs(&self) -> usize;
	fn n_outputs(&self) -> usize;
	fn input_by_name(&self, name: &str) -> Option<usize>;
	fn output_by_name(&self, name: &str) -> Option<usize>;
}

#[derive(Clone, Copy)]
pub struct InputLink {
	pub src_node: NodeIndex,
	pub output_index: usize,
}

#[derive(Clone, Copy)]
pub struct OutputLink {
	pub dst_node: NodeIndex,
	pub input_index: usize,
}

struct NodeData {
	node: Rc<dyn Node>,
	inputs: Vec<Option<InputLink>>,
	outputs: Vec<Vec<OutputLink>>,
}

define_index_type!(NodeIndex);
type NodeVec = IndexVec<NodeIndex, NodeData>;

pub struct Graph {
	nodes_by_name: HashMap<Cow<'static, str>, NodeIndex>,
	nodes: NodeVec,
}

impl Default for Graph {
	fn default() -> Self {
		Self::new()
	}
}

impl Graph {
	pub fn new() -> Self {
		Self {
			nodes_by_name: std::collections::HashMap::new(),
			nodes: NodeVec::new(),
		}
	}

	pub fn add_node<S: Into<Cow<'static, str>>>(
		&mut self,
		name: S,
		node: Rc<dyn Node>,
	) -> NodeIndex {
		let name = name.into();
		if let Some(&index) = self.nodes_by_name.get(&name) {
			return index;
		}

		let index = self.nodes.push(NodeData {
			inputs: vec![None; node.n_inputs()],
			outputs: vec![Vec::new(); node.n_outputs()],
			node,
		});
		self.nodes_by_name.insert(name, index);
		index
	}

	pub fn connect(
		&mut self,
		(src_node, src_port): (&str, &str),
		(dst_node, dst_port): (&str, &str),
	) -> Result<(), &'static str> {
		/*let src_output_index = self.nodes_postorder.raw[src_node.0]
			.node
			.output_by_name(src_port)
			.ok_or("Invalid source port name")?;
		let dst_input_index = self.nodes_postorder.raw[dst_node.0]
			.node
			.input_by_name(dst_port)
			.ok_or("Invalid destination port name")?;

		self.nodes_postorder[dst_node].inputs[dst_input_index] =
			Some(InputLink { src_node, output_index: src_output_index });
		self.nodes_postorder[src_node].outputs[src_output_index]
			.push(OutputLink { dst_node, input_index: dst_input_index });*/
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
