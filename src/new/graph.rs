//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::borrow::Cow;
use std::collections::HashMap;
use std::hint::cold_path;
use std::rc::Rc;

use crate::util::index_vec::{IndexTrait, IndexVec};
use crate::{ErrExtra, ErrPack, define_index_type};

//--------------------------------------------------------------------------------------------------

pub trait Node {
	fn input_names(&self) -> &[&str];
	fn output_names(&self) -> &[&str];
}

#[derive(Clone, Copy)]
pub struct InputPortReference {
	pub node_index: NodeIndex,
	pub port_index: usize,
}

#[derive(Clone)]
pub struct NamedInputPortReference {
	pub node_index: NodeIndex,
	pub port_name: Cow<'static, str>,
}

pub trait CheckInputPort {
	fn check(&self, graph: &Graph) -> Result<InputPortReference, ErrPack<GraphConnectError>>;
}

impl CheckInputPort for InputPortReference {
	fn check(&self, graph: &Graph) -> Result<InputPortReference, ErrPack<GraphConnectError>> {
		if self.node_index.is_sentinel() {
			// Graph output port
			if self.port_index >= graph.outputs.len() {
				cold_path();
				return Err(ErrPack {
					code: GraphConnectError,
					extra: Some(Box::new(ErrExtra {
						message: format!("Graph has no output port {}", self.port_index).into(),
						nested: None,
					})),
				});
			}
			return Ok(*self);
		}
		let Some(node) = graph.nodes.get(self.node_index) else {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!("Node index {} not found", self.node_index.raw).into(),
					nested: None,
				})),
			});
		};
		if self.port_index >= node.inputs.len() {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!(
						"Node index {} has no input port {}",
						self.node_index.raw, self.port_index
					)
					.into(),
					nested: None,
				})),
			});
		}
		Ok(*self)
	}
}

impl CheckInputPort for NamedInputPortReference {
	fn check(&self, graph: &Graph) -> Result<InputPortReference, ErrPack<GraphConnectError>> {
		let Some(node) = graph.nodes.get(self.node_index) else {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!("Node index {} not found", self.node_index.raw).into(),
					nested: None,
				})),
			});
		};
		let Some(&port_index) = node.input_names.get(self.port_name.as_str()) else {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!(
						"Node index {} has no input port named {}",
						self.node_index.raw, self.port_name
					)
					.into(),
					nested: None,
				})),
			});
		};
		Ok(InputPortReference { node_index: self.node_index, port_index })
	}
}

#[derive(Clone, Copy)]
pub struct OutputPortReference {
	pub node_index: NodeIndex,
	pub port_index: usize,
}

#[derive(Clone)]
pub struct NamedOutputPortReference {
	pub node_index: NodeIndex,
	pub port_name: Cow<'static, str>,
}

pub trait CheckOutputPort {
	fn check(&self, graph: &Graph) -> Result<OutputPortReference, ErrPack<GraphConnectError>>;
}

impl CheckOutputPort for OutputPortReference {
	fn check(&self, graph: &Graph) -> Result<OutputPortReference, ErrPack<GraphConnectError>> {
		if self.node_index.is_sentinel() {
			// Graph input port
			if self.port_index >= graph.inputs.len() {
				cold_path();
				return Err(ErrPack {
					code: GraphConnectError,
					extra: Some(Box::new(ErrExtra {
						message: format!("Graph has no input port {}", self.port_index).into(),
						nested: None,
					})),
				});
			}
			return Ok(*self);
		}
		let Some(node) = graph.nodes.get(self.node_index) else {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!("Node index {} not found", self.node_index.raw).into(),
					nested: None,
				})),
			});
		};
		if self.port_index >= node.outputs.len() {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!(
						"Node index {} has no output port {}",
						self.node_index.raw, self.port_index
					)
					.into(),
					nested: None,
				})),
			});
		}
		Ok(*self)
	}
}

impl CheckOutputPort for NamedOutputPortReference {
	fn check(&self, graph: &Graph) -> Result<OutputPortReference, ErrPack<GraphConnectError>> {
		let Some(node) = graph.nodes.get(self.node_index) else {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!("Node index {} not found", self.node_index.raw).into(),
					nested: None,
				})),
			});
		};
		let Some(&port_index) = node.output_names.get(self.port_name.as_str()) else {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: format!(
						"Node index {} has no output port named {}",
						self.node_index.raw, self.port_name
					)
					.into(),
					nested: None,
				})),
			});
		};
		Ok(OutputPortReference { node_index: self.node_index, port_index })
	}
}

pub struct InputPortData {
	name: String,
	connection: Option<OutputPortReference>,
}

pub struct OutputPortData {
	name: String,
	connections: Vec<InputPortReference>,
}

struct NodeData {
	node: Rc<dyn Node>,
	node_name: String,
	input_names: HashMap<String, usize>,
	output_names: HashMap<String, usize>,
	inputs: Vec<InputPortData>,
	outputs: Vec<OutputPortData>,
}

define_index_type!(NodeIndex);
type NodeVec = IndexVec<NodeIndex, NodeData>;

pub struct NodeReference {
	node_index: NodeIndex,
	node: Rc<dyn Node>,
}

impl NodeReference {
	pub fn input(&self) -> InputPortReference {
		InputPortReference {
			node_index: self.node_index,
			port_index: 0,
		}
	}

	pub fn output(&self) -> OutputPortReference {
		OutputPortReference {
			node_index: self.node_index,
			port_index: 0,
		}
	}

	pub fn named_input<S: Into<Cow<'static, str>>>(
		&self,
		input_name: S,
	) -> NamedInputPortReference {
		NamedInputPortReference {
			node_index: self.node_index,
			port_name: input_name.into(),
		}
	}

	pub fn named_output<S: Into<Cow<'static, str>>>(
		&self,
		output_name: S,
	) -> NamedOutputPortReference {
		NamedOutputPortReference {
			node_index: self.node_index,
			port_name: output_name.into(),
		}
	}
}

#[derive(Clone, Copy, Debug)]
pub struct GraphConnectError;

pub struct Graph {
	nodes_by_name: HashMap<String, NodeIndex>,
	nodes: NodeVec,
	inputs: Vec<Option<OutputPortReference>>,
	outputs: Vec<Vec<InputPortReference>>,
}

impl Graph {
	pub fn new(inputs: &[Cow<'static, str>], outputs: &[Cow<'static, str>]) -> Self {
		Self {
			nodes_by_name: std::collections::HashMap::new(),
			nodes: NodeVec::new(),
			inputs: vec![None; inputs.len()],
			outputs: vec![Vec::new(); outputs.len()],
		}
	}

	pub fn add_node<S: Into<String>>(&mut self, name: S, node: Rc<dyn Node>) -> NodeReference {
		let name = name.into();
		let input_names = node.input_names();
		let output_names = node.output_names();
		let index = self.nodes.push(NodeData {
			node: node.clone(),
			node_name: name.clone(),
			input_names: input_names
				.iter()
				.enumerate()
				.map(|(i, &name)| (name.into(), i))
				.collect(),
			output_names: output_names
				.iter()
				.enumerate()
				.map(|(i, &name)| (name.into(), i))
				.collect(),
			inputs: input_names
				.iter()
				.map(|&name| InputPortData { name: name.into(), connection: None })
				.collect(),
			outputs: output_names
				.iter()
				.map(|&name| OutputPortData {
					name: name.into(),
					connections: Vec::new(),
				})
				.collect(),
		});
		self.nodes_by_name.insert(name, index);

		NodeReference { node_index: index, node }
	}

	#[allow(clippy::needless_pass_by_value)]
	pub fn connect(
		&mut self,
		from: impl CheckOutputPort,
		to: impl CheckInputPort,
	) -> Result<(), ErrPack<GraphConnectError>> {
		let from: OutputPortReference = from.check(self)?;
		let to: InputPortReference = to.check(self)?;

		if from.node_index == to.node_index {
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: "Cannot connect a node to itself".into(),
					nested: None,
				})),
			});
		}
		if from.node_index >= to.node_index {
			// TODO - we could allow this, but then we would need to check for cycles
			cold_path();
			return Err(ErrPack {
				code: GraphConnectError,
				extra: Some(Box::new(ErrExtra {
					message: "Cannot connect from a node with an index greater than or equal to the destination node".into(),
					nested: None,
				})),
			});
		}

		#[allow(clippy::indexing_slicing)]
		{
			if self.nodes[to.node_index].inputs[to.port_index].connection.is_some() {
				cold_path();
				return Err(ErrPack {
					code: GraphConnectError,
					extra: Some(Box::new(ErrExtra {
						message: "Input port is already connected".into(),
						nested: None,
					})),
				});
			}
			self.nodes[to.node_index].inputs[to.port_index].connection = Some(from);
			self.nodes[from.node_index].outputs[from.port_index].connections.push(to);
		}

		Ok(())
	}

	pub fn input(&self) -> OutputPortReference {
		OutputPortReference {
			node_index: NodeIndex::new_sentinel(),
			port_index: 0,
		}
	}

	pub fn output(&self) -> InputPortReference {
		InputPortReference {
			node_index: NodeIndex::new_sentinel(),
			port_index: 0,
		}
	}

	pub fn print_graphviz(&self) -> String {
		let mut s = String::new();
		let _ = self.__print_graphviz(&mut s);
		s
	}

	#[allow(clippy::indexing_slicing)]
	pub fn __print_graphviz<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
		writeln!(w, "digraph G {{")?;
		writeln!(w, "\trankdir=LR;")?;
		writeln!(w, "\tnewrank=true;")?;

		for i in self.nodes.indexes() {
			let node = &self.nodes[i];
			writeln!(w, "\tnode_{} [label=\"{}\"]", i.raw, &node.node_name)?;

			for inp_idx in 0..node.inputs.len() {
				if let Some(conn) = &node.inputs[inp_idx].connection {
					if conn.node_index.is_sentinel() {
						// Graph input port
						writeln!(
							w,
							"\tgraph_inp_{} -> node_{}:inp_{};",
							conn.port_index, i.raw, inp_idx
						)?;
					} else {
						writeln!(
							w,
							"\tnode_{}:out_{} -> node_{}:inp_{};",
							conn.node_index.raw, conn.port_index, i.raw, inp_idx
						)?;
					}
				}
			}
		}

		writeln!(w, "}}")?;
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
