/*
// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

//--------------------------------------------------------------------------------------------------

trait GraphNode {
	fn free(&mut self);

	fn backward(&mut self);
}

struct GraphNodeBase {
	rc_minus_one: usize,
	node: NonNull<dyn GraphNode>,
	inputs: *const [*mut GraphNodeBase],
	grad: Option<Value>,
}

impl GraphNodeBase {
	fn new_dangling() -> ManuallyDrop<Self> {
		let node = unsafe { &mut DANGLING_GRAPH_NODE as *mut dyn GraphNode };
		let node = NonNull::new(node).unwrap();
		ManuallyDrop::new(GraphNodeBase {
			rc_minus_one: 0,
			node,
			inputs: unsafe { DANGLING_GRAPH_NODE.inputs.as_slice() },
			grad: None,
		})
	}

	fn init(&mut self, node: NonNull<dyn GraphNode>, inputs: *const [*mut GraphNodeBase]) {
		debug_assert!(self.rc_minus_one == 0);
		self.node = node;
		self.inputs = inputs;
	}
}

impl RefCounted for GraphNodeBase {
	type Deref = dyn GraphNode;

	fn inc_rc(&mut self) {
		self.rc_minus_one += 1;
	}

	fn dec_rc(&mut self) {
		if self.rc_minus_one == 0 {
			unsafe { (*self.node.as_ptr()).free() }
		} else {
			self.rc_minus_one -= 1;
		}
	}

	fn get(&self) -> &Self::Deref {
		unsafe { &*self.node.as_ptr() }
	}
}

struct DanglingGraphNode {
	inputs: [*mut GraphNodeBase; 0],
}

impl GraphNode for DanglingGraphNode {
	fn free(&mut self) {
		panic!("DanglingGraphNode::free() called");
	}

	fn backward(&mut self) {
		panic!("DanglingGraphNode::backward() called");
	}
}

static mut DANGLING_GRAPH_NODE: DanglingGraphNode = DanglingGraphNode { inputs: [] };

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tensor {
	val: Value,
	grad_fn: Option<Rc<GraphNodeBase>>,
}

//--------------------------------------------------------------------------------------------------

pub trait Optimizer {
	fn add_param(&mut self, param: Value);
	fn zero_grad(&mut self);
	fn step(&mut self);
}

//--------------------------------------------------------------------------------------------------

struct SGDParam {
	grad: MutVal,
}

pub struct SGD {
	params: Vec<SGDParam>,
}

impl Optimizer for SGD {
	fn add_param(&mut self, param: Value) {
		self.params.push(SGDParam {
			grad: param.as_mut(),
		});
	}

	fn zero_grad(&mut self) {
		panic!("SGD::zero_grad() not implemented");
	}

	fn step(&mut self) {
		panic!("SGD::step() not implemented");
	}
}

//--------------------------------------------------------------------------------------------------

fn model(x: Input) {
	let lin1 = nn.Linear::new("lin1", 784, 128);
	let act = nn.ReLU::new();
	let lin2 = nn.Linear::new("lin2", 128, 10);
	return lin2(act(lin1(x)));
}


*/
