// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use x17ai::debug_2d;
use x17ai::nn::layers::{Layer, Linear, LossFunction, SoftmaxCrossEntropy};
use x17ai::nn::{EvalContext, ModelContext};
use x17ai::tensor::device::cpu::CPUDevice;
use x17ai::tensor::math::Savable;
use x17ai::tensor::{self, HasDType, Tensor};

/*
struct Attention {
	pub input_features: usize,
	pub heads: usize,
	pub qk_size: usize,
	pub v_size: usize,
	pub dtype: DType,

	pub k: Linear,
	pub q: Linear,
	pub v: Linear,
}

impl Attention {
	pub fn new(
		input_features: usize,
		heads: usize,
		qk_size: usize,
		v_size: usize,
		dtype: DType,
		alloc: &mut dyn Allocator,
	) -> Attention {
		let k = Linear::new(input_features, qk_size, dtype, alloc);
		let q = Linear::new(input_features, heads * qk_size, dtype, alloc);
		let v = Linear::new(input_features, heads * v_size, dtype, alloc);

		Attention {
			input_features,
			heads,
			qk_size,
			v_size,
			dtype,
			k,
			q,
			v,
		}
	}
}

impl Module for Attention {
	// input is of the form: [..., inputs, embeding]
	fn forward(&self, input: &Tensor) -> Tensor {
		// explanation of dimension names:
		// *: batch (can be any number of dimensions >= 0)
		// i: input sequence
		// h: head
		// q, k, v: key, query, value

		// TODO - use scopes so tensors are freed when not needed

		// input: [*, i, input_features]
		let seq_len = input.shape()[-2];

		// k: [*, i, k]
		// -> [*, 1, i, k]
		// -> [*, h, i, k]
		let k = self.k.forward(input);
		let k = k.reshape_last_n(2, &[1, seq_len, self.qk_size]);

		// q: [*, i, h * q]
		// -> [*, i, h, q]
		// -> [*, h, i, q]
		// -> [*, h, q, i]
		let q = self.q.forward(input);
		let q = q.reshape_last_n(1, &[self.heads, self.qk_size]);
		let q = q.transposed(-3, -2);
		let q = q.transposed(-2, -1);

		// v: [*, i, h * v]
		// -> [*, i, h, v]
		// -> [*, h, i, v]
		let v = self.v.forward(input);
		let v = v.reshape_last_n(1, &[self.heads, self.v_size]);
		let w_shape = v.shape().to_vec(); // [*, i, h, v]
		let v = v.transposed(-3, -2);

		// scores: [*, h, i, i]
		let scores = matmul(&k, &q);

		// w = reweighted v
		// w: [*, h, i, v]
		// -> [*, i, h, v]
		// -> [*, i, w = h * v]

		let w = v.new_tensor(&w_shape, v.dtype()); // [*, i, h, v]
		let w = w.transposed(-3, -2); // [*, h, i, v]

		matmul_(scores, v, w);
		let w = w.transposed(-3, -2); // [*, i, h, v]
		let w = w.reshape_last_n(2, &[self.heads * self.v_size]);

		w
	}
}

struct Transformer {
	pub attention: Attention,
	pub feed_forward: Linear,
}

impl Transformer {
	pub fn new(
		input_features: usize,
		heads: usize,
		qk_size: usize,
		v_size: usize,
		dtype: DType,
		alloc: &mut dyn Allocator,
	) -> Transformer {
		let attention = Attention::new(input_features, heads, qk_size, v_size, dtype, alloc);
		let feed_forward = Linear::new(heads * v_size, 2 * input_features, dtype, alloc);
		Transformer { attention, feed_forward }
	}
}

impl Module for Transformer {
	fn output_info(&self, input: &Tensor) -> (Rc<Shape>, DType) {
		(input.shape.clone(), input.dtype)
	}

	fn forward_(&self, input: &Tensor, output: &Tensor, _ctx: &Context) {
		let a = self.rms_norm.forward(input);
		let b = self.attention.forward(a);
		let c = self.feed_forward.forward(b);
		swiglu_(c, output);
	}
}
*/
fn main() {
	stderrlog::new().verbosity(10).init().unwrap();

	let dev = CPUDevice::new("CPU".to_string());
	let mut mctx = ModelContext::new(dev.clone());

	let mut model = Linear::new(3, 2, f32::dtype, &mut mctx);
	model.randomize();

	let mut loss = SoftmaxCrossEntropy::new(2);
	loss.randomize();

	let input = Tensor::new_empty_on(&[2, 3], f32::dtype, dev.clone());
	tensor::math::randn().save_to(&input);

	let expected = Tensor::new_debug_2d(dev.clone(), debug_2d![f32; [1.0, 0.0], [0.0, 1.0],]);

	let mut ectx_model = EvalContext::new(true);
	let output_logits = model.forward(input.clone(), &mut ectx_model);

	let mut ectx_loss = EvalContext::new(true);
	let output = loss.forward(output_logits.clone(), &mut ectx_loss);

	let loss_value = loss.loss(output.clone(), expected.clone());

	for (name, param) in model.named_params("model_params") {
		println!("{}: {}", name, param.borrow().value());
	}

	println!("input = {}", input);
	println!("output_logits = {}", output_logits);
	println!("output = {}", output);
	println!("expected = {}", expected);
	println!("loss_value = {}", loss_value);
	println!("--------------------------------------------------");

	for _ in 0..1000 {
		//		println!("Step {}", i);
		//		println!();

		mctx.zero_grad();

		let d_logits = loss.backward_start(output.clone(), expected.clone(), &mut ectx_loss);
		model.backward_finish(d_logits.clone(), &mut ectx_model);

		mctx.step();

		let output_logits = model.forward(input.clone(), &mut ectx_model);
		let output = loss.forward(output_logits.clone(), &mut ectx_loss);
		let loss_value = loss.loss(output.clone(), expected.clone());

		//		println!("output_logits = {}", output_logits);
		//		println!("output = {}", output);
		//		println!("loss_value = {}", loss_value);
		println!("{}", loss_value);
		//println!("--------------------------------------------------");
	}
	for (name, param) in model.named_params("model_params") {
		println!("{}: {}", name, param.borrow().value());
	}
	println!("input = {}", input);
	println!("expected = {}", expected);
}
