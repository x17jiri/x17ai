// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use x17ai::debug_2d;
use x17ai::nn::layers::{Layer, Linear, LossFunction, SoftmaxCrossEntropy};
use x17ai::nn::{EvalContext, ModelContext};
use x17ai::tensor::device::cpu::CPUDevice;
use x17ai::tensor::math::Savable;
use x17ai::tensor::{self, HasDType, Tensor};

use std::io::Write;

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
	/*
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

		/*	let t = Tensor::new_empty_on(&[5, 7], f32::dtype, dev.clone());
		let _ = t.read_from_file("tensor.bin");
		println!("t = {}", t);*/
	*/
	//	let Q = Tensor::new_empty_on(&[10, 8, 64], f32::dtype, dev.clone());
	let Q = Tensor::new_empty_on(&[10, 4, 5], f32::dtype, dev.clone());
	let K = Q.new_empty_like();
	let V = Q.new_empty_like();
	let expected_out = Q.new_empty_like();
	let out = Q.new_empty_like();

	Q.read_from_file("Q.bin").unwrap();
	K.read_from_file("K.bin").unwrap();
	V.read_from_file("V.bin").unwrap();
	expected_out.read_from_file("expected_out.bin").unwrap();

	tensor::math::attention(&Q, &K, &V).save_to(&out);
	let mut out_txt = std::fs::File::create("out.txt").unwrap();
	writeln!(out_txt, "out = {}", out).unwrap();

	/*
	Q = tensor([[[-0.5728,  0.2759, -1.2481,  1.2678,  0.0717],
			 [ 0.7365, -0.1365, -1.0994, -0.5166,  1.5378],
			 [ 0.2821,  0.2301,  0.1800, -0.0684,  0.8570],
			 [-0.1845, -0.5259, -0.9884,  0.3163, -0.5252]],

			[[ 1.2729, -1.8302,  0.1313, -0.0683, -0.5589],
			 [ 0.0100, -0.3041,  0.2271,  1.0116,  1.0592],
			 [-0.3654,  2.6270,  0.8530,  1.2593,  0.0454],
			 [ 0.5415,  1.2081,  1.5722,  1.3396,  0.2734]],

			[[-0.2628, -0.0934,  1.1173,  0.2901,  0.3048],
			 [ 0.9385, -1.4884,  0.8204, -1.7372,  2.7728],
			 [ 0.0775, -0.7416, -0.2225, -0.0360, -1.8122],
			 [ 1.0723, -0.7707, -0.8560, -1.4648,  1.2018]]])
	K = tensor([[[-0.4923,  0.7813,  0.8644, -1.5706,  0.7498],
			 [-1.4921, -3.1150, -0.9135, -0.7144,  1.2236],
			 [ 2.0291,  0.0276,  1.0505, -1.9432, -1.3391],
			 [-0.4148, -1.1546, -0.3438,  0.9069, -0.5321]],

			[[ 1.0127, -0.1844, -1.0132,  2.0210,  0.8130],
			 [-0.5455, -0.5098, -1.9764,  0.3509, -1.4468],
			 [ 0.0681, -1.0504,  0.6592,  0.1989, -1.4774],
			 [-0.3097,  0.6771,  0.2947, -1.4067, -1.4417]],

			[[ 0.2734, -0.4254,  1.1729,  2.0100, -1.1407],
			 [-0.2713, -0.1622, -0.5532,  0.9638, -0.9287],
			 [ 0.7248, -0.7776, -0.1271,  1.5814, -0.4532],
			 [ 1.5086, -0.0872,  0.5254, -0.2733,  0.5933]]])
	V = tensor([[[ 2.6798, -0.5179,  1.2582, -0.3989, -0.2235],
			 [-1.3648, -0.4122,  1.0682, -0.8368, -1.1953],
			 [ 0.7335,  0.3082, -0.6078, -1.4065, -0.6577],
			 [ 0.3178, -0.8711,  1.2750, -0.3165,  0.2562]],

			[[ 0.1106, -1.0367, -2.2284, -0.4045, -1.2343],
			 [-0.3686,  0.4288, -0.3380, -0.1972,  0.3778],
			 [-0.3289,  0.2250, -0.1146,  1.0722, -0.1600],
			 [ 0.6813, -1.2130, -1.1754,  0.1417,  0.2431]],

			[[ 0.2859, -1.3778,  0.6605, -1.2354,  1.1392],
			 [-1.9311, -0.2242, -2.3424,  0.1420, -1.0550],
			 [ 0.2309, -0.5817,  1.5256, -2.0075, -0.6199],
			 [-1.7653,  2.8820, -0.2060, -1.4014,  0.2813]]])
	scores= [
		[
			[-2.518785, 3.254305, 0.728666]
			[-2.254767, 0.901010, 1.780698]
			[0.795129, -0.546805, 1.513646]
		],
		[
			[2.581363, -0.565546, -1.495607]
			[1.298100, -1.476732, -0.087805]
			[7.120363, -5.996059, -4.716572]
		],
		[
			[-0.246985, -1.383708, -0.494012]
			[-2.280602, -2.038418, -0.445138]
			[2.399786, 3.307809, 1.425397]
		],
		[
			[1.589838, -0.277994, -1.149758]
			[-1.090536, -1.165001, 1.333642]
			[-1.228544, -0.778282, 2.348536]
		],
	]
		*/
}
