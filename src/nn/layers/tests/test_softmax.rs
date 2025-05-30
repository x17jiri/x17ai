// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use super::super::softmax::*;

use crate::debug_2d;
use crate::nn::EvalContext;
use crate::nn::layers::Layer;
use crate::tensor::device::cpu::CPUDevice;
use crate::tensor::{self, Tensor};

// Note: The expected tensors were generated by gen_test_data.py

#[test]
fn test_softmax() {
	let softmax = Softmax::new(5);
	let dev = CPUDevice::new("CPU".to_string());

	#[rustfmt::skip] let inp = Tensor::new_debug_2d(
		dev.clone(),
		debug_2d![
			f32;
			[-1.2719, -0.6884, -0.6477, -1.3343, -1.7648],
			[-1.9440,  0.9989,  2.8260, -0.3503, -0.5406],
			[ 0.1619, -0.9744, -0.6539,  1.9764,  0.7423],
			[ 0.0689,  1.1983,  0.0077, -0.6580, -0.4917],
		]
	);

	#[rustfmt::skip] let expected_out = Tensor::new_debug_2d(
		dev.clone(),
		debug_2d![
			f32;
			[ 0.1610,  0.2886,  0.3006,  0.1513,  0.0984],
			[ 0.0068,  0.1292,  0.8028,  0.0335,  0.0277],
			[ 0.1032,  0.0331,  0.0457,  0.6336,  0.1844],
			[ 0.1642,  0.5081,  0.1545,  0.0794,  0.0938],
		]
	);

	let mut ctx = EvalContext::new(true);
	let out = softmax.forward(inp.clone(), &mut ctx);

	assert!(tensor::math::approx_eq(&out, &expected_out, 1e-4));

	#[rustfmt::skip] let d_out = Tensor::new_debug_2d(
		dev.clone(),
		debug_2d![
			f32;
			[ 0.1000,  0.2000, -0.3000, -0.1000,  0.7000],
			[ 0.0500, -0.1500,  0.1000,  0.0000,  0.6500],
			[-0.2000,  0.1000,  0.0500,  0.0500,  0.3331],
			[ 0.0000,  0.1000, -0.0500, -0.0500, -0.1442],
		]
	);

	#[rustfmt::skip] let expected_d_inp = Tensor::new_debug_2d(
		dev.clone(),
		debug_2d![
			f32;
			[ 0.0101,  0.0469, -0.1014, -0.0208,  0.0652],
			[-0.0002, -0.0296,  0.0167, -0.0027,  0.0158],
			[-0.0287,  0.0007, -0.0013, -0.0178,  0.0470],
			[-0.0042,  0.0378, -0.0117, -0.0060, -0.0159],
		]
	);

	let d_inp = softmax.backward(d_out.clone(), &mut ctx);

	println!("d_inp = {d_inp}");
	println!("expected_d_inp = {expected_d_inp}");

	assert!(tensor::math::approx_eq(&d_inp, &expected_d_inp, 1e-4));
}
