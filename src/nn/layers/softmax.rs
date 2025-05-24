// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use std::cell::RefCell;
use std::rc::Rc;

use crate::nn::eval_context::EvalContext;
use crate::nn::param::Param;
use crate::tensor::math::Savable;
use crate::tensor::{self, Tensor, TensorSize};

use super::Layer;

pub enum SoftmaxGradientMode {
	Precise,
	StraightThrough,
}

pub struct Softmax {
	shape: [TensorSize; 1],
	gradient_mode: SoftmaxGradientMode,
}

impl Softmax {
	pub fn new(n_inputs: TensorSize) -> Softmax {
		Softmax {
			shape: [n_inputs],
			gradient_mode: SoftmaxGradientMode::Precise,
		}
	}

	pub fn set_gradient_mode(&mut self, mode: SoftmaxGradientMode) {
		self.gradient_mode = mode;
	}
}

impl Layer for Softmax {
	fn input_shape(&self) -> &[TensorSize] {
		&self.shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.shape
	}

	fn collect_params(&self, _f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn collect_named_params(&self, _prefix: &str, _f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		// no parameters to collect
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		// try to reuse `inp` for `out` if possible
		let (out, out_ref);
		if inp.owns_buffer() {
			out = None;
			out_ref = &inp;
		} else {
			out = Some(inp.new_empty_like());
			out_ref = out.as_ref().unwrap();
		}

		tensor::math::softmax(&inp).save_to(out_ref);

		if ctx.is_training() {
			match self.gradient_mode {
				SoftmaxGradientMode::Precise => ctx.tensors.set([out_ref.clone()]),
				SoftmaxGradientMode::StraightThrough => {},
			}
		}

		out.unwrap_or(inp)
	}

	fn randomize(&mut self) {
		// no parameters to randomize
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		match self.gradient_mode {
			SoftmaxGradientMode::Precise => {
				let [out] = ctx.tensors.get();

				let g = out.new_replace_tail(1, &[1]); // [..., 1]
				tensor::math::dot(&out, &d_out).save_to(&g);

				// try to reuse the `d_out` for `d_inp` if possible
				let (d_inp, d_inp_ref);
				if d_out.owns_buffer() {
					d_inp = None;
					d_inp_ref = &d_out;
				} else {
					d_inp = Some(d_out.new_empty_like());
					d_inp_ref = d_inp.as_ref().unwrap();
				}

				// TODO - we could merge `sub` and `mul` into a single kernel
				tensor::math::sub(&d_out, &g).save_to(d_inp_ref);
				tensor::math::mul(d_inp_ref, &out).save_to(d_inp_ref);

				d_inp.unwrap_or(d_out)
			},
			SoftmaxGradientMode::StraightThrough => d_out,
		}
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}
/*
#[cfg(test)]
mod tests {
	use super::*;
	use crate::cpu::CPUDevice;
	use crate::device::Device;
	use crate::dtype::DType;
	use assert_approx_eq::assert_approx_eq;

	#[test]
	fn test_softmax() {
		let softmax = Softmax::new(4);
		let dev = CPUDevice::new("CPU".to_string());

		// inp: [ 0.7596,  1.3778,  0.3756, -3.3343]
		let inp = Tensor::new_empty_on(&[4], DType::F32, dev.clone());
		let inp_slice = dev.tensor_as_slice::<f32>(&inp);
		inp_slice[0].set(0.7596);
		inp_slice[1].set(1.3778);
		inp_slice[2].set(0.3756);
		inp_slice[3].set(-3.3343);

		// out: [0.2814, 0.5222, 0.1917, 0.0047] - expected value calculated by PyTorch
		let out = inp.new_empty_like();
		let out_slice = dev.tensor_as_slice::<f32>(&out);

		let mut ctx = EvalContext::new(true);
		softmax.forward(inp.clone(), &mut ctx).save_to(&out);

		assert_approx_eq!(out_slice[0].get(), 0.2814, 1e-4);
		assert_approx_eq!(out_slice[1].get(), 0.5222, 1e-4);
		assert_approx_eq!(out_slice[2].get(), 0.1917, 1e-4);
		assert_approx_eq!(out_slice[3].get(), 0.0047, 1e-4);

		// d_out: [ 0.1000,  0.2000, -0.3000, -0.1000]
		let d_out = out.new_empty_like();
		let d_out_slice = dev.tensor_as_slice::<f32>(&d_out);
		d_out_slice[0].set(0.1000);
		d_out_slice[1].set(0.2000);
		d_out_slice[2].set(-0.3000);
		d_out_slice[3].set(-0.1000);

		// d_in: [ 0.0071,  0.0655, -0.0718, -0.0008] - expected value calculated by PyTorch
		let d_in = out.new_empty_like();
		let d_in_slice = dev.tensor_as_slice::<f32>(&d_in);

		softmax.backward(d_out.clone(), &mut ctx).save_to(&d_in);

		assert_approx_eq!(d_in_slice[0].get(), 0.0071, 1e-4);
		assert_approx_eq!(d_in_slice[1].get(), 0.0655, 1e-4);
		assert_approx_eq!(d_in_slice[2].get(), -0.0718, 1e-4);
		assert_approx_eq!(d_in_slice[3].get(), -0.0008, 1e-4);
	}
}
*/
