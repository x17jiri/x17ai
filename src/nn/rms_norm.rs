use std::cell::RefCell;
use std::rc::Rc;

use crate::eval_context::EvalContext;
use crate::expr::{self, Accumulable, Savable};
use crate::nn::{BackpropLayer, Layer, LossLayer};
use crate::param::Param;
use crate::tensor::{Tensor, TensorSize};

pub enum RMSNormGradientMode {
	Precise,
	StraightThrough,
}

pub struct RMSNorm {
	shape: [TensorSize; 1],
	eps: f64,
	gradient_mode: RMSNormGradientMode,
}

impl RMSNorm {
	pub fn new(classes: TensorSize, eps: f64) -> RMSNorm {
		RMSNorm {
			shape: [classes],
			eps,
			gradient_mode: RMSNormGradientMode::Precise,
		}
	}
}

impl Layer for RMSNorm {
	fn randomize(&mut self) {
		// no parameters to randomize
	}

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
		let out = inp.new_empty_like();

		expr::rms_norm(&inp, self.eps).save_to(&out);

		if ctx.is_training() {
			match self.gradient_mode {
				RMSNormGradientMode::StraightThrough => {},

				_ => {
					// TODO - pull the scale out of the `rms_norm()` call
					let scale = out.new_replace_tail(1, &[1]);
					expr::dot(&inp, &inp).scale(1.0 / self.shape[0] as f64).save_to(&scale);
					expr::rsqrt(&scale, self.eps).save_to(&scale);

					ctx.tensors.set([out.clone(), scale]);
				},
			}
		}

		out
	}
}

impl BackpropLayer for RMSNorm {
	fn init_optimizer(&self) {
		// no parameters to optimize
	}

	fn zero_grad(&self) {
		// no parameters to update
	}

	fn step(&self, _opt_coef: &crate::optimizer::OptCoef) {
		// no parameters to update
	}

	fn backward(&self, d_out: Tensor, _ctx: &mut EvalContext) -> Tensor {
		match self.gradient_mode {
			RMSNormGradientMode::Precise => {
				let [out, scale] = _ctx.tensors.get();

				let g = scale.new_empty_like(); // [..., 1]
				let d_inp = out.new_empty_like(); // [..., classes]

				// TODO - could we merge `mul, sub, mul` into a single kernel?
				expr::dot(&out, &d_out).scale(1.0 / self.shape[0] as f64).save_to(&g);
				expr::mul(&out, &g).save_to(&d_inp);
				expr::sub(&d_out, &d_inp).save_to(&d_inp);
				expr::mul(&d_inp, &scale).save_to(&d_inp);

				d_inp
			},
			RMSNormGradientMode::StraightThrough => d_out,
		}
	}

	fn backward_first(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
		// no parameters to update
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cpu::CPUDevice;
	use crate::device::Device;
	use crate::dtype::DType;
	use crate::nn::rms_norm;
	use assert_approx_eq::assert_approx_eq;

	#[test]
	fn test_rms_norm() {
		let rms_norm = RMSNorm::new(4, 1e-5);
		let dev = CPUDevice::new("CPU".to_string());

		// inp: [ 0.7596,  1.3778,  0.3756, -3.3343]
		let inp = Tensor::new_empty_on(&[4], DType::F32, dev.clone());
		let inp_slice = dev.tensor_as_slice::<f32>(&inp);
		inp_slice[0].set(0.7596);
		inp_slice[1].set(1.3778);
		inp_slice[2].set(0.3756);
		inp_slice[3].set(-3.3343);

		// out: [ 0.4099,  0.7436,  0.2027, -1.7994] - expected value calculated by PyTorch
		let out = inp.new_empty_like();
		let out_slice = dev.tensor_as_slice::<f32>(&out);

		let mut ctx = EvalContext::new(true);
		rms_norm.forward(inp.clone(), &mut ctx).save_to(&out);

		assert_approx_eq!(out_slice[0].get(), 0.4099, 1e-4);
		assert_approx_eq!(out_slice[1].get(), 0.7436, 1e-4);
		assert_approx_eq!(out_slice[2].get(), 0.2027, 1e-4);
		assert_approx_eq!(out_slice[3].get(), -1.7994, 1e-4);

		// d_out: [ 0.1000,  0.2000, -0.3000, -0.1000]
		let d_out = out.new_empty_like();
		let d_out_slice = dev.tensor_as_slice::<f32>(&d_out);
		d_out_slice[0].set(0.1000);
		d_out_slice[1].set(0.2000);
		d_out_slice[2].set(-0.3000);
		d_out_slice[3].set(-0.1000);

		// d_in: [ 0.0369,  0.0770, -0.1703,  0.0210] - expected value calculated by PyTorch
		let d_in = out.new_empty_like();
		let d_in_slice = dev.tensor_as_slice::<f32>(&d_in);

		rms_norm.backward(d_out.clone(), &mut ctx).save_to(&d_in);

		assert_approx_eq!(d_in_slice[0].get(), 0.0369, 1e-4);
		assert_approx_eq!(d_in_slice[1].get(), 0.0770, 1e-4);
		assert_approx_eq!(d_in_slice[2].get(), -0.1703, 1e-4);
		assert_approx_eq!(d_in_slice[3].get(), 0.0210, 1e-4);
	}
}
