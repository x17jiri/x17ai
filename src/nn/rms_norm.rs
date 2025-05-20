use std::cell::RefCell;
use std::rc::Rc;

use crate::eval_context::EvalContext;
use crate::expr::{self, Accumulable, Savable};
use crate::nn::Layer;
use crate::param::Param;
use crate::tensor::{Tensor, TensorSize};

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RMSNormGradientMode {
	Precise,
	NormGradients,
	StraightThrough,
}

pub struct RMSNorm {
	shape: [TensorSize; 1],
	eps: f64,
	gradient_mode: RMSNormGradientMode,
}

impl RMSNorm {
	pub fn new(n_inputs: TensorSize, eps: f64) -> RMSNorm {
		RMSNorm {
			shape: [n_inputs],
			eps,
			gradient_mode: RMSNormGradientMode::Precise,
		}
	}
}

impl Layer for RMSNorm {
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

		if ctx.is_training() && self.gradient_mode == RMSNormGradientMode::Precise {
			let scale = out_ref.new_replace_tail(1, &[1]);

			expr::rms_norm(&inp, self.eps).scale_storage(&scale).save_to(out_ref);

			ctx.tensors.set([out_ref.clone(), scale]);
		} else {
			expr::rms_norm(&inp, self.eps).save_to(out_ref);
		}

		out.unwrap_or(inp)
	}

	fn randomize(&mut self) {
		// no parameters to randomize
	}

	fn init_optimizer(&self) {
		// no parameters to optimize
	}

	fn zero_grad(&self) {
		// no parameters to update
	}

	fn step(&self, _opt_coef: &crate::optimizer::OptCoef) {
		// no parameters to update
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		match self.gradient_mode {
			RMSNormGradientMode::Precise => {
				let [out, scale] = ctx.tensors.get();

				let g = scale.new_empty_like(); // [..., 1]
				expr::dot(&out, &d_out).scale(1.0 / self.shape[0] as f64).save_to(&g);

				// try to reuse `out` for `d_inp` if possible
				let (d_inp, d_inp_ref);
				if out.owns_buffer() {
					d_inp = None;
					d_inp_ref = &out;
				} else {
					d_inp = Some(out.new_empty_like());
					d_inp_ref = d_inp.as_ref().unwrap();
				}

				// TODO - could we merge `mul, sub, mul` into a single kernel?
				expr::mul(&out, &g).save_to(d_inp_ref);
				expr::sub(&d_out, d_inp_ref).save_to(d_inp_ref);
				expr::mul(d_inp_ref, &scale).save_to(d_inp_ref);

				d_inp.unwrap_or(out)
			},
			RMSNormGradientMode::NormGradients => {
				// try to reuse `d_out` for `d_inp` if possible
				let (d_inp, d_inp_ref);
				if d_out.owns_buffer() {
					d_inp = None;
					d_inp_ref = &d_out;
				} else {
					d_inp = Some(d_out.new_empty_like());
					d_inp_ref = d_inp.as_ref().unwrap();
				}

				expr::rms_norm(&d_out, self.eps).save_to(d_inp_ref);

				d_inp.unwrap_or(d_out)
			},
			RMSNormGradientMode::StraightThrough => d_out,
		}
	}

	fn backward_finish(&self, _d_out: Tensor, _ctx: &mut EvalContext) {
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
