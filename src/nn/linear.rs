use crate::context::Context;
use crate::device::Device;
use crate::dtype::DType;
use crate::expr::{
	Accumulable, MatrixAccumulable, MatrixSavable, Savable, col_matrix, dot, matrix, mm, randn,
};
use crate::nn::Layer;
use crate::optimizer::OptParam;
use crate::tensor::{Tensor, TensorSize};
use std::cell::RefCell;
use std::rc::Rc;

/// Multihead Linear Layer transforming inputs to outputs
///
/// This is basically a thin wrapper around a matrix multiplication.
/// It does not include a bias term.
///
/// This works like n parallel linear transformations of the same input:
///
///     input: [*, inputs]
///     output: [*, head, outputs]
pub struct Linear {
	pub inputs: TensorSize,
	pub outputs: TensorSize,
	pub heads: TensorSize,

	pub input_shape: [TensorSize; 1],
	pub output_shape: [TensorSize; 2],

	pub w: Tensor,
	pub w_opt: Rc<RefCell<OptParam>>,

	pub forward_scale: f64,
	pub backward_scale: f64,
	pub dtype: DType,
}

impl Linear {
	pub fn new(
		inputs: TensorSize, outputs: TensorSize, heads: TensorSize, dtype: DType, ctx: &mut Context,
	) -> Linear {
		let w_opt = ctx.add_param(dtype, heads, outputs * inputs);
		let w = w_opt.borrow().value().clone();
		let w = w.reshape_all(&[heads * outputs, inputs]);

		let forward_scale = 1.0 / (inputs as f64).sqrt();
		let backward_scale = 1.0 / (outputs as f64).sqrt();

		#[rustfmt::skip]
		Linear {
			inputs, outputs, heads,
			input_shape: [inputs],
			output_shape: [heads, outputs],
			w, w_opt,
			forward_scale, backward_scale, dtype,
		}
	}
}

impl Layer for Linear {
	fn randomize(&mut self) {
		let w_opt = self.w_opt.borrow_mut();
		let w = w_opt.value();
		randn().save_to(w);
	}

	fn input_shape(&self) -> &[TensorSize] {
		&self.input_shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.output_shape
	}

	fn forward(&self, inp: &Tensor, out: &Tensor) {
		// [..., heads, outputs] -> [..., heads * outputs]
		let out = out.clone().reshape(2, &[self.heads * self.outputs]);

		let w = matrix(&self.w);
		let i = col_matrix(&inp);
		let o = col_matrix(&out);
		mm(w, i).scale(self.forward_scale).save_to(o);

		self.w_opt.borrow_mut().save_tensors([inp.clone()]);
	}

	fn backward(&self, d_out: &Tensor, d_inp: Option<&Tensor>) {
		let [inp] = self.w_opt.borrow_mut().load_tensors();

		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.clone().reshape(2, &[self.heads * self.outputs]);

		// [heads, outputs * inputs] -> [heads * outputs, inputs]
		let d_weights = self.w_opt.borrow().grad().clone();
		let d_weights = d_weights.reshape_all(&[self.heads * self.outputs, self.inputs]);

		let d_w = matrix(&d_weights);
		let i = col_matrix(&inp);
		let d_o = col_matrix(&d_out);
		mm(d_o, i.T()).scale(1.0).acc_to(d_w, 1.0, 1.0);

		if let Some(d_inp) = d_inp {
			let w = matrix(&self.w);
			let d_i = col_matrix(d_inp);
			mm(w.T(), d_o).scale(self.backward_scale).save_to(d_i);
		}
	}
}
