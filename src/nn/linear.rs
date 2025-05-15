use crate::device::Device;
use crate::dtype::DType;
use crate::eval_ctx::EvalContext;
use crate::expr::{
	Accumulable, MatrixAccumulable, MatrixSavable, Savable, col_matrix, dot, matrix, mm, randn,
};
use crate::model_ctx::ModelContext;
use crate::nn::{BackpropLayer, Layer, TensorStore};
use crate::optimizer::OptParam;
use crate::param::Param;
use crate::tensor::{self, Tensor, TensorSize};
use std::cell::RefCell;
use std::intrinsics::cold_path;
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
	inputs: TensorSize,
	outputs: TensorSize,
	heads: TensorSize,

	input_shape: [TensorSize; 1],
	output_shape: [TensorSize; 2],

	pub(crate) w: Rc<RefCell<Param>>, // TODO - can we remove the `pub(crate)`?

	forward_scale: f64,
	backward_scale: f64,
}

impl Linear {
	pub fn new(
		inputs: TensorSize, outputs: TensorSize, heads: TensorSize, dtype: DType,
		ctx: &mut ModelContext,
	) -> Linear {
		Linear {
			inputs,
			outputs,
			heads,

			input_shape: [inputs],
			output_shape: [heads, outputs],

			w: ctx.new_param(&[heads * outputs, inputs], dtype),

			forward_scale: 1.0 / (inputs as f64).sqrt(),
			backward_scale: 1.0 / (outputs as f64).sqrt(),
		}
	}
}

impl Layer for Linear {
	fn randomize(&mut self) {
		randn().save_to(self.w.borrow().value());
	}

	fn input_shape(&self) -> &[TensorSize] {
		&self.input_shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.output_shape
	}

	fn forward(&self, inp: &Tensor, out: &Tensor, ctx: &mut EvalContext) {
		// [..., heads, outputs] -> [..., heads * outputs]
		let out = out.clone().reshape(2, &[self.heads * self.outputs]);
		let w = self.w.borrow();

		let w = matrix(w.value());
		let i = col_matrix(&inp);
		let o = col_matrix(&out);
		mm(w, i).scale(self.forward_scale).save_to(o);

		if ctx.is_training() {
			ctx.tensors.set([inp.clone()]);
		}
	}
}

impl BackpropLayer for Linear {
	fn backward(&self, d_out: &Tensor, d_inp: Option<&Tensor>, ctx: &mut EvalContext) {
		let [inp] = ctx.tensors.get();

		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.clone().reshape(2, &[self.heads * self.outputs]);

		let i = col_matrix(&inp);
		let d_o = col_matrix(&d_out);
		let d_w = mm(d_o, i.T()).scale(1.0);

		self.w.borrow_mut().update_grad(|grad, already_have_grad| {
			if already_have_grad {
				cold_path();
				d_w.acc_to(matrix(grad), 1.0, 1.0);
			} else {
				d_w.save_to(matrix(grad));
			}
		});

		if let Some(d_inp) = d_inp {
			let w = self.w.borrow();
			let w = matrix(w.value());
			let d_i = col_matrix(d_inp);
			mm(w.T(), d_o).scale(self.backward_scale).save_to(d_i);
		}
	}
}
