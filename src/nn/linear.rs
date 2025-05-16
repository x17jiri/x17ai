use crate::device::Device;
use crate::dtype::DType;
use crate::eval_context::EvalContext;
use crate::expr::{
	Accumulable, MatrixAccumulable, MatrixSavable, Savable, col_matrix, dot, matrix, mm, randn,
};
use crate::format;
use crate::model_context::ModelContext;
use crate::nn::{BackpropLayer, Layer};
use crate::optimizer::{OptCoef, OptParam};
use crate::param::Param;
use crate::tensor::{self, Tensor, TensorSize};
use std::cell::RefCell;
use std::intrinsics::cold_path;
use std::rc::Rc;

//--------------------------------------------------------------------------------------------------

/// Linear Layer transforming inputs to outputs
///
/// This is basically a thin wrapper around a matrix multiplication.
/// It does not include a bias term.
///
///     input: [..., inputs]
///     output: [..., outputs]
pub struct Linear {
	input_shape: [TensorSize; 1],
	output_shape: [TensorSize; 1],

	weights: Rc<RefCell<Param>>,

	forward_scale: f64,
	backward_scale: f64, // TODO - do we need to scale the backward pass?
}

impl Linear {
	pub fn new(
		inputs: TensorSize, outputs: TensorSize, dtype: DType, ctx: &mut ModelContext,
	) -> Linear {
		Linear {
			input_shape: [inputs],
			output_shape: [outputs],

			weights: ctx.new_param(&[outputs, inputs], dtype),

			forward_scale: 1.0 / (inputs as f64).sqrt(),
			backward_scale: 1.0 / (outputs as f64).sqrt(),
		}
	}

	fn calc_d_weights(&self, d_out: &Tensor, inp: &Tensor) {
		let i = col_matrix(inp);
		let d_o = col_matrix(d_out);
		let d_w = mm(d_o, i.T()).scale(1.0);

		let mut weights = self.weights.borrow_mut();
		weights.update_grad(|grad, already_have_grad| {
			if already_have_grad {
				cold_path();
				d_w.acc_to(matrix(grad), 1.0, 1.0);
			} else {
				d_w.save_to(matrix(grad));
			}
		});
	}

	fn calc_d_inp(&self, d_out: &Tensor) -> Tensor {
		// [... , outputs] -> [... , inputs]
		let d_inp = d_out.new_replace_tail(1, &self.input_shape);
		let d_i = col_matrix(&d_inp);

		let d_o = col_matrix(d_out);

		let w = self.weights.borrow();
		let w = matrix(w.value());
		mm(w.T(), d_o).scale(self.backward_scale).save_to(d_i);
		d_inp
	}
}

impl Layer for Linear {
	fn randomize(&mut self) {
		randn().save_to(self.weights.borrow().value());
	}

	fn input_shape(&self) -> &[TensorSize] {
		&self.input_shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.output_shape
	}

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		f(self.weights.clone());
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		f(format!("{}.weights", prefix), self.weights.clone());
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		// [..., inputs] -> [..., outputs]
		let out = inp.new_replace_tail(1, &self.output_shape);

		let weights = self.weights.borrow();

		let w = matrix(weights.value());
		let i = col_matrix(&inp);
		let o = col_matrix(&out);
		mm(w, i).scale(self.forward_scale).save_to(o);

		if ctx.is_training() {
			ctx.tensors.set([inp]);
		}

		out
	}
}

impl BackpropLayer for Linear {
	fn init_optimizer(&self) {
		let inputs = self.input_shape[0];
		let outputs = self.output_shape[0];
		let mut weights = self.weights.borrow_mut();
		weights.init_optimizer(1, inputs * outputs);
	}

	fn zero_grad(&self) {
		let mut weights = self.weights.borrow_mut();
		weights.zero_grad();
	}

	fn step(&self, opt_coef: &OptCoef) {
		let mut weights = self.weights.borrow_mut();
		weights.step(opt_coef);
	}

	fn final_backward(&self, d_out: Tensor, ctx: &mut EvalContext) {
		let [inp] = ctx.tensors.get();
		self.calc_d_weights(&d_out, &inp);
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		let [inp] = ctx.tensors.get();
		self.calc_d_weights(&d_out, &inp);
		self.calc_d_inp(&d_out)
	}
}

//--------------------------------------------------------------------------------------------------

/// Multihead Linear Layer.
///
/// It works similarly to linear layer, but the same inputs are transformed
/// by multiple heads.
///
///     input: [..., inputs]
///     output: [..., head, outputs]
pub struct MultiheadLinear {
	linear: Linear,

	output_shape: [TensorSize; 2],
}

impl MultiheadLinear {
	pub fn new(
		inputs: TensorSize, outputs: TensorSize, heads: TensorSize, dtype: DType,
		ctx: &mut ModelContext,
	) -> MultiheadLinear {
		let mut linear = Linear::new(inputs, heads * outputs, dtype, ctx);

		// TODO - should we change the backward scale?
		linear.backward_scale = 1.0 / (outputs as f64).sqrt();

		MultiheadLinear { linear, output_shape: [heads, outputs] }
	}
}

impl Layer for MultiheadLinear {
	fn randomize(&mut self) {
		self.linear.randomize();
	}

	fn input_shape(&self) -> &[TensorSize] {
		&self.linear.input_shape
	}

	fn output_shape(&self) -> &[TensorSize] {
		&self.output_shape
	}

	fn collect_params(&self, f: &mut dyn FnMut(Rc<RefCell<Param>>)) {
		self.linear.collect_params(f);
	}

	fn collect_named_params(&self, prefix: &str, f: &mut dyn FnMut(String, Rc<RefCell<Param>>)) {
		self.linear.collect_named_params(format!("{}.linear", prefix).as_str(), f);
	}

	fn forward(&self, inp: Tensor, ctx: &mut EvalContext) -> Tensor {
		let out = self.linear.forward(inp, ctx);

		// [..., heads * outputs] -> [..., heads, outputs]
		out.reshape(1, &self.output_shape)
	}
}

impl BackpropLayer for MultiheadLinear {
	fn init_optimizer(&self) {
		let inputs = self.linear.input_shape[0];
		let heads = self.output_shape[0];
		let outputs = self.output_shape[1];
		let mut weights = self.linear.weights.borrow_mut();
		weights.init_optimizer(heads, inputs * outputs);
	}

	fn zero_grad(&self) {
		self.linear.zero_grad();
	}

	fn step(&self, opt_coef: &OptCoef) {
		self.linear.step(opt_coef);
	}

	fn final_backward(&self, d_out: Tensor, ctx: &mut EvalContext) {
		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.reshape(2, &self.linear.output_shape);

		self.linear.final_backward(d_out, ctx)
	}

	fn backward(&self, d_out: Tensor, ctx: &mut EvalContext) -> Tensor {
		// [..., heads, outputs] -> [..., heads * outputs]
		let d_out = d_out.reshape(2, &self.linear.output_shape);

		self.linear.backward(d_out, ctx)
	}
}

//--------------------------------------------------------------------------------------------------
