//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

use crate::new::autograd::{Autograd, AutogradExpr, BackwardFn};
use crate::new::expr::{Expr, TensorRef, ToExpr};
use crate::tensor::DType;
use crate::util::LossyFrom;

//--------------------------------------------------------------------------------------------------

#[allow(clippy::manual_map)]
pub fn linear(inp: AutogradExpr, weights: AutogradExpr, internal_dtype: DType) -> AutogradExpr {
	let (mut inp, inp_backward) = inp.unpack();
	let (weights, weights_backward) = weights.unpack();

	let d_inp_data = if let Some(inp_backward) = inp_backward {
		Some(DInpData {
			inp_backward,
			weights: weights.clone(),
			backward_scale: 1.0, // TODO
		})
	} else {
		None
	};
	let d_w_data = if let Some(weights_backward) = weights_backward {
		let (inp_capture, new_inp) = inp.capture_into_new("linear.I");
		inp = new_inp;
		Some(DWData { weights_backward, inp_capture })
	} else {
		None
	};

	let inp = inp.label("linear.I");
	let (inp, io_dtype) = inp.get_dtype_or_log_error();
	let inp = inp.cast(internal_dtype);

	let weights = weights.label("linear.W");
	let weights = weights.cast(internal_dtype);

	let n_inputs = weights.size(-1);
	let scale = 1.0 / f64::lossy_from(n_inputs).sqrt();
	let scale = Expr::new_const(format!("scale = 1.0 / √{n_inputs}"), scale);

	let out = (weights.mat_times_col(inp) * scale).cast(io_dtype);
	let out = out.label(format!("linear.O"));

	if d_inp_data.is_some() || d_w_data.is_some() {
		let backward = Box::new(LinearBackwardFn { d_inp_data, d_w_data, internal_dtype });
		AutogradExpr::new(out, Some(backward))
	} else {
		AutogradExpr::new(out, None)
	}
}

//--------------------------------------------------------------------------------------------------

pub struct DInpData {
	inp_backward: Box<dyn BackwardFn>,
	weights: Expr,
	backward_scale: f64,
}

pub struct DWData {
	weights_backward: Box<dyn BackwardFn>,
	inp_capture: Rc<TensorRef>,
}

pub struct LinearBackwardFn {
	d_inp_data: Option<DInpData>,
	d_w_data: Option<DWData>,
	internal_dtype: DType,
}

impl BackwardFn for LinearBackwardFn {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd) {
		let Self { d_inp_data, d_w_data, internal_dtype } = Box::into_inner(self);

		let d_out = d_out.label("linear.backward.d_out");
		let (d_out, io_dtype) = d_out.get_dtype_or_log_error();
		let d_out = d_out.cast(internal_dtype);

		// d_w
		if let Some(DWData { weights_backward, inp_capture }) = d_w_data {
			let inp = inp_capture.to_expr().cast(internal_dtype);

			let d_weights = d_out.clone().cols_times_rows(inp);

			let d_weights = d_weights.cast(io_dtype);
			let d_weights = d_weights.label("linear.backward.d_weights");
			autograd.enqueue(weights_backward, d_weights);
		}

		// d_inp
		if let Some(DInpData { inp_backward, weights, backward_scale }) = d_inp_data {
			let backward_scale = Expr::new_const("backward_scale", backward_scale);
			let weights = weights.label("linear.backward.weights");
			let weights = weights.cast(internal_dtype);

			let d_inp = (d_out.row_times_mat(weights)) * backward_scale;

			let d_inp = d_inp.cast(io_dtype);
			let d_inp = d_inp.label("linear.backward.d_inp");
			autograd.enqueue(inp_backward, d_inp);
		}
	}
}

//--------------------------------------------------------------------------------------------------
