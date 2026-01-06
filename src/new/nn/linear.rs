//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

use crate::new::autograd::{AutogradExpr, BackwardFn};
use crate::new::expr::{Expr, TensorRef};
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

/*
impl BackwardFn for LinearBackwardFn {
	fn run(
		self: Box<Self>,
		d_out: Tensor,
		queue: &mut autograd::Queue,
	) -> Result<(), ErrPack<TensorOpError>> {
		let Self {
			weights,
			inp,
			inp_backward,
			backward_scale,
			internal_dtype,
			input_shape,
		} = Box::into_inner(self);

		let mut weights = weights.borrow_mut();
		let d_o = col(&d_out)?;

		// d_w
		if let Some(inp) = inp
			&& weights.requires_grad()
		{
			let i = row(&inp)?;
			let d_w = (d_o * i).scale(1.0);
			weights.update_grad(d_out.dtype(), |current_grad| match current_grad {
				CurrentGradValue::Uninit(tensor) => {
					let grad = mat(tensor)?;
					grad.clear_acc(d_w, internal_dtype)
				},
				CurrentGradValue::Value(_tensor) => {
					todo!("Linear layer backward with existing gradient is not implemented yet");
				},
			})?;
		}

		// d_inp
		if let Some(inp_backward) = inp_backward {
			let w = mat(weights.value())?;
			let d_i = (w.T() * d_o).scale(backward_scale);
			// [... , outputs] -> [... , inputs]
			let d_inp = d_out.new_replace_tail(1, &input_shape, d_out.dtype())?;
			col(&d_inp)?.assign(d_i, internal_dtype)?;
			queue.add(inp_backward, d_inp);
		}

		Ok(())
	}
}
*/

//--------------------------------------------------------------------------------------------------
