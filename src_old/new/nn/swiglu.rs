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

//--------------------------------------------------------------------------------------------------

pub fn swiglu(inp: AutogradExpr, internal_dtype: DType) -> AutogradExpr {
	let (inp, inp_backward) = inp.unpack();
	let inp_backup = inp.clone();

	let inp = inp.label("swiglu.I");
	let (inp, io_dtype) = inp.get_dtype_or_log_error();
	let inp = inp.cast(internal_dtype);

	let lin = inp.clone().select_even();
	let gate = inp.select_odd();

	let one = Expr::new_const("ONE", 1.0);

	let out = lin * gate.clone() * ((-gate).exp() + one).recip();
	let out = out.cast(io_dtype).label("swiglu.O");

	if let Some(inp_backward) = inp_backward {
		let (inp_capture, inp) = inp_backup.capture_into_new("swiglu.I");
		let out = out.first(inp);
		let backward = Box::new(SwiGLUBackwardFn {
			inp: inp_capture,
			inp_backward,
			internal_dtype,
		});
		AutogradExpr::new(out, Some(backward))
	} else {
		AutogradExpr::new(out, inp_backward)
	}
}

//--------------------------------------------------------------------------------------------------

pub struct SwiGLUBackwardFn {
	inp: Rc<TensorRef>,
	inp_backward: Box<dyn BackwardFn>,
	internal_dtype: DType,
}

impl BackwardFn for SwiGLUBackwardFn {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd) {
		let Self { inp, inp_backward, internal_dtype } = Box::into_inner(self);
		let d_out = d_out.label("swiglu.backward.d_out");
		let (d_out, io_dtype) = d_out.get_dtype_or_log_error();
		let d_out = d_out.cast(internal_dtype);

		let inp = inp.to_expr().label("swiglu.backward.I");
		let inp = inp.cast(internal_dtype);

		let lin = inp.clone().select_even();
		let gate = inp.select_odd();

		let one = Expr::new_const("ONE", 1.0);

		let d_lin = d_out.clone() * gate.clone() * (one.clone() + (-gate.clone()).exp()).recip();

		let sigmoid = (one + (-gate.clone()).exp()).recip();
		let swish = gate * sigmoid.clone();
		let d_gate = d_out * lin * (sigmoid.clone() + swish.clone() - (sigmoid * swish));

		let d_inp = d_lin.even_odd(d_gate);
		let d_inp = d_inp.cast(io_dtype);
		let d_inp = d_inp.label("swiglu.backward.d_inp");

		autograd.enqueue(inp_backward, d_inp);
	}
}

//--------------------------------------------------------------------------------------------------
