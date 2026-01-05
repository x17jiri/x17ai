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
	let (inp, io_dtype) = inp.get_dtype_or_log_error();
	let inp = inp.label("swiglu.I");
	let inp = inp.cast(internal_dtype);

	let lin = inp.clone().select_even();
	let gate = inp.select_odd();

	let one = Expr::new_const("ONE", 1.0);

	let out = lin * gate.clone() * ((-gate).exp() + one).recip();
	let out = out.cast(io_dtype).label("swiglu.O");

	/*if let Some(inp_backward) = inp_backward {
		let (out_capture, out) = out.capture_into_new("swiglu.out");
		let (magn_recip_capture, magn_recip) = magn_recip.capture_into_new("swiglu.magn_recip");
		let out = out.first(magn_recip);
		let backward = Box::new(RMSNormBackwardFn_Precise {
			out: out_capture,
			magn_recip: magn_recip_capture,
			inp_backward,
		});
		AutogradExpr::new(out, Some(backward))
	} else*/
	{ AutogradExpr::new(out, inp_backward) }
}

//--------------------------------------------------------------------------------------------------

pub struct RMSNormBackwardFn_Precise {
	out: Rc<TensorRef>,
	magn_recip: Rc<TensorRef>,
	inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for RMSNormBackwardFn_Precise {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd) {
		let Self { out, magn_recip, inp_backward } = Box::into_inner(self);
		let d_out = d_out.label("rms_norm.backward.d_out");
		let (d_out, io_dtype) = d_out.get_dtype_or_log_error();
		let internal_dtype = magn_recip.dtype();

		let magn_recip = magn_recip.to_expr().cast(internal_dtype);
		let out = out.to_expr().cast(internal_dtype);
		let d_out = d_out.cast(internal_dtype);
		let g = (out.clone() * d_out.clone()).mean();
		let g = g.label("rms_norm.backward.g");

		let d_inp = (d_out - (out * g)) * magn_recip;
		let d_inp = d_inp.cast(io_dtype);
		let d_inp = d_inp.label("rms_norm.backward.d_inp");

		autograd.enqueue(inp_backward, d_inp);
	}
}

//--------------------------------------------------------------------------------------------------
