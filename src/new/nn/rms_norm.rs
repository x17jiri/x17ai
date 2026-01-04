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

pub fn rms_norm(inp: AutogradExpr, eps: f64, internal_dtype: DType) -> AutogradExpr {
	let (inp, inp_backward) = inp.unpack();
	let (inp, io_dtype) = inp.get_dtype_or_log_error();
	let inp = inp.label("rms_norm.inp");
	let inp = inp.cast(internal_dtype);

	let eps = Expr::new_const("eps", eps);

	let magn_recip = ((inp.clone() * inp.clone()).mean().sqrt() + eps).recip();
	let magn_recip = magn_recip.label("rms_norm.magn_recip");

	let out = (inp * magn_recip.clone()).cast(io_dtype);
	let out = out.label("rms_norm.out");

	if let Some(inp_backward) = inp_backward {
		let (out_capture, out) = out.capture_into_new("rms_norm.out");
		let (magn_recip_capture, magn_recip) = magn_recip.capture_into_new("rms_norm.magn_recip");
		AutogradExpr::new(
			out.first(magn_recip),
			Some(Box::new(RMSNormBackwardFn_Precise {
				out: out_capture,
				magn_recip: magn_recip_capture,
				inp_backward,
			})),
		)
	} else {
		AutogradExpr::new(out, None)
	}
}

pub struct RMSNormBackwardFn_Precise {
	out: Rc<TensorRef>,
	magn_recip: Rc<TensorRef>,
	inp_backward: Box<dyn BackwardFn>,
}

impl BackwardFn for RMSNormBackwardFn_Precise {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd) {
		let Self { out, magn_recip, inp_backward } = Box::into_inner(self);
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
