//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::cold_path;
use std::rc::Rc;

use crate::new::autograd::{Autograd, AutogradExpr, BackwardFn};
use crate::new::expr::{CanBeBatched, Expr, TensorRef, ToExpr};
use crate::tensor::DType;

//--------------------------------------------------------------------------------------------------

pub fn rms_norm(inp: AutogradExpr, eps: f64, internal_dtype: DType) -> AutogradExpr {
	let (inp, inp_backward) = inp.unpack();
	let (inp, io_dtype) = if let Some(inp_dtype) = inp.dtype() {
		(inp, inp_dtype)
	} else {
		cold_path();
		(inp.log_error(format!("RMSNorm: input has unknown dtype")), internal_dtype)
	};

	let eps = Expr::new_const("eps".into(), eps);
	let inp = inp.label("rms_norm.inp".into());
	let inp = inp.cast(internal_dtype);

	let magn_recip = ((inp.clone() * inp.clone()).mean().sqrt() + eps).recip();
	let magn_recip = magn_recip.label("rms_norm.magn_recip".into());

	let out = (inp * magn_recip.clone()).cast(io_dtype);
	let out = out.label("rms_norm.out".into());

	if let Some(inp_backward) = inp_backward {
		let out_capture =
			TensorRef::new("rms_norm.out".into(), io_dtype, &[1024], CanBeBatched::Yes);
		let magn_recip_capture =
			TensorRef::new("rms_norm.magn_recip".into(), internal_dtype, &[1], CanBeBatched::Yes);
		let out = out.capture(out_capture.clone());
		let magn_recip = magn_recip.capture(magn_recip_capture.clone());
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

pub struct FakeBackwardFn;

impl BackwardFn for FakeBackwardFn {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd) {
		autograd.eval(d_out);
	}
}

impl BackwardFn for RMSNormBackwardFn_Precise {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd) {
		let Self { out, magn_recip, inp_backward } = Box::into_inner(self);
		let internal_dtype = magn_recip.dtype();
		let (d_out, io_dtype) = if let Some(grad_dtype) = d_out.dtype() {
			(d_out, grad_dtype)
		} else {
			cold_path();
			(
				d_out.log_error(format!("RMSNormBackwardFn_Precise: d_out has unknown dtype")),
				internal_dtype,
			)
		};

		let magn_recip = magn_recip.to_expr().cast(internal_dtype);
		let out = out.to_expr().cast(internal_dtype);
		let d_out = d_out.cast(internal_dtype);
		let g = (out.clone() * d_out.clone()).mean();

		let d_inp = (d_out - (out * g)) * magn_recip;
		let d_inp = d_inp.cast(io_dtype);

		autograd.enqueue(inp_backward, d_inp);
	}
}

//--------------------------------------------------------------------------------------------------
