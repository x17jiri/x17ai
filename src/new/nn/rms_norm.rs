//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::rc::Rc;

use crate::new::autograd::{Autograd, AutogradExpr, BackwardFn};
use crate::new::expr::{CanBeBatched, Expr, TensorRef};
use crate::tensor::DType;

//--------------------------------------------------------------------------------------------------

pub fn rms_norm(
	inp: AutogradExpr,
	eps: f64,
	internal_dtype: DType,
	output_dtype: DType,
) -> AutogradExpr {
	let (inp, inp_backward) = inp.unpack();

	let eps = Expr::new_const("eps".into(), eps);
	let inp = inp.label("rms_norm.inp".into());
	let inp = inp.cast(internal_dtype);

	let magn_recip = ((inp.clone() * inp.clone()).mean().sqrt() + eps).recip();
	let magn_recip = magn_recip.label("rms_norm.magn_recip".into());

	let out = (inp * magn_recip.clone()).cast(output_dtype);
	let out = out.label("rms_norm.out".into());

	if let Some(inp_backward) = inp_backward {
		let out_capture =
			TensorRef::new("rms_norm.out".into(), output_dtype, vec![1024], CanBeBatched::Yes);
		let magn_recip_capture = TensorRef::new(
			"rms_norm.magn_recip".into(),
			internal_dtype,
			vec![1024],
			CanBeBatched::Yes,
		);
		let out = out.capture(out_capture);
		let magn_recip = magn_recip.capture(magn_recip_capture);
		AutogradExpr::new(out.first(magn_recip), None) // TODO: implement backward
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
	fn run(self: Box<Self>, _d_out: Expr, _autograd: &mut Autograd) {
	}
}

impl BackwardFn for RMSNormBackwardFn_Precise {
	fn run(self: Box<Self>, d_out: Expr, autograd: &mut Autograd) {
		let Self { out, magn_recip, inp_backward } = Box::into_inner(self);
		let internal_dtype = common_dtype(d_out.dtype(), magn_recip.dtype());
		let sum_to_mean = out.sum_to_mean();

		let g = magn_recip.new_empty_like(internal_dtype)?; // [..., 1]
		g.assign(custom_kernel!(
			internal_dtype,
			[out: &out, d_out: &d_out], (sum_to_mean: sum_to_mean), {
				(out * d_out).sum() * sum_to_mean
			}
		))?;

		let d_inp = out.reuse_or_new_like()?;

		d_inp.assign(custom_kernel!(
			internal_dtype,
			[d_out: &d_out, out: &out, g: &g, magn_recip: &magn_recip], (), {
				(d_out - (out * g)) * magn_recip
			}
		))?;

		queue.add(inp_backward, d_inp);
		Ok(())
	}
}

//--------------------------------------------------------------------------------------------------
