//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::new::expr::{CanBeBatched, Expr, TensorRef};
use crate::tensor::DType;

//--------------------------------------------------------------------------------------------------

pub fn rms_norm0(inp: Expr, eps: f64, internal_dtype: DType, output_dtype: DType) -> Expr {
	let eps = Expr::new_const("eps".into(), eps);
	let inp = inp.label("rms_norm.inp".into());
	let inp = inp.cast(internal_dtype);

	let magn_recip = ((inp.clone() * inp.clone()).mean().sqrt() + eps).recip();
	let magn_recip = magn_recip.label("rms_norm.magn_recip".into());

	let out = (inp * magn_recip).cast(output_dtype);
	let out = out.label("rms_norm.out".into());
	out
}

pub fn rms_norm(
	inp: Expr,
	eps: f64,
	internal_dtype: DType,
	output_dtype: DType,
	grad: bool,
) -> Expr {
	let out_capture = if grad {
		Some(TensorRef::new("rms_norm.out".into(), output_dtype, vec![1024], CanBeBatched::Yes))
	} else {
		None
	};
	let magn_recip_capture = if grad {
		Some(TensorRef::new(
			"rms_norm.magn_recip".into(),
			internal_dtype,
			vec![1024],
			CanBeBatched::Yes,
		))
	} else {
		None
	};

	let eps = Expr::new_const("eps".into(), eps);
	let inp = inp.label("rms_norm.inp".into());
	let inp = inp.cast(internal_dtype);

	let magn_recip = ((inp.clone() * inp.clone()).mean().sqrt() + eps).recip();
	let magn_recip = magn_recip.label("rms_norm.magn_recip".into());
	let magn_recip = magn_recip.optional_capture(magn_recip_capture);

	let out = (inp * magn_recip).cast(output_dtype);
	let out = out.label("rms_norm.out".into());
	let out = out.optional_capture(out_capture);
	out
}

//--------------------------------------------------------------------------------------------------
