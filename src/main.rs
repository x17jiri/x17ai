//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#![allow(warnings)] // TODO - disabling warnings for main.rs. Remove this later.
#![allow(non_snake_case)]
#![allow(clippy::manual_is_multiple_of)]
#![feature(generic_const_exprs)]
#![feature(macro_metavar_expr)]
#![feature(string_from_utf8_lossy_owned)]
#![feature(f16)]
#![feature(thin_box)]
#![feature(box_into_inner)]

use std::num::NonZeroUsize;

use x17ai::device::cuda::{
	generate_gemm_kernel,
	GemmEpilogue,
	GemmInput,
	GemmKernelConfig,
	Scale,
};
use x17ai::dtype::DType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let config = GemmKernelConfig {
		a: GemmInput {
			dtype: DType::Int8,
			cols: 2048,
			rows: None,
			trans: false,
		},
		b: GemmInput {
			dtype: DType::Int8,
			cols: 2048,
			rows: NonZeroUsize::new(2048),
			trans: true,
		},
		epilogue: GemmEpilogue::Scale(Scale {
			value: 1.0 / f64::sqrt(2048.0),
			description: "1 / sqrt(2048)".into(),
		}),
		c_dtype: DType::Int8,
	};

	println!("{}", generate_gemm_kernel(&config));
	Ok(())
}
