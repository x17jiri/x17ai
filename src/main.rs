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

use x17ai::device::cpu::CPUDevice;
use x17ai::literal::TensorLiteral2D;
use x17ai::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let device = CPUDevice::new();
	let literal = TensorLiteral2D::<f32>::new(&[
		[1.0, 2.0, 3.0],
		[4.0, 5.0, 6.0],
	]);
	let tensor = Tensor::new(&literal, device)?;

//	println!("{tensor}");
	Ok(())
}
