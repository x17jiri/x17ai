// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]

#[cold]
fn cold_path() {}

#[derive(Debug)]
pub enum Error {
	TooManyDims,
	TooManyElems,
}

mod cpu_dev;
mod format;
mod shape;
mod tensor;
mod x17ai;

use crate::cpu_dev::CPUDevice;
use crate::shape::Shape;
use crate::tensor::{DType, Device};

fn main() {
	let dev = CPUDevice::new("CPU".to_string());
	let buf = dev.new_buffer(DType::Float(32), 21).unwrap();
	let shape = Shape::new(&[7, 3]).unwrap();
	let mut tensor = tensor::Tensor { buf, shape };
	println!("init = {}", tensor);
	tensor.shape.transpose(0,1);
	println!("transposed = {}", tensor);
	tensor.zero_();
	println!("zeroed = {}", tensor);
}
