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
	let buf = dev.new_buffer(DType::Float(32), 35).unwrap();
	let shape = Shape::new(&[7, 5]).unwrap();
	let mut tensor = tensor::Tensor { buf, shape };
	println!("init = {}", tensor);
	tensor.shape.transpose(0, 1);
	println!("transposed = {}", tensor);
	let tensor2 = tensor.slice(1, 2..4);
	let tensor3 = tensor.slice(0, 1..3);
	println!("sliced2 = {}", tensor2);
	println!("sliced3 = {}", tensor3);
	tensor3.zero_();
	println!("zeroed1 = {}", tensor);
	println!("zeroed2 = {}", tensor2);
	println!("zeroed3 = {}", tensor3);
}
