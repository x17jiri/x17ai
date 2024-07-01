// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(stmt_expr_attributes)]

#[cold]
fn cold_path() {}

#[derive(Debug)]
pub enum Error {
	TooManyDims,
	TooManyElems,
}

mod cpu_dev;
mod format;
mod rand;
mod shape;
mod tensor;
mod x17ai;

use crate::cpu_dev::CPUDevice;
use crate::shape::Shape;
use crate::tensor::{DType, Device, Tensor};

fn main() {
	let mut rng = rand::Rng::new_default();

	let dev = CPUDevice::new("CPU".to_string());
	let mut tensor = Tensor::new_randn(dev.clone(), DType::Float(32), &[7, 5], &mut rng);
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
	let tensor4 = tensor2.exp();
	println!("exp = {}", tensor4);

	let tensor_max = tensor4.max();
	println!("max = {}", tensor_max);

	let tensor_sum = tensor4.sum();
	println!("sum = {}", tensor_sum);
}
