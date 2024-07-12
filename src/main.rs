// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(stmt_expr_attributes)]
#![warn(clippy::cast_lossless)]
#![feature(let_chains)]

#[cold]
fn cold_path() {}

#[derive(Debug)]
pub enum Error {
	TooManyDims,
	TooManyElems,
}

mod cpu_dev;
mod expr;
mod format;
mod rand;
mod shape;
mod tensor;

use expr::*;
use std::rc::Rc;

trait Trait {}

struct X {
	x: bool,
}

impl Trait for X {}

struct Y<T: ?Sized> {
	a: usize,
	b: T,
}

fn main1() {
	let a: Rc<dyn Trait> = Rc::new(X { x: true });

	let b: Rc<Y<dyn Trait>> = Rc::new(Y::<X> { a: 4, b: X { x: true } });
	println!("hello")
}
fn main() {
	let dev: Rc<dyn Device> = expr::CPUDevice::new("CPU".to_string());
	/*
		let e = randn(Shape::new(&[7, 5]), DType::Float(32));
		let x = exp(e.clone());
		let z = zeros(Shape::new(&[7, 5]), DType::Float(32));
		let y = add(x.clone(), z);
		let mm = matmul(x, transpose(y, 0, 1));

		let qq = transpose(e, 0, 1);

		//	let t = dev.eval(qq);

		let d3 = randn(Shape::new(&[3, 9, 7]), DType::Float(32));
		//let d3 = zeros(Shape::new(&[3, 9, 7]), DType::Float(32));
		let t = dev.eval(sum(d3, &[1]));
	*/

	let x = randn(Shape::new(&[3, 2]), DType::Float(32));
	let x = dev.clone().eval(x);
	let xx = x.clone();

	let sum = sum(mul(x.clone(), x.clone()), &[1]);
	let q = sqrt(sum);
	let scale = div(ones(Shape::new(&[3, 2]), DType::Float(32)), q);
	let norm = mul(x, scale);
	let t = dev.eval(norm);

	println!("x = {}", xx);
	println!("t = {}", t);
}

/*
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
*/
