// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#![allow(incomplete_features)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(stmt_expr_attributes)]
#![warn(clippy::cast_lossless)]
#![feature(let_chains)]
#![allow(unused_imports)] // TODO - remove when project stabilizes

#[cold]
fn cold_path() {}

#[derive(Debug)]
pub enum Error {
	TooManyDims,
	TooManyElems,
}

mod compute_seq;
mod cpu;
mod device;
mod dtype;
mod expr;
mod format;
mod rand;
mod shape;
mod tensor;

use crate::compute_seq::*;
use crate::cpu::*;
use crate::device::*;
use crate::dtype::*;
use crate::expr::*;
use crate::format::*;
use crate::rand::*;
use crate::shape::*;
use crate::tensor::*;
use std::rc::Rc;

fn rms_norm(input: Rc<Expr>) -> Rc<Expr> {
	let squared = sqr(input.clone());
	let sum = sum(squared, &[-1]);
	let n = fill(
		sum.shape.clone(),
		input.dtype,
		ConstExpr::Float(input.shape[-1] as f64),
	);
	let scale = sqrt(div(n, sum));
	mul(input, scale)
}

fn linear(input: Rc<Expr>, weights: Rc<Expr>) -> Rc<Expr> {
	let i_shape = input.shape.dims();
	let ndim = i_shape.len();
	let mut vec = vec![0; ndim + 1];
	for i in 0..ndim - 1 {
		vec[i] = i_shape[i];
	}
	vec[ndim - 1] = 1;
	vec[ndim] = i_shape[ndim - 1];
	let i_shape = Shape::new(&vec);

	let input = simple_reshape(input, i_shape);

	matmul(input, weights)
}

fn main() {
	let dev: Rc<dyn Device> = CPUDevice::new("CPU".to_string());
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
	/*
	let r = randn(Shape::new(&[3, 2]), DType::Float(32));
	let x = dev.clone().eval(r, None);
	let xx = x.clone();

	let m = mul(x.clone(), x.clone());
	let mm = dev.clone().eval(m.clone(), Some("m.dot"));

	let sum = sum(m, &[1]);
	let ss = dev.clone().eval(sum.clone(), Some("sum.dot"));
	let q = sqrt(mul(
		sum,
		fill(Shape::new(&[3, 1]), DType::Float(32), ConstExpr::Float(0.5)),
	));
	let qq = dev.clone().eval(q.clone(), Some("q.dot"));

	let scale = div(
		fill(Shape::new(&[3, 1]), DType::Float(32), ConstExpr::Float(1.0)),
		q,
	);
	let norm = mul(x, scale);
	let t = dev.eval(norm.clone(), Some("t.dot"));

	println!("x = {}", xx);
	println!("m = {}", mm);
	println!("sum = {}", ss);
	println!("q = {}", qq);
	println!("t = {}", t);
	*/

	let r = randn(Shape::new(&[2, 5]), DType::Float(32));
	let rr = dev.clone().eval(r.clone(), None);

	let t = rms_norm(input(rr.clone()));
	let tt = dev.clone().eval(t.clone(), Some("t.dot"));

	println!("r = {}", rr);
	println!("t = {}", tt);

	let x = randn(Shape::new(&[5]), DType::Float(32));
	let w = randn(Shape::new(&[5, 3]), DType::Float(32));
	let t2 = linear(x, w);

	let tt2 = dev.eval(t2.clone(), Some("t2.dot"));

	println!("t2 = {}", tt2);
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
