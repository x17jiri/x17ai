// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use smallvec::{smallvec, SmallVec};

pub struct UnaryOp {}

pub struct BinaryOp {}

pub struct ReduceParams {
	pub batch_size: usize,
	pub input_size: usize,
}

type ReduceForward = fn(params: &ReduceParams, input: &Tensor, output: &Tensor);

pub struct Expr {
	pub kind: ExprKind,
	pub shape: Rc<Shape>,
	pub dtype: DType,
	pub inputs: SmallVec<[Rc<Expr>; 2]>,
}

pub enum ExprKind {
	Input(Rc<Tensor>),
	Unary(UnaryOp),
	Binary(BinaryOp),
	Reduce(ReduceParams, ReduceForward),
	// TODO - may need new kind for matmul because its inpus can't alias with output
}

impl Expr {
	pub fn new_input(tensor: Rc<Tensor>) -> Self {
		Self {
			kind: ExprKind::Input(tensor),
			shape: tensor.shape.clone(),
			dtype: tensor.dtype,
			inputs: SmallVec::new(),
		}
	}
}

pub fn rms_norm(expr: Rc<Expr>) -> Rc<Expr> {
	let (batch_dims, input_dim) = expr.shape.split(-1);
	let batch_size = batch_dims.iter().product();
	let input_size = input_dim[0];
	Rc::new(Expr {
		kind: ExprKind::Reduce(
			ReduceParams { batch_size, input_size },
			|params: &ReduceParams, input: &Tensor, output: &Tensor| {
				input.buffer.rms_norm(input, output, params);
			},
		),
		shape: expr.shape.clone(),
		dtype: expr.dtype,
		inputs: smallvec![expr],
	})
}
