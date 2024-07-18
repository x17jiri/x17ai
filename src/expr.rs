// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use smallvec::{smallvec, SmallVec};

pub struct UnaryExpr {}

pub struct BinaryExpr {}

pub struct ReduceExpr {
	pub params: ReduceParams,
	pub forward: ReduceForward,
}

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

impl Expr {
	pub fn reshape(self: Rc<Expr>, from_dim: isize, new_dims: &[usize]) -> Rc<Expr> {
		// TODO
	}
}

pub struct InputExpr {
	pub buffer: Rc<dyn Buffer>,
	pub byte_offset: usize,
}

pub enum ExprKind {
	Input(InputExpr),
	Unary(UnaryExpr),
	Binary(BinaryExpr),
	Reduce(ReduceExpr),
	// MatMul(MatMulExpr),
}

impl Expr {
	pub fn new_input(tensor: Tensor) -> Rc<Expr> {
		Rc::new(Self {
			kind: ExprKind::Input(InputExpr {
				buffer: tensor.buffer,
				byte_offset: tensor.byte_offset,
			}),
			shape: tensor.shape,
			dtype: tensor.dtype,
			inputs: SmallVec::new(),
		})
	}
}

pub fn rms_norm(expr: Rc<Expr>) -> Rc<Expr> {
	let (batch_dims, input_dim) = expr.shape.split(-1);
	let batch_size = batch_dims.iter().product();
	let input_size = input_dim[0];
	Rc::new(Expr {
		kind: ExprKind::Reduce(ReduceExpr {
			params: ReduceParams { batch_size, input_size },
			forward: |params: &ReduceParams, input: &Tensor, output: &Tensor| {
				input.buffer.rms_norm(input, output, params);
			},
		}),
		shape: expr.shape.clone(),
		dtype: expr.dtype,
		inputs: smallvec![expr],
	})
}

pub fn softmax(expr: Rc<Expr>) -> Rc<Expr> {
	// TODO
}

pub fn m_dot_v(m: Rc<Expr>, v: Rc<Expr>) -> Rc<Expr> {
	// TODO
}

pub fn m_dot_m(m1: Rc<Expr>, m2: Rc<Expr>) -> Rc<Expr> {
	// TODO
}

pub fn mt_dot_m(m1: Rc<Expr>, m2: Rc<Expr>) -> Rc<Expr> {
	// TODO
}

pub fn m_dot_mt(m1: Rc<Expr>, m2: Rc<Expr>) -> Rc<Expr> {
	// TODO
}

pub fn silu_glu(a: Rc<Expr>) -> Rc<Expr> {
	// TODO
}

pub fn attention_mask(a: Rc<Expr>) -> Rc<Expr> {
	// TODO
}
