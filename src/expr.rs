// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use smallvec::SmallVec;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub struct Shape {
	// TODO - SmallVec is larger than it should be
	__dims: SmallVec<[usize; 4]>,
}

impl Shape {
	pub fn new_scalar() -> Rc<Self> {
		Rc::new(Self { __dims: SmallVec::new() })
	}

	pub fn new(dims: &[usize]) -> Rc<Self> {
		Rc::new(Self { __dims: SmallVec::from_slice(dims) })
	}

	pub fn ndim(&self) -> usize {
		self.__dims.len()
	}

	pub fn dims(&self) -> &[usize] {
		&self.__dims
	}
}

pub trait Device {
	fn name(&self) -> &str;

	fn eval(&self, expr: Rc<Expr>) -> Rc<Tensor>;
}

pub trait TensorData {}

pub struct Tensor<Data: ?Sized = dyn TensorData> {
	shape: Rc<Shape>,
	dtype: DType,
	device: Rc<dyn Device>,
	data: Data,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DType {
	Float(u8),
	Int(u8),
	Uint(u8),
}

pub struct Expr {
	pub shape: Rc<Shape>,
	pub dtype: DType,
	pub kind: ExprKind,
}

pub enum ExprKind {
	Leaf(LeafExpr),
	Unary(UnaryExpr),
	Binary(BinaryExpr),
	Reduction(ReductionExpr),
}

pub enum LeafExpr {
	IntConst(i64),
	UintConst(u64),
	FloatConst(f64),
	Randn(),
	Read(Rc<Tensor<dyn TensorData>>),
}

pub enum UnaryOp {
	Exp,
}

pub struct UnaryExpr {
	pub a: Rc<Expr>,
	pub op: UnaryOp,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOp {
	Add,
}

pub struct BinaryExpr {
	pub a: Rc<Expr>,
	pub b: Rc<Expr>,
	pub op: BinaryOp,
}

pub enum ReductionOp {
	Sum,
	Max,
}

pub struct ReductionExpr {
	pub a: Rc<Expr>,
	pub op: ReductionOp,
	//	pub axis: isize,
}

pub fn zeros(shape: Rc<Shape>, dtype: DType) -> Rc<Expr> {
	let c = match dtype {
		DType::Float(_) => LeafExpr::FloatConst(0.0),
		DType::Int(_) => LeafExpr::IntConst(0),
		DType::Uint(_) => LeafExpr::UintConst(0),
	};
	Rc::new(Expr { shape, dtype, kind: ExprKind::Leaf(c) })
}

pub fn randn(shape: Rc<Shape>, dtype: DType) -> Rc<Expr> {
	Rc::new(Expr {
		shape,
		dtype,
		kind: ExprKind::Leaf(LeafExpr::Randn()),
	})
}

pub trait ExprLike {
	fn get(self) -> Rc<Expr>;
}

impl ExprLike for Rc<Expr> {
	fn get(self) -> Rc<Expr> {
		self
	}
}

impl ExprLike for Rc<Tensor> {
	fn get(self) -> Rc<Expr> {
		Rc::new(Expr {
			shape: self.shape.clone(),
			dtype: self.dtype,
			kind: ExprKind::Leaf(LeafExpr::Read(self)),
		})
	}
}

pub fn unary_op<A: ExprLike>(a: A, op: UnaryOp) -> Rc<Expr> {
	let a = a.get();
	match a.dtype {
		DType::Float(_) => {},
		_ => panic!("exp() requires a float input"),
	}
	Rc::new(Expr {
		shape: a.shape.clone(),
		dtype: a.dtype,
		kind: ExprKind::Unary(UnaryExpr { a, op: UnaryOp::Exp }),
	})
}

pub fn exp<A: ExprLike>(a: A) -> Rc<Expr> {
	unary_op(a, UnaryOp::Exp)
}

fn binary_op<A: ExprLike, B: ExprLike>(a: A, b: B, op: BinaryOp) -> Rc<Expr> {
	let a = a.get();
	let b = b.get();
	if a.shape != b.shape {
		panic!("{:?} requires shapes to match", op);
	}
	if a.dtype != b.dtype {
		panic!("{:?} requires dtypes to match", op);
	}
	Rc::new(Expr {
		shape: a.shape.clone(),
		dtype: a.dtype,
		kind: ExprKind::Binary(BinaryExpr { a, b, op }),
	})
}

pub fn add<A: ExprLike, B: ExprLike>(a: A, b: B) -> Rc<Expr> {
	binary_op(a, b, BinaryOp::Add)
}

fn reduction_op<A: ExprLike>(a: A, op: ReductionOp) -> Rc<Expr> {
	let a = a.get();
	Rc::new(Expr {
		shape: Shape::new_scalar(),
		dtype: a.dtype,
		kind: ExprKind::Reduction(ReductionExpr { a, op }),
	})
}

pub fn sum<A: ExprLike>(a: A) -> Rc<Expr> {
	reduction_op(a, ReductionOp::Sum)
}

pub fn max<A: ExprLike>(a: A) -> Rc<Expr> {
	reduction_op(a, ReductionOp::Max)
}
