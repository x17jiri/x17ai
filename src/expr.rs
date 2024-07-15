// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use bit_set::BitSet;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Write as FmtWrite;
use std::intrinsics::{likely, unlikely};
use std::io::Write as IoWrite;
use std::rc::Rc;

pub struct Expr {
	pub shape: Rc<Shape>,
	pub dtype: DType,
	pub kind: ExprKind,
}

impl Expr {
	fn __draw(
		&self,
		file: &mut std::fs::File,
		id: &mut usize,
		roots: Option<&HashMap<*const Expr, isize>>,
	) -> usize {
		let my_id = *id;
		*id += 1;

		let color = if let Some(roots) = roots
			&& roots.contains_key(&(self as *const Expr))
		{
			", fillcolor=lightblue, style=filled"
		} else {
			""
		};
		// using node pointers as IDs
		match self.kind {
			ExprKind::Input(..) => {
				writeln!(file, "\t\"{}\" [label=\"in\"{}]", my_id, color);
			},
			ExprKind::Randn() => {
				writeln!(file, "\t\"{}\" [label=\"randn\"{}]", my_id, color);
			},
			ExprKind::Const(ref c) => {
				writeln!(file, "\t\"{}\" [label=\"const({})\"{}]", my_id, c, color);
			},
			ExprKind::Unary(ref u) => {
				writeln!(
					file,
					"\t\"{}\" [label=\"{}\"{}]",
					my_id,
					u.op.symbol(),
					color
				);

				let a_id = u.a.__draw(file, id, roots);

				writeln!(file, "\t\"{}\" -> \"{}\"", a_id, my_id);
			},
			ExprKind::Binary(ref b) => {
				writeln!(
					file,
					"\t\"{}\" [label=\"{}\"{}]",
					my_id,
					b.op.symbol(),
					color
				);

				let a_id = b.a.__draw(file, id, roots);
				let b_id = b.b.__draw(file, id, roots);

				writeln!(file, "\t\"{}\" -> \"{}\"", a_id, my_id);
				writeln!(file, "\t\"{}\" -> \"{}\"", b_id, my_id);
			},
			ExprKind::Reduce(ref r) => {
				writeln!(
					file,
					"\t\"{}\" [label=\"{}\"{}]",
					my_id,
					r.op.symbol(),
					color
				);

				let a_id = r.a.__draw(file, id, roots);

				writeln!(file, "\t\"{}\" -> \"{}\"", a_id, my_id);
			},
			ExprKind::MatMul(ref m) => {
				writeln!(file, "\t\"{}\" [label=\"MatMul\"{}]", my_id, color);

				let a_id = m.a.__draw(file, id, roots);
				let b_id = m.b.__draw(file, id, roots);

				writeln!(file, "\t\"{}\" -> \"{}\"", a_id, my_id);
				writeln!(file, "\t\"{}\" -> \"{}\"", b_id, my_id);
			},
			ExprKind::Transpose(ref t) => {
				writeln!(
					file,
					"\t\"{}\" [label=\"Transpose {} x {}\"{}]",
					my_id, t.x1, t.x2, color
				);

				let a_id = t.a.__draw(file, id, roots);

				writeln!(file, "\t\"{}\" -> \"{}\"", a_id, my_id);
			},
			ExprKind::Broadcast(ref b) => {
				writeln!(file, "\t\"{}\" [label=\"Broadcast\"{}]", my_id, color);

				let a_id = b.a.__draw(file, id, roots);

				writeln!(file, "\t\"{}\" -> \"{}\"", a_id, my_id);
			},
			ExprKind::SimpleReshape(ref s) => {
				writeln!(file, "\t\"{}\" [label=\"SimpleReshape\"{}]", my_id, color);

				let a_id = s.a.__draw(file, id, roots);

				writeln!(file, "\t\"{}\" -> \"{}\"", a_id, my_id);
			},
		}

		my_id
	}

	pub fn draw(&self, filename: &str, roots: Option<&HashMap<*const Expr, isize>>) {
		let mut file = std::fs::File::create(filename).unwrap();
		writeln!(file, "digraph G {{").unwrap();
		writeln!(file, "\trankdir=BT").unwrap();
		let mut id = 0;
		self.__draw(&mut file, &mut id, roots);
		writeln!(file, "}}").unwrap();
	}
}

pub enum ExprKind {
	Input(Rc<Tensor<dyn TensorData>>),
	Randn(),
	Const(ConstExpr),
	Unary(UnaryExpr),
	Binary(BinaryExpr),
	Reduce(ReduceExpr),
	MatMul(MatMulExpr),
	Transpose(TransposeExpr),
	Broadcast(BroadcastExpr),
	SimpleReshape(SimpleReshapeExpr),
}

// TODO
// - the way we currently handle ConstExpr means that we'll generate a new kernel for each constant.
// Think about the pros and cons and if we should pass constants as kernel params.
pub enum ConstExpr {
	Int(i64),
	Uint(u64),
	Float(f64),
}

impl fmt::Display for ConstExpr {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			ConstExpr::Int(c) => write!(f, "{}", c),
			ConstExpr::Uint(c) => write!(f, "{}", c),
			ConstExpr::Float(c) => write!(f, "{}", c),
		}
	}
}

#[derive(Debug)]
pub enum UnaryOp {
	Exp,
	Sqrt,
}

impl UnaryOp {
	pub fn symbol(&self) -> &'static str {
		match self {
			UnaryOp::Exp => "exp",
			UnaryOp::Sqrt => "sqrt",
		}
	}
}

pub struct UnaryExpr {
	pub a: Rc<Expr>,
	pub op: UnaryOp,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOp {
	Add,
	Sub,
	Mul,
	Div,
}

impl BinaryOp {
	pub fn symbol(&self) -> &'static str {
		match self {
			BinaryOp::Add => "+",
			BinaryOp::Sub => "-",
			BinaryOp::Mul => "*",
			BinaryOp::Div => "/",
		}
	}
	pub fn is_commutative(&self) -> bool {
		match self {
			BinaryOp::Add => true,
			BinaryOp::Mul => true,
			_ => false,
		}
	}
}

pub struct BinaryExpr {
	pub a: Rc<Expr>,
	pub b: Rc<Expr>,
	pub op: BinaryOp,
}

#[derive(Debug)]
pub enum ReduceOp {
	Sum,
	Max,
	Min,
}

impl ReduceOp {
	pub fn symbol(&self) -> &'static str {
		match self {
			ReduceOp::Sum => "SUM",
			ReduceOp::Max => "MAX",
			ReduceOp::Min => "MIN",
		}
	}
}

pub struct ReduceExpr {
	pub a: Rc<Expr>,
	pub op: ReduceOp,

	// TODO - BitSet always allocate memory.
	// We could probably use a single `usize` to store the bits.
	// The number of dimensions is usually very small.
	pub dims_to_reduce: BitSet,
}

pub struct MatMulExpr {
	pub a: Rc<Expr>,
	pub b: Rc<Expr>,
}

// swaps dimensions x1 and x2
pub struct TransposeExpr {
	pub a: Rc<Expr>,
	pub x1: usize,
	pub x2: usize,
}

pub struct BroadcastExpr {
	pub a: Rc<Expr>,
}

// A reshape is "simple" if it only adds or removes dimensions of size 1
pub struct SimpleReshapeExpr {
	pub a: Rc<Expr>,
}

pub fn zeros(shape: Rc<Shape>, dtype: DType) -> Rc<Expr> {
	let c = match dtype {
		DType::Float(_) => ConstExpr::Float(0.0),
		DType::Int(_) => ConstExpr::Int(0),
		DType::Uint(_) => ConstExpr::Uint(0),
	};
	Rc::new(Expr { shape, dtype, kind: ExprKind::Const(c) })
}

pub fn fill(shape: Rc<Shape>, dtype: DType, val: ConstExpr) -> Rc<Expr> {
	Rc::new(Expr { shape, dtype, kind: ExprKind::Const(val) })
}

pub fn randn(shape: Rc<Shape>, dtype: DType) -> Rc<Expr> {
	Rc::new(Expr { shape, dtype, kind: ExprKind::Randn() })
}

pub fn input(tensor: Rc<Tensor<dyn TensorData>>) -> Rc<Expr> {
	Rc::new(Expr {
		shape: tensor.shape.clone(),
		dtype: tensor.dtype,
		kind: ExprKind::Input(tensor),
	})
}

pub fn unary_op(a: Rc<Expr>, op: UnaryOp) -> Rc<Expr> {
	match a.dtype {
		DType::Float(_) => {},
		_ => panic!("{}() requires a float input", op.symbol()),
	}
	Rc::new(Expr {
		shape: a.shape.clone(),
		dtype: a.dtype,
		kind: ExprKind::Unary(UnaryExpr { a, op }),
	})
}

pub fn exp(a: Rc<Expr>) -> Rc<Expr> {
	unary_op(a, UnaryOp::Exp)
}

pub fn sqrt(a: Rc<Expr>) -> Rc<Expr> {
	unary_op(a, UnaryOp::Sqrt)
}

pub fn sqr(a: Rc<Expr>) -> Rc<Expr> {
	mul(a.clone(), a)
}

fn binary_op(mut a: Rc<Expr>, mut b: Rc<Expr>, op: BinaryOp) -> Rc<Expr> {
	if a.dtype != b.dtype {
		panic!("{:?} requires dtypes to match", op);
	}
	match a.shape.broadcast_type(&b.shape) {
		BroadcastType::Error => panic!("{:?} requires shapes to match", op),
		BroadcastType::NoBroadcast => {},
		BroadcastType::Broadcast(a_broadcast, b_broadcast, new_shape) => {
			if a_broadcast {
				a = Rc::new(Expr {
					shape: new_shape.clone(),
					dtype: a.dtype,
					kind: ExprKind::Broadcast(BroadcastExpr { a }),
				});
			}
			if b_broadcast {
				b = Rc::new(Expr {
					shape: new_shape,
					dtype: b.dtype,
					kind: ExprKind::Broadcast(BroadcastExpr { a: b }),
				});
			}
		},
	}
	Rc::new(Expr {
		shape: a.shape.clone(),
		dtype: a.dtype,
		kind: ExprKind::Binary(BinaryExpr { a, b, op }),
	})
}

pub fn add(a: Rc<Expr>, b: Rc<Expr>) -> Rc<Expr> {
	binary_op(a, b, BinaryOp::Add)
}

pub fn sub(a: Rc<Expr>, b: Rc<Expr>) -> Rc<Expr> {
	binary_op(a, b, BinaryOp::Sub)
}

pub fn mul(a: Rc<Expr>, b: Rc<Expr>) -> Rc<Expr> {
	binary_op(a, b, BinaryOp::Mul)
}

pub fn div(a: Rc<Expr>, b: Rc<Expr>) -> Rc<Expr> {
	binary_op(a, b, BinaryOp::Div)
}

fn reduce_op(a: Rc<Expr>, op: ReduceOp, dims_to_reduce: &[isize]) -> Rc<Expr> {
	// NOTE: This will also check if the dimensions are valid
	let new_shape = a.shape.new_reduced(dims_to_reduce);

	let mut bitset = BitSet::with_capacity(a.shape.ndim());
	let ndim = a.shape.ndim();
	for dim in dims_to_reduce {
		let dim = if *dim >= 0 {
			*dim as usize
		} else {
			ndim - ((-dim) as usize)
		};
		bitset.insert(dim);
	}
	Rc::new(Expr {
		shape: new_shape,
		dtype: a.dtype,
		kind: ExprKind::Reduce(ReduceExpr { a, op, dims_to_reduce: bitset }),
	})
}

pub fn sum(a: Rc<Expr>, dims_to_reduce: &[isize]) -> Rc<Expr> {
	reduce_op(a, ReduceOp::Sum, dims_to_reduce)
}

pub fn max(a: Rc<Expr>, dims_to_reduce: &[isize]) -> Rc<Expr> {
	reduce_op(a, ReduceOp::Max, dims_to_reduce)
}

pub fn min(a: Rc<Expr>, dims_to_reduce: &[isize]) -> Rc<Expr> {
	reduce_op(a, ReduceOp::Min, dims_to_reduce)
}

pub fn matmul(a: Rc<Expr>, b: Rc<Expr>) -> Rc<Expr> {
	if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
		panic!("MatMul requires 2D tensors");
	}
	let a_rows = a.shape.dims()[0];
	let a_cols = a.shape.dims()[1];
	let b_rows = b.shape.dims()[0];
	let b_cols = b.shape.dims()[1];
	if a_cols != b_rows {
		panic!("MatMul shapes do not match");
	}
	Rc::new(Expr {
		shape: Shape::new(&[a_rows, b_cols]),
		dtype: a.dtype,
		kind: ExprKind::MatMul(MatMulExpr { a, b }),
	})
}

pub fn transpose(a: Rc<Expr>, x1: usize, x2: usize) -> Rc<Expr> {
	let ndim = a.shape.ndim();
	if x1 >= ndim || x2 >= ndim {
		panic!("Invalid dimensions");
	}
	Rc::new(Expr {
		shape: a.shape.new_transposed(x1, x2),
		dtype: a.dtype,
		kind: ExprKind::Transpose(TransposeExpr { a, x1, x2 }),
	})
}

pub fn simple_reshape(a: Rc<Expr>, new_shape: Rc<Shape>) -> Rc<Expr> {
	let i_shape = a.shape.dims();
	let o_shape = new_shape.dims();

	// check that o_shape only adds or removes dimensions of size 1
	let i = i_shape.iter().filter(|&&x| x != 1);
	let o = o_shape.iter().filter(|&&x| x != 1);
	if unlikely(i.ne(o)) {
		panic!("Requested reshape is not simple");
	}

	Rc::new(Expr {
		shape: new_shape,
		dtype: a.dtype,
		kind: ExprKind::SimpleReshape(SimpleReshapeExpr { a }),
	})
}
