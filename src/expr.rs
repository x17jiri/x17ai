// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use core::fmt;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::rc::Rc;

pub struct Indent(usize);

impl fmt::Display for Indent {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		for _ in 0..self.0 {
			write!(f, "\t")?;
		}
		Ok(())
	}
}

#[derive(Debug, PartialEq, Clone)]
pub struct Shape {
	// TODO - SmallVec has more overhead than necessary
	__dims: Vec<usize>,
}

impl Shape {
	pub fn new_scalar() -> Rc<Self> {
		Rc::new(Self { __dims: Vec::new() })
	}

	pub fn new(dims: &[usize]) -> Rc<Self> {
		Rc::new(Self { __dims: dims.to_vec() })
	}

	pub fn ndim(&self) -> usize {
		self.__dims.len()
	}

	pub fn dims(&self) -> &[usize] {
		&self.__dims
	}

	pub fn elems(&self) -> usize {
		self.__dims.iter().product()
	}

	pub fn strides(&self) -> Vec<usize> {
		let mut strides = self.__dims.clone();
		let mut stride = 1;
		for i in (0..strides.len()).rev() {
			let t = strides[i];
			strides[i] = stride;
			stride *= t;
		}
		strides
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

impl DType {
	pub fn bits(&self) -> usize {
		match self {
			DType::Float(b) => *b as usize,
			DType::Int(b) => *b as usize,
			DType::Uint(b) => *b as usize,
		}
	}
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

#[derive(Debug)]
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

#[derive(Debug)]
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

pub struct CPUDevice {
	name: String,
}

struct CPUTensorData {
	data: Box<[Cell<u64>]>,
}

struct CPUKernel {
	name: String,
	code: String,
}

impl CPUDevice {
	// create a new CPU device
	pub fn new(name: String) -> Rc<CPUDevice> {
		Rc::<CPUDevice>::new(CPUDevice { name })
	}

	// create a new uninitialized Tensor
	unsafe fn new_uninit(elem_bits: usize, elems: usize) -> CPUTensorData {
		let Some(total_bits) = elems
			.checked_mul(elem_bits)
			.and_then(|total_bits| total_bits.checked_add(63))
		else {
			panic!("Too many elements");
		};
		let words = total_bits / 64;

		let layout = std::alloc::Layout::array::<Cell<u64>>(words).unwrap();
		let mem = std::alloc::alloc(layout) as *mut Cell<u64>;
		if mem.is_null() {
			panic!("Memory allocation failed");
		}
		let slice = std::slice::from_raw_parts_mut(mem, words);
		CPUTensorData { data: Box::from_raw(slice) }
	}
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn eval(&self, expr: Rc<Expr>) -> Rc<Tensor> {
		/*
		let elems = expr.shape.elems();
		let buffer = unsafe { CPUDevice::new_uninit(expr.dtype.bits(), elems) };
		*/
		let compiler = CPUCompiler::new(expr);
		let kernels = compiler.gen_kernels();
		for kernel in kernels {
			println!("{}:\n{}", kernel.name, kernel.code);
		}
		unimplemented!();
	}
}

struct CPUCompiler {
	expr: Rc<Expr>,

	// as long as `expr` is not dropped, the pointers are valid
	roots: HashMap<*const Expr, usize>,
	roots_postorder: Vec<*const Expr>,
	processed: HashSet<*const Expr>,
}

impl CPUCompiler {
	pub fn new(expr: Rc<Expr>) -> Self {
		let e = expr.as_ref() as *const Expr;
		let mut s = CPUCompiler {
			expr,
			roots: HashMap::new(),
			roots_postorder: Vec::new(),
			processed: HashSet::new(),
		};
		let added = s.find_kernel_roots(unsafe { &*e });
		if !added {
			s.roots.insert(e, 0);
			s.roots_postorder.push(e);
		}
		s
	}

	// returns true if `expr` was added as a root
	fn find_kernel_roots(&mut self, expr: &Expr) -> bool {
		let e = expr as *const Expr;
		if self.processed.contains(&e) {
			return false;
		}
		self.processed.insert(e);

		match &expr.kind {
			// randn always needs its own kernel
			ExprKind::Leaf(LeafExpr::Randn()) => {
				self.roots.insert(e, self.roots_postorder.len());
				self.roots_postorder.push(e);
				true
			},
			ExprKind::Leaf(_) => {
				// do nothing
				false
			},
			ExprKind::Unary(u) => {
				self.find_kernel_roots(&*u.a);
				false
			},
			ExprKind::Binary(b) => {
				self.find_kernel_roots(&*b.a);
				self.find_kernel_roots(&*b.b);
				false
			},
			ExprKind::Reduction(r) => {
				self.find_kernel_roots(&*r.a);
				self.roots.insert(e, self.roots_postorder.len());
				self.roots_postorder.push(e);
				true
			},
		}
	}

	fn gen_kernels(&self) -> Vec<CPUKernel> {
		let mut result = Vec::new();
		for root in 0..self.roots_postorder.len() {
			let name = format!("kernel_{}", root);

			// SAFETY: as long as `expr` is not dropped, the pointers are valid
			let root = unsafe { &*self.roots_postorder[root] };

			if let ExprKind::Leaf(LeafExpr::Randn()) = &root.kind {
				result.push(CPUKernel { name, code: format!("RANDN_TODO;\n") });
				continue;
			}

			result.push(CPUKernel { name, code: self.gen_root_expr(root) });
		}
		result
	}

	fn gen_root_expr(&self, root: &Expr) -> String {
		let shape = root.shape.as_ref();

		let ndim = shape.ndim();
		let dims = shape.dims();
		let strides = shape.strides();

		let mut code = String::new();
		let mut index = String::new();
		for dim in 0..ndim {
			write!(
				code,
				"{}\tfor (std::size_t i_{} = 0; i_{} < {}; ++i_{}) {{\n",
				Indent(dim),
				dim,
				dim,
				dims[dim],
				dim
			);
			if !index.is_empty() {
				index.push_str(" + ");
			}
			write!(index, "{}*i_{}", strides[dim], dim);
		}

		let output = format!("out_ptr[{}]", index);

		match &root.kind {
			ExprKind::Leaf(LeafExpr::Randn()) => {
				unreachable!("Randn should have been handled by gen_kernels()")
			},
			ExprKind::Reduction(r) => {
				unimplemented!("Reduction");
			},
			_ => {
				let val = self.gen_expr(root, root);
				write!(code, "{}{} = {};\n", Indent(ndim + 1), output, val);
			},
		}

		for dim in (0..ndim).rev() {
			for _ in 0..dim {
				code.push('\t');
			}
			write!(code, "\t}}\n");
		}

		code
	}

	fn gen_expr(&self, root: &Expr, expr: &Expr) -> String {
		if (root as *const Expr) != (expr as *const Expr)
			&& let Some(kernel_index) = self.roots.get(&(expr as *const Expr))
		{
			return format!("kernel_{}", kernel_index);
		}

		match &expr.kind {
			ExprKind::Leaf(LeafExpr::IntConst(c)) => format!("{}", c),
			ExprKind::Leaf(LeafExpr::UintConst(c)) => format!("{}", c),
			ExprKind::Leaf(LeafExpr::FloatConst(c)) => format!("{}", c),
			ExprKind::Leaf(LeafExpr::Read(t)) => {
				format!("READ_TENSOR_TODO")
			},
			ExprKind::Unary(un) => {
				let a = self.gen_expr(root, &un.a);
				match un.op {
					UnaryOp::Exp => format!("exp({})", a),
				}
			},
			ExprKind::Binary(bin) => {
				let a = self.gen_expr(root, &bin.a);
				let b = self.gen_expr(root, &bin.b);
				match bin.op {
					BinaryOp::Add => format!("add({}, {})", a, b),
				}
			},
			_ => {
				panic!("Unsupported expression");
			},
		}
	}
}
