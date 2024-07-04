// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use core::fmt;
use std::cell::Cell;
use std::collections::HashSet;
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

	fn find_kernel_roots(&self, expr: Rc<Expr>) -> HashSet<*const Expr> {
		let mut roots: HashSet<*const Expr> = HashSet::new();
		let mut processed: HashSet<*const Expr> = HashSet::new();
		let mut to_process: Vec<*const Expr> = Vec::new();

		to_process.push(Rc::as_ptr(&expr));
		roots.insert(Rc::as_ptr(&expr));
		while !to_process.is_empty() {
			let e = to_process.pop().unwrap();
			if processed.contains(&e) {
				roots.insert(e);
				continue;
			}
			processed.insert(e);

			// SAFETY: as long as `expr` is not dropped, the pointers are valid
			let e = unsafe { &*e };

			match &e.kind {
				ExprKind::Leaf(_) => {},
				ExprKind::Unary(u) => {
					to_process.push(&*u.a);
				},
				ExprKind::Binary(b) => {
					to_process.push(&*b.a);
					to_process.push(&*b.b);
				},
				ExprKind::Reduction(r) => {
					to_process.push(&*r.a);
					roots.insert(&*e);
				},
			}
		}
		roots.into_iter().collect()
	}

	fn gen_kernels(&self, roots: HashSet<*const Expr>) -> Vec<CPUKernel> {
		let mut result = Vec::new();
		for root in roots {
			// SAFETY: as long as `expr` is not dropped, the pointers are valid
			let root = unsafe { &*root };
			let name = format!("kernel_{}", result.len());
			let shape = root.shape.as_ref();

			let ndim = shape.ndim();
			let dims = shape.dims();
			let strides = shape.strides();

			let mut code = String::new();
			let mut index = String::new();
			for dim in 0..ndim {
				code.write_fmt(format_args!(
					"{}\tfor (std::size_t i_{} = 0; i_{} < {}; ++i_{}) {{\n",
					Indent(dim),
					dim,
					dim,
					dims[dim],
					dim
				))
				.unwrap();
				if !index.is_empty() {
					index.push_str(" + ");
				}
				index
					.write_fmt(format_args!("{}*i_{}", strides[dim], dim))
					.unwrap();
			}
			code.write_fmt(format_args!(
				"{}\tout_ptr = output + {};\n",
				Indent(ndim),
				index
			))
			.unwrap();
			/*
			match &root.kind {
				ExprKind::Leaf(LeafExpr::Randn()) => {
					let code = format!("randn({})", root.shape.elems());
					kernels.push(CPUKernel { name: "randn".to_string(), code });
				},
				ExprKind::Unary(u) => {
					let code = match u.op {
						UnaryOp::Exp => format!("exp({})", u.a.shape.elems()),
					};
					kernels.push(CPUKernel { name: "exp".to_string(), code });
				},
				ExprKind::Binary(b) => {
					let code = match b.op {
						BinaryOp::Add => {
							format!("add({}, {})", b.a.shape.elems(), b.b.shape.elems())
						},
					};
					kernels.push(CPUKernel { name: "add".to_string(), code });
				},
				ExprKind::Reduction(r) => {
					let code = match r.op {
						ReductionOp::Sum => format!("sum({})", r.a.shape.elems()),
						ReductionOp::Max => format!("max({})", r.a.shape.elems()),
					};
					kernels.push(CPUKernel { name: "sum".to_string(), code });
				},
				_ => {},
			}
			*/

			for dim in (0..ndim).rev() {
				for _ in 0..dim {
					code.push('\t');
				}
				code.write_fmt(format_args!("\t}}\n")).unwrap();
			}

			result.push(CPUKernel { name, code });
		}
		result
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
		let roots = self.find_kernel_roots(expr.clone());
		let kernels = self.gen_kernels(roots);
		for kernel in kernels {
			println!("{}:\n{}", kernel.name, kernel.code);
		}
		unimplemented!();
	}
}
