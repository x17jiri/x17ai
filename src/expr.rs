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

struct DimIndex {
	i: String,
	stride: String,
}

struct Index {
	dims: Vec<DimIndex>,
}

impl Index {
	fn new() -> Self {
		Index { dims: Vec::new() }
	}
}

impl fmt::Display for Index {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		for i in 0..self.dims.len() {
			let dim = &self.dims[i];
			if i != 0 {
				write!(f, " + ")?;
			}
			write!(f, "{}*{}", dim.stride, dim.i)?;
		}
		Ok(())
	}
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
	roots: HashMap<*const Expr, isize>,
	roots_postorder: Vec<(*const Expr, Vec<usize>)>,
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
		s.find_kernel_roots(unsafe { &*e });
		s.roots.insert(e, -1);
		s
	}

	fn find_kernel_roots(&mut self, expr: &Expr) {
		let e = expr as *const Expr;
		if self.processed.contains(&e) {
			self.roots.insert(e, -1);
			return;
		}
		self.processed.insert(e);

		match &expr.kind {
			// randn always needs its own kernel
			ExprKind::Leaf(LeafExpr::Randn()) => {
				self.roots.insert(e, -1);
			},
			ExprKind::Leaf(_) => {
				// do nothing
			},
			ExprKind::Unary(u) => {
				self.find_kernel_roots(&*u.a);
			},
			ExprKind::Binary(b) => {
				self.find_kernel_roots(&*b.a);
				self.find_kernel_roots(&*b.b);
			},
			ExprKind::Reduction(r) => {
				self.find_kernel_roots(&*r.a);
				self.roots.insert(e, -1);
			},
		}
	}

	fn find_postorder(&mut self, expr: &Expr, inputs: &mut Vec<usize>) {
		let e = expr as *const Expr;
		if let Some(t) = self.roots.get(&e)
			&& *t >= 0
		{
			inputs.push(*t as usize);
			return;
		}

		match &expr.kind {
			// randn always needs its own kernel
			ExprKind::Leaf(LeafExpr::Randn()) => {
				self.roots.insert(e, self.roots_postorder.len());
				inputs.push(self.roots_postorder.len());
				self.roots_postorder.push((e, Vec::new()));
			},
			ExprKind::Leaf(_) => {
				// do nothing
			},
			ExprKind::Unary(u) => {
				self.find_kernel_roots(&*u.a, inputs);
			},
			ExprKind::Binary(b) => {
				self.find_kernel_roots(&*b.a, inputs);
				self.find_kernel_roots(&*b.b, inputs);
			},
			ExprKind::Reduction(r) => {
				let mut nested_inputs = Vec::new();
				self.find_kernel_roots(&*r.a, &mut nested_inputs);
				self.roots.insert(e, self.roots_postorder.len());
				inputs.push(self.roots_postorder.len());
				self.roots_postorder.push((e, nested_inputs));
			},
		}
	}

	fn gen_kernels(&self) -> Vec<CPUKernel> {
		let mut result = Vec::new();
		for root in 0..self.roots_postorder.len() {
			let name = format!("kernel_{}", root);

			let (root, inputs) = &self.roots_postorder[root];
			// SAFETY: as long as `expr` is not dropped, the pointers are valid
			let root = unsafe { &**root };

			if let ExprKind::Leaf(LeafExpr::Randn()) = &root.kind {
				result.push(CPUKernel { name, code: format!("RANDN_TODO;\n") });
				continue;
			}

			let ndim = root.shape.ndim();
			let mut code = format!("void {}(\n", name);
			for i in 0..ndim {
				write!(code, "\tstd::size_t const dim_{},\n", i);
			}
			for i in 0..inputs.len() {
				write!(code, "\tfloat const *const kernel_{},\n", inputs[i]);
			}
			write!(code, "\tfloat *const out_ptr\n) {{\n");
			write!(code, "\tstd::size_t elems = 1;\n");
			for i in (0..ndim).rev() {
				write!(
					code,
					"\tstd::size_t const stride_{} = elems; elems *= dim_{};\n",
					i, i
				);
			}
			code.push_str(self.gen_root_expr(root).as_str());
			code.push_str("}\n");

			result.push(CPUKernel { name, code });
		}
		result
	}

	fn gen_root_expr(&self, root: &Expr) -> String {
		let shape = root.shape.as_ref();

		let ndim = shape.ndim();

		let mut code = String::new();
		let mut index = Index::new();
		for dim in 0..ndim {
			#[rustfmt::skip] write!(
				code, "{}\tfor (std::size_t i_{} = 0; i_{} < dim_{}; ++i_{}) {{\n",
				Indent(dim), dim, dim, dim, dim
			);
			index.dims.push(DimIndex {
				i: format!("i_{}", dim),
				stride: format!("stride_{}", dim),
			});
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
				let val = self.gen_expr(root, root, &index);
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

	fn gen_expr(&self, root: &Expr, expr: &Expr, index: &Index) -> String {
		if (root as *const Expr) != (expr as *const Expr)
			&& let Some(kernel_index) = self.roots.get(&(expr as *const Expr))
		{
			return format!("kernel_{}[{}]", kernel_index, index);
		}

		match &expr.kind {
			ExprKind::Leaf(LeafExpr::IntConst(c)) => format!("{}", c),
			ExprKind::Leaf(LeafExpr::UintConst(c)) => format!("{}", c),
			ExprKind::Leaf(LeafExpr::FloatConst(c)) => format!("{}", c),
			ExprKind::Leaf(LeafExpr::Read(t)) => {
				format!("READ_TENSOR_TODO")
			},
			ExprKind::Unary(un) => {
				let a = self.gen_expr(root, &un.a, index);
				match un.op {
					UnaryOp::Exp => format!("exp({})", a),
				}
			},
			ExprKind::Binary(bin) => {
				let a = self.gen_expr(root, &bin.a, index);
				let b = self.gen_expr(root, &bin.b, index);
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
