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

	pub fn new_transposed(&self, x1: usize, x2: usize) -> Rc<Self> {
		let mut new_dims = self.__dims.clone();
		new_dims.swap(x1, x2);
		Rc::new(Self { __dims: new_dims })
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
	MatMul(MatMulExpr),
	Transpose(TransposeExpr),
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

impl UnaryOp {
	pub fn symbol(&self) -> &'static str {
		match self {
			UnaryOp::Exp => "exp",
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
}

impl BinaryOp {
	pub fn symbol(&self) -> &'static str {
		match self {
			BinaryOp::Add => "+",
		}
	}
	pub fn is_commutative(&self) -> bool {
		match self {
			BinaryOp::Add => true,
		}
	}
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

impl ReductionOp {
	pub fn symbol(&self) -> &'static str {
		match self {
			ReductionOp::Sum => "SUM",
			ReductionOp::Max => "MAX",
		}
	}
}

pub struct ReductionExpr {
	pub a: Rc<Expr>,
	pub op: ReductionOp,

	pub dims_to_collapse: Vec<usize>,
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

fn reduction_op<A: ExprLike>(a: A, op: ReductionOp, mut dims: Vec<usize>) -> Rc<Expr> {
	let a = a.get();
	dims.sort_unstable();
	Rc::new(Expr {
		shape: Shape::new_scalar(),
		dtype: a.dtype,
		kind: ExprKind::Reduction(ReductionExpr { a, op, dims_to_collapse: dims }),
	})
}

pub fn sum<A: ExprLike>(a: A, dims: Vec<usize>) -> Rc<Expr> {
	reduction_op(a, ReductionOp::Sum, dims)
}

pub fn max<A: ExprLike>(a: A, dims: Vec<usize>) -> Rc<Expr> {
	reduction_op(a, ReductionOp::Max, dims)
}

pub fn matmul<A: ExprLike, B: ExprLike>(a: A, b: B) -> Rc<Expr> {
	let a = a.get();
	let b = b.get();
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

pub fn transpose<A: ExprLike>(a: A, x1: usize, x2: usize) -> Rc<Expr> {
	let a = a.get();
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

pub struct CPUDevice {
	name: String,
}

struct CPUTensorData {
	data: Box<[Cell<u64>]>,
}

struct CPUKernel {
	name: String,
	code: String,
	ndim: usize,
	param_count: usize,
}

#[derive(Debug, Clone)]
struct Index {
	perm: Vec<usize>,
	code: String,
}

impl Index {
	fn new(perm: Vec<usize>) -> Self {
		let mut code = String::new();

		for i in 0..perm.len() {
			if i != 0 {
				code.push_str(" + ");
			}
			write!(code, "i_{}", perm[i]).unwrap();
			for j in i + 1..perm.len() {
				write!(code, "*dim_{}", perm[j]).unwrap();
			}
		}

		Index { perm, code }
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
	roots_postorder: Vec<(*const Expr, String, Vec<usize>)>, // (expr, key, inputs)
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

		let mut parent_inputs = Vec::new();
		s.find_postorder(e, &mut parent_inputs);

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
			ExprKind::Leaf(LeafExpr::Randn()) | ExprKind::Leaf(LeafExpr::Read(..)) => {
				self.roots.insert(e, -1);
			},
			ExprKind::Leaf(..) => {
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
			ExprKind::MatMul(m) => {
				// TODO - we mark both inputs as roots,
				// which will make the common idiom `x * w.T` inefficient
				self.roots.insert(m.a.as_ref() as *const Expr, -1);
				self.roots.insert(m.b.as_ref() as *const Expr, -1);

				self.find_kernel_roots(&*m.a);
				self.find_kernel_roots(&*m.b);
				self.roots.insert(e, -1);
			},
			ExprKind::Transpose(t) => {
				self.find_kernel_roots(&*t.a);
			},
		}
	}

	fn trav_children(&mut self, expr: &Expr, parent_inputs: &mut Vec<usize>) -> String {
		match &expr.kind {
			ExprKind::Leaf(LeafExpr::IntConst(c)) => format!("int({})", c),
			ExprKind::Leaf(LeafExpr::UintConst(c)) => format!("uint({})", c),
			ExprKind::Leaf(LeafExpr::FloatConst(c)) => format!("float({})", c),
			ExprKind::Unary(un) => {
				let a = self.find_postorder(&*un.a, parent_inputs);
				format!("{}({})", un.op.symbol(), a)
			},
			ExprKind::Binary(bin) => {
				let i = parent_inputs.len();
				let mut a = self.find_postorder(&*bin.a, parent_inputs);
				let j = parent_inputs.len();
				let mut b = self.find_postorder(&*bin.b, parent_inputs);
				if bin.op.is_commutative() && a > b {
					// swap a, b
					std::mem::swap(&mut a, &mut b);
					// swap inputs
					parent_inputs[i..].rotate_left(j - i);
				}
				format!("{}{}{}", a, bin.op.symbol(), b)
			},
			ExprKind::Leaf(LeafExpr::Randn()) | ExprKind::Leaf(LeafExpr::Read(..)) => {
				"i".to_string()
			},
			ExprKind::Reduction(r) => {
				let a = self.find_postorder(&*r.a, parent_inputs);
				format!("{}({})", r.op.symbol(), a)
			},
			ExprKind::MatMul(m) => {
				let a = self.find_postorder(&*m.a, parent_inputs);
				let b = self.find_postorder(&*m.b, parent_inputs);
				format!("matmul({},{})", a, b)
			},
			ExprKind::Transpose(t) => {
				let a = self.find_postorder(&*t.a, parent_inputs);
				format!(
					"transpose({},{},{})",
					a,
					std::cmp::min(t.x1, t.x2),
					std::cmp::max(t.x1, t.x2)
				)
			},
		}
	}

	fn find_postorder(&mut self, expr: *const Expr, parent_inputs: &mut Vec<usize>) -> String {
		let expr_ptr = expr;
		let expr = unsafe { &*expr_ptr };

		// Check if `expr` is in the `roots` set
		if let Some(entry) = self.roots.get(&expr_ptr) {
			// `expr` is a root. The value stored in `entry` is:
			// - negative if not yet processed, or
			// - an index into `roots_postorder` if already processed
			let index = *entry;
			if index >= 0 {
				// already processed - just add `expr` as a an input to the parent
				parent_inputs.push(index as usize);
			} else {
				// not yet processed - process it
				let mut my_inputs = Vec::new();

				let key = self.trav_children(expr, &mut my_inputs);

				parent_inputs.push(self.roots_postorder.len());
				self.roots.insert(expr, self.roots_postorder.len() as isize);
				self.roots_postorder.push((expr, key, my_inputs));
			}

			"i".to_string()
		} else {
			// `expr` is NOT a root - process its children
			let key = self.trav_children(expr, parent_inputs);
			key
		}
	}

	fn gen_kernels(&self) -> Vec<CPUKernel> {
		let mut result = Vec::new();
		for kernel_index in 0..self.roots_postorder.len() {
			let (root, _key, inputs) = &self.roots_postorder[kernel_index];
			// TODO - use `key` to try to find the kernel in the cache

			let name = format!("kernel_{}", kernel_index);

			// SAFETY: as long as `expr` is not dropped, the pointers are valid
			let root = unsafe { &**root };
			let ndim = root.shape.ndim();

			if let ExprKind::Leaf(LeafExpr::Randn()) = &root.kind {
				debug_assert!(inputs.is_empty());
				result.push(CPUKernel {
					name,
					code: format!("RANDN_TODO;\n"),
					ndim,
					param_count: 0,
				});
				continue;
			}

			let mut code = format!("void {}(\n", name);
			for i in 0..ndim {
				write!(code, "\tuintptr_t const dim_{},\n", i);
			}
			for i in 0..inputs.len() {
				write!(code, "\tfloat const *const kernel_{},\n", inputs[i]);
			}
			write!(code, "\tfloat *const out_ptr\n) {{\n");

			code.push_str(
				match &root.kind {
					ExprKind::Reduction(red) => self.gen_kernel_reduction(root, red),
					ExprKind::MatMul(..) => self.gen_kernel_matmul(root),
					_ => self.gen_kernel_pointwise(root),
				}
				.as_str(),
			);

			code.push_str("}\n");

			result.push(CPUKernel {
				name,
				code,
				ndim,
				param_count: inputs.len(),
			});
		}
		result
	}

	fn gen_kernel_pointwise(&self, root: &Expr) -> String {
		let shape = root.shape.as_ref();
		let ndim = shape.ndim();

		let mut code = String::new();
		let index = Index::new((0..ndim).collect());
		write!(code, "\t// pointwise kernel\n");
		for dim in 0..ndim {
			#[rustfmt::skip] write!(
				code, "{}\tfor (uintptr_t i_{} = 0; i_{} < dim_{}; ++i_{}) {{\n",
				Indent(dim), dim, dim, dim, dim
			);
		}

		self.gen_root_expr(
			&mut code,
			Indent(ndim + 1),
			root,
			&index,
			format!("out_ptr[{}]", index.code),
		);

		for dim in (0..ndim).rev() {
			for _ in 0..dim {
				code.push('\t');
			}
			write!(code, "\t}}\n");
		}

		code
	}

	fn gen_kernel_reduction(&self, root: &Expr, red: &ReductionExpr) -> String {
		let shape = root.shape.as_ref();
		let ndim = shape.ndim();

		let mut code = String::new();
		let index = Index::new((0..ndim).collect());
		write!(code, "\t// reduction kernel\n");

		let mut loop_cnt = 0;
		for dim in 0..ndim {
			// TODO - use binary_search()? ndim will usually be very small, so contains() may be faster
			if red.dims_to_collapse.contains(&dim) {
				continue;
			}
			#[rustfmt::skip] write!(
				code, "{}\tfor (uintptr_t i_{} = 0; i_{} < dim_{}; ++i_{}) {{\n",
				Indent(loop_cnt), dim, dim, dim, dim
			);
			loop_cnt += 1;
		}

		let outer_loops = loop_cnt;
		write!(code, "{}\tfloat value = 0.0;\n", Indent(loop_cnt));

		for dim in 0..ndim {
			// TODO - use binary_search()? ndim will usually be very small, so contains() may be faster
			if !red.dims_to_collapse.contains(&dim) {
				continue;
			}
			#[rustfmt::skip] write!(
				code, "{}\tfor (uintptr_t i_{} = 0; i_{} < dim_{}; ++i_{}) {{\n",
				Indent(loop_cnt), dim, dim, dim, dim
			);
			loop_cnt += 1;
		}

		self.gen_root_expr(
			&mut code,
			Indent(loop_cnt + 1),
			red.a.as_ref(),
			&index,
			"value".to_string(),
		);

		for dim in (0..outer_loops).rev() {
			write!(code, "{}\t}}\n", Indent(loop_cnt));
			loop_cnt -= 1;
		}

		write!(code, "out_ptr[{}] = value;", index.code);

		for dim in (outer_loops..ndim).rev() {
			write!(code, "{}\t}}\n", Indent(loop_cnt));
			loop_cnt -= 1;
		}

		code
	}

	fn gen_kernel_matmul(&self, root: &Expr) -> String {
		unimplemented!("gen_kernel_matmul");
	}

	fn gen_root_expr(
		&self,
		code: &mut String,
		indent: Indent,
		root: &Expr,
		index: &Index,
		outvar: String,
	) {
		match &root.kind {
			ExprKind::Leaf(LeafExpr::Randn()) => {
				unreachable!("Randn should have been handled by gen_kernels()")
			},
			ExprKind::Reduction(r) => {
				unimplemented!("Reduction");
			},
			ExprKind::MatMul(m) => {
				unimplemented!("MatMul");
			},
			_ => {
				let val = self.gen_expr(root, root, index);
				write!(code, "{}{} = {};\n", indent, outvar, val);
			},
		}
	}

	fn gen_expr(&self, root: &Expr, expr: &Expr, index: &Index) -> String {
		if (root as *const Expr) != (expr as *const Expr)
			&& let Some(kernel_index) = self.roots.get(&(expr as *const Expr))
		{
			return format!("kernel_{}[{}]", kernel_index, index.code);
		}

		match &expr.kind {
			ExprKind::Leaf(LeafExpr::IntConst(c)) => format!("int({})", c),
			ExprKind::Leaf(LeafExpr::UintConst(c)) => format!("uint({})", c),
			ExprKind::Leaf(LeafExpr::FloatConst(c)) => format!("float({})", c),
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
					BinaryOp::Add => format!("({} + {})", a, b),
				}
			},
			ExprKind::Transpose(t) => {
				let mut new_perm = index.perm.clone();
				new_perm.swap(t.x1, t.x2);
				let new_index = Index::new(new_perm);
				self.gen_expr(root, &t.a, &new_index)
			},
			_ => {
				panic!("Unsupported expression");
			},
		}
	}
}
