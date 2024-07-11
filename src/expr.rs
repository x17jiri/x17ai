// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

extern crate libloading;
extern crate tempfile;

#[cfg(unix)]
use libloading::os::unix::Symbol as RawSymbol;
#[cfg(windows)]
use libloading::os::windows::Symbol as RawSymbol;

use bit_set::BitSet;
use core::fmt;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::path::Path;
use std::rc::{Rc, Weak};

use crate::rand::Rng;

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
	// TODO - could use some sort of small vec optimization
	__dims: Box<[usize]>,
	__elems: usize,
}

impl Shape {
	pub fn new_scalar() -> Rc<Self> {
		Rc::new(Self {
			__dims: Vec::new().into_boxed_slice(),
			__elems: 1,
		})
	}

	pub fn new(dims: &[usize]) -> Rc<Self> {
		let elems = dims.iter().product();
		Rc::new(Self { __dims: dims.into(), __elems: elems })
	}

	pub fn new_transposed(&self, x1: usize, x2: usize) -> Rc<Self> {
		let mut new_dims = self.__dims.clone();
		new_dims.swap(x1, x2);
		Rc::new(Self { __dims: new_dims, __elems: self.__elems })
	}

	pub fn new_reduced(&self, dims_to_reduce: &[usize]) -> Rc<Self> {
		let mut new_dims = self.__dims.clone();
		for dim in dims_to_reduce {
			if *dim >= new_dims.len() {
				panic!("Invalid dimension");
			}
			new_dims[*dim] = 1;
		}
		let elems = new_dims.iter().product();
		Rc::new(Self { __dims: new_dims, __elems: elems })
	}

	pub fn ndim(&self) -> usize {
		self.__dims.len()
	}

	pub fn dims(&self) -> &[usize] {
		&self.__dims
	}

	pub fn elems(&self) -> usize {
		self.__elems
	}
}

pub trait Device {
	fn name(&self) -> &str;

	fn eval(self: Rc<Self>, expr: Rc<Expr>) -> Rc<Tensor>;

	fn owns(&self, tensor: &Tensor) -> bool;
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

impl fmt::Display for DType {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			DType::Float(b) => write!(f, "f{}", b),
			DType::Int(b) => write!(f, "i{}", b),
			DType::Uint(b) => write!(f, "u{}", b),
		}
	}
}

pub struct Expr {
	pub shape: Rc<Shape>,
	pub dtype: DType,
	pub kind: ExprKind,
}

pub enum ExprKind {
	Read(Rc<Tensor<dyn TensorData>>),
	Randn(),
	Const(ConstExpr),
	Unary(UnaryExpr),
	Binary(BinaryExpr),
	Reduction(ReductionExpr),
	MatMul(MatMulExpr),
	Transpose(TransposeExpr),
}

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
pub enum ReductionOp {
	Sum,
	Max,
	Min,
}

impl ReductionOp {
	pub fn symbol(&self) -> &'static str {
		match self {
			ReductionOp::Sum => "SUM",
			ReductionOp::Max => "MAX",
			ReductionOp::Min => "MIN",
		}
	}
}

pub struct ReductionExpr {
	pub a: Rc<Expr>,
	pub op: ReductionOp,

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

pub fn zeros(shape: Rc<Shape>, dtype: DType) -> Rc<Expr> {
	let c = match dtype {
		DType::Float(_) => ConstExpr::Float(0.0),
		DType::Int(_) => ConstExpr::Int(0),
		DType::Uint(_) => ConstExpr::Uint(0),
	};
	Rc::new(Expr { shape, dtype, kind: ExprKind::Const(c) })
}

pub fn randn(shape: Rc<Shape>, dtype: DType) -> Rc<Expr> {
	Rc::new(Expr { shape, dtype, kind: ExprKind::Randn() })
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
			kind: ExprKind::Read(self),
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

pub fn sqrt<A: ExprLike>(a: A) -> Rc<Expr> {
	unary_op(a, UnaryOp::Sqrt)
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

pub fn sub<A: ExprLike, B: ExprLike>(a: A, b: B) -> Rc<Expr> {
	binary_op(a, b, BinaryOp::Sub)
}

pub fn mul<A: ExprLike, B: ExprLike>(a: A, b: B) -> Rc<Expr> {
	binary_op(a, b, BinaryOp::Mul)
}

pub fn div<A: ExprLike, B: ExprLike>(a: A, b: B) -> Rc<Expr> {
	binary_op(a, b, BinaryOp::Div)
}

fn reduction_op<A: ExprLike>(a: A, op: ReductionOp, dims_to_reduce: &[usize]) -> Rc<Expr> {
	let a = a.get();
	let mut bitset = BitSet::with_capacity(a.shape.ndim());
	for dim in dims_to_reduce {
		bitset.insert(*dim);
	}
	Rc::new(Expr {
		shape: a.shape.new_reduced(dims_to_reduce),
		dtype: a.dtype,
		kind: ExprKind::Reduction(ReductionExpr { a, op, dims_to_reduce: bitset }),
	})
}

pub fn sum<A: ExprLike>(a: A, dims_to_reduce: &[usize]) -> Rc<Expr> {
	reduction_op(a, ReductionOp::Sum, dims_to_reduce)
}

pub fn max<A: ExprLike>(a: A, dims_to_reduce: &[usize]) -> Rc<Expr> {
	reduction_op(a, ReductionOp::Max, dims_to_reduce)
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
	kernel_cache: HashMap<String, Rc<CPUKernel>>,
	tempdir: tempfile::TempDir,
	kernel_counter: usize,
	rng: Cell<Rng>,
}

struct CPUTensorData {
	__data: Box<[Cell<u64>]>,
}

impl CPUTensorData {
	fn cast<T>(&self) -> &[Cell<T>] {
		let bytes = self.__data.len() * std::mem::size_of::<u64>();
		let elems = bytes / std::mem::size_of::<T>();
		let ptr = self.__data.as_ptr() as *const Cell<T>;
		unsafe { std::slice::from_raw_parts(ptr, elems) }
	}
}

impl TensorData for CPUTensorData {}

type CPUKernelFunc =
	unsafe extern "C" fn(dims: *const usize, inputs: *const *const f32, output: *mut f32) -> ();

struct CPUKernel {
	ndim: usize,
	param_count: usize,
	lib: libloading::Library,
	func: RawSymbol<CPUKernelFunc>,
}

impl CPUKernel {
	fn new(code: &str, ndim: usize, param_count: usize, tempdir: &Path) -> Self {
		println!("compiling kernel:\n{}", code);

		let cpp_file = tempdir.join("kernel.cpp");
		std::fs::write(&cpp_file, code).unwrap();

		let so_file = tempdir.join("kernel.so");

		std::io::stdout().flush().unwrap();
		std::io::stderr().flush().unwrap();

		let mut child =
			std::process::Command::new("/home/spock/sw/llvm-project-2/llvm-build/bin/clang")
				.arg("-std=c++17")
				.arg("-Wall")
				.arg("-g")
				.arg("-ggdb")
				.arg("-O3")
				.arg("-shared")
				.arg("-o")
				.arg(so_file.as_os_str())
				.arg(cpp_file.as_os_str())
				.current_dir(tempdir)
				.spawn()
				.expect("Failed to spawn clang");

		let ok = child.wait().expect("Failed to compile kernel").success();
		if !ok {
			panic!("Failed to compile kernel");
		}

		let lib = unsafe { libloading::Library::new(so_file).unwrap() };
		let func: libloading::Symbol<CPUKernelFunc> = unsafe { lib.get(b"kernel").unwrap() };
		let func: RawSymbol<CPUKernelFunc> = unsafe { func.into_raw() };

		Self { ndim, param_count, lib, func }
	}
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
		Rc::<CPUDevice>::new(CPUDevice {
			name,
			kernel_cache: HashMap::new(),
			tempdir: tempfile::tempdir().unwrap(),
			kernel_counter: 0,
			rng: Cell::new(Rng::new_default()),
		})
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
		CPUTensorData { __data: Box::from_raw(slice) }
	}

	fn gen_kernel(&self, sequence: &ComputeSequence, item: &PostorderItem) -> CPUKernel {
		let root = item.expr;
		let inputs = item.inputs;
		let ndim = root.shape.ndim();

		let mut code = String::new();
		writeln!(code, "#include <cstdint>");
		writeln!(code);
		writeln!(code, "using Index = uintptr_t;");
		writeln!(code, "using Count = uintptr_t;");
		writeln!(code);
		writeln!(code, "using f32 = float;");
		writeln!(code, "using f64 = double;");
		writeln!(code);
		writeln!(code, "// cache key: `{}`", item.cache_key);
		writeln!(code, "extern \"C\" void kernel(");
		writeln!(
			code,
			"\tCount const *dims, // array of {} dimension sizes",
			ndim
		);
		writeln!(
			code,
			"\tvoid const * const *inputs, // array of {} input tensors",
			inputs.len()
		);
		writeln!(code, "\t{} *output // output tensor", root.dtype);
		writeln!(code, ") {{");

		for i in 0..ndim {
			writeln!(code, "\tCount const dim_{} = dims[{}];", i, i);
		}
		writeln!(code);
		for i in 0..inputs.len() {
			let dtype = sequence.item_dtype(i);
			writeln!(
				code,
				"\t{} const * const input_{} = reinterpret_cast<{} const *>(inputs[{}]);",
				dtype, i, dtype, i
			);
		}
		writeln!(code);

		match &root.kind {
			ExprKind::Reduction(red) => {
				self.gen_kernel_reduction(&mut code, sequence, root, red);
			},
			ExprKind::MatMul(..) => {
				self.gen_kernel_matmul(&mut code, sequence, root);
			},
			_ => {
				self.gen_kernel_pointwise(&mut code, sequence, root);
			},
		}

		writeln!(code, "}}");

		CPUKernel::new(&code, ndim, inputs.len(), self.tempdir.path())
	}

	fn gen_kernel_pointwise(&self, code: &mut String, sequence: &ComputeSequence, root: &Expr) {
		let ndim = root.shape.ndim();

		let index = Index::new((0..ndim).collect());
		writeln!(code, "\t// pointwise kernel");
		for dim in 0..ndim {
			#[rustfmt::skip] writeln!(
				code, "{}\tfor (Index i_{} = 0; i_{} < dim_{}; ++i_{}) {{",
				Indent(dim), dim, dim, dim, dim
			);
		}

		write!(code, "{}\toutput = ", Indent(ndim));
		let mut input_counter = 0;
		self.gen_expr(code, sequence, root, root, &index, &mut input_counter);
		writeln!(code, ";");

		for dim in (0..ndim).rev() {
			for _ in 0..dim {
				code.push('\t');
			}
			writeln!(code, "\t}}");
		}
	}

	fn gen_kernel_reduction(
		&self,
		code: &mut String,
		sequence: &ComputeSequence,
		root: &Expr,
		red: &ReductionExpr,
	) {
		let ndim = root.shape.ndim();

		let index = Index::new((0..ndim).collect());
		writeln!(code, "\t// reduction kernel");

		let mut loop_cnt = 0;
		for dim in 0..ndim {
			// TODO - use binary_search()? ndim will usually be very small, so contains() may be faster
			if red.dims_to_reduce.contains(dim) {
				continue;
			}
			#[rustfmt::skip] writeln!(
				code, "{}\tfor (Index i_{} = 0; i_{} < dim_{}; ++i_{}) {{",
				Indent(loop_cnt), dim, dim, dim, dim
			);
			loop_cnt += 1;
		}

		let outer_loops = loop_cnt;
		// TODO: `0.0` is only correct for `sum`
		writeln!(
			code,
			"{}\t{} value = {}(0);",
			Indent(loop_cnt),
			root.dtype,
			root.dtype
		);

		for dim in 0..ndim {
			// TODO - use binary_search()? ndim will usually be very small, so contains() may be faster
			if !red.dims_to_reduce.contains(dim) {
				continue;
			}
			#[rustfmt::skip] writeln!(
				code, "{}\tfor (Index i_{} = 0; i_{} < dim_{}; ++i_{}) {{",
				Indent(loop_cnt), dim, dim, dim, dim
			);
			loop_cnt += 1;
		}

		// TODO: `+=` is only correct for `sum`
		write!(code, "{}\tvalue += ", Indent(loop_cnt));
		let mut input_counter = 0;
		self.gen_expr(code, sequence, root, &red.a, &index, &mut input_counter);
		writeln!(code, ";");

		for _ in (outer_loops..ndim).rev() {
			loop_cnt -= 1;
			writeln!(code, "{}\t}}", Indent(loop_cnt));
		}

		for i in &red.dims_to_reduce {
			writeln!(
				code,
				"{}\tCount dim_{} = 1; Index i_{} = 0;",
				Indent(outer_loops),
				i,
				i
			);
		}

		writeln!(
			code,
			"{}\toutput[{}] = value;",
			Indent(loop_cnt),
			index.code
		);

		for _ in (0..outer_loops).rev() {
			loop_cnt -= 1;
			writeln!(code, "{}\t}}", Indent(loop_cnt));
		}
	}

	fn gen_kernel_matmul(&self, _code: &mut String, _sequence: &ComputeSequence, _root: &Expr) {
		unimplemented!("gen_kernel_matmul");
	}

	fn gen_expr(
		&self,
		code: &mut String,
		sequence: &ComputeSequence,
		root: &Expr,
		expr: &Expr,
		index: &Index,
		input_counter: &mut usize,
	) {
		if (root as *const Expr) != (expr as *const Expr) && sequence.is_root(expr) {
			write!(code, "input_{}[{}]", input_counter, index.code).unwrap();
			*input_counter += 1;
			return;
		}

		match &expr.kind {
			ExprKind::Const(c) => {
				write!(code, "{}({})", expr.dtype, c).unwrap();
			},
			ExprKind::Read(t) => {
				write!(code, "READ_TENSOR_TODO").unwrap();
			},
			ExprKind::Unary(un) => {
				write!(code, "{}(", un.op.symbol()).unwrap();
				self.gen_expr(code, sequence, root, &un.a, index, input_counter);
				write!(code, ")").unwrap();
			},
			ExprKind::Binary(bin) => {
				write!(code, "(").unwrap();
				if bin.op.is_commutative() && sequence.has_swapped_operands(expr) {
					self.gen_expr(code, sequence, root, &bin.a, index, input_counter);
					write!(code, " {} ", bin.op.symbol()).unwrap();
					self.gen_expr(code, sequence, root, &bin.b, index, input_counter);
				} else {
					self.gen_expr(code, sequence, root, &bin.b, index, input_counter);
					write!(code, " {} ", bin.op.symbol()).unwrap();
					self.gen_expr(code, sequence, root, &bin.a, index, input_counter);
				}
				write!(code, ")").unwrap();
			},
			ExprKind::Transpose(t) => {
				let mut new_perm = index.perm.clone();
				new_perm.swap(t.x1, t.x2);
				let new_index = Index::new(new_perm);
				self.gen_expr(code, sequence, root, &t.a, &new_index, input_counter)
			},
			_ => {
				panic!("Unsupported expression");
			},
		}
	}

	fn randn(&self, data: &mut CPUTensorData) {
		for i in data.cast::<f32>() {
			let rng = self.rng.as_ptr();
			let rng = unsafe { &mut *rng };
			i.set(rng.get_normal() as f32);
		}
	}
}

impl Device for CPUDevice {
	fn name(&self) -> &str {
		&self.name
	}

	fn eval(self: Rc<CPUDevice>, expr: Rc<Expr>) -> Rc<Tensor> {
		let sequence = ComputeSequence::new(expr);
		let mut buffers: Vec<CPUTensorData> = Vec::with_capacity(sequence.len());
		let mut rc_buffers: Vec<(usize, *const CPUTensorData)> = Vec::with_capacity(sequence.len());
		for item in sequence.iter() {
			let expr = item.expr;
			match &expr.kind {
				ExprKind::Read(tensor) => {
					if !self.owns(&tensor) {
						unimplemented!("TODO: copy tensor from another device");
					}
					buffers.push(CPUTensorData { __data: Vec::new().into_boxed_slice() });

					let tensor_data = &tensor.data;
					let tensor_data = tensor_data as *const dyn TensorData;
					let tensor_data = tensor_data as *const CPUTensorData;

					let rc = usize::MAX;
					rc_buffers.push((rc, tensor_data));
				},
				ExprKind::Randn() => {
					let rc = item.ref_count;
					match expr.dtype {
						DType::Float(32) => {
							buffers.push(unsafe { CPUDevice::new_uninit(32, expr.shape.elems()) });

							let new_buf = buffers.last().unwrap() as *const CPUTensorData;
							rc_buffers.push((rc, new_buf));

							self.randn(buffers.last_mut().unwrap());
						},
						_ => panic!("Unsupported dtype for randn"),
					};
				},
				_ => {
					let rc = item.ref_count;
					buffers.push(unsafe {
						CPUDevice::new_uninit(expr.dtype.bits(), expr.shape.elems())
					});
					rc_buffers.push((rc, buffers.last().unwrap() as *const CPUTensorData));

					// TODO - try to find the kernel in the cache
					let kernel = self.gen_kernel(&sequence, &item);

					let dims = expr.shape.dims();
					let dims = dims.as_ptr();

					let inputs = item
						.inputs
						.iter()
						.map(|i| {
							let tensor_data = unsafe { &*rc_buffers[*i].1 };
							let tensor_data = tensor_data.cast::<f32>();
							let tensor_data = tensor_data.as_ptr();
							tensor_data as *const f32
						})
						.collect::<Vec<_>>();
					let inputs = inputs.as_ptr();

					let output = buffers.last().unwrap();
					let output = output.cast::<f32>();
					let output = output.as_ptr();
					let output = output as *mut f32;

					unsafe {
						(kernel.func)(dims, inputs, output);
					}

					// TODO - decrement ref counts of inputs
				},
			}
		}

		let result = Rc::new(Tensor::<CPUTensorData> {
			shape: sequence.expr.shape.clone(),
			dtype: sequence.expr.dtype,
			device: self,
			data: buffers.pop().unwrap(),
		});

		result
	}

	fn owns(&self, tensor: &Tensor) -> bool {
		(tensor.device.as_ref() as *const dyn Device) == (self as *const CPUDevice)
	}
}

struct _PostorderItem {
	expr: *const Expr,
	cache_key: String,
	inputs: Vec<usize>,
	ref_count: usize,
}

struct PostorderItem<'e> {
	expr: &'e Expr,
	cache_key: &'e str,
	inputs: &'e [usize],
	ref_count: usize,
}

struct ComputeSequence {
	expr: Rc<Expr>,

	// SAFETY: We have a shared ownership of `expr` and since it is immutable,
	// as long as we don't drop it, all pointers to sub-expressions are valid.
	roots: HashMap<*const Expr, isize>,
	postorder: Vec<_PostorderItem>,
	processed: HashSet<*const Expr>,

	swapped_operands: HashSet<*const Expr>,
}

struct ComputeSequenceIter<'a> {
	postorder_iter: std::slice::Iter<'a, _PostorderItem>,
}

impl<'e> Iterator for ComputeSequenceIter<'e> {
	type Item = PostorderItem<'e>;

	fn next(&mut self) -> Option<Self::Item> {
		self.postorder_iter.next().map(|item| PostorderItem {
			expr: unsafe { &*item.expr },
			cache_key: &item.cache_key,
			inputs: &item.inputs,
			ref_count: item.ref_count,
		})
	}
}

impl<'e> ExactSizeIterator for ComputeSequenceIter<'e> {
	fn len(&self) -> usize {
		self.postorder_iter.len()
	}
}

impl<'e> IntoIterator for &'e ComputeSequence {
	type Item = PostorderItem<'e>;
	type IntoIter = ComputeSequenceIter<'e>;

	fn into_iter(self) -> Self::IntoIter {
		ComputeSequenceIter { postorder_iter: self.postorder.iter() }
	}
}

impl ComputeSequence {
	pub fn new(expr: Rc<Expr>) -> Self {
		let e = expr.as_ref() as *const Expr;
		let mut result = ComputeSequence {
			expr,
			roots: HashMap::new(),
			postorder: Vec::new(),
			processed: HashSet::new(),
			swapped_operands: HashSet::new(),
		};

		result.find_kernel_roots(unsafe { &*e });
		result.roots.insert(e, -1);

		let mut parent_inputs = Vec::new();
		result.find_postorder(e, &mut parent_inputs);

		result
	}

	pub fn item_dtype(&self, index: usize) -> DType {
		let item = &self.postorder[index];
		unsafe { (*item.expr).dtype }
	}

	pub fn len(&self) -> usize {
		self.postorder.len()
	}

	pub fn iter(&self) -> ComputeSequenceIter {
		self.into_iter()
	}

	fn find_kernel_roots(&mut self, expr: &Expr) {
		let e = expr as *const Expr;
		if self.processed.contains(&e) {
			self.roots.insert(e, -1);
			return;
		}
		self.processed.insert(e);

		match &expr.kind {
			ExprKind::Read(..) | ExprKind::Randn() => {
				self.roots.insert(e, -1);
			},
			ExprKind::Const(..) => {
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

				self.roots.insert(r.a.as_ref() as *const Expr, -1);
				self.roots.insert(e, -1);
			},
			ExprKind::MatMul(m) => {
				self.find_kernel_roots(&*m.a);
				self.find_kernel_roots(&*m.b);

				// TODO - we mark both inputs as roots,
				// which will make the common idiom `x * w.T` inefficient
				self.roots.insert(m.a.as_ref() as *const Expr, -1);
				self.roots.insert(m.b.as_ref() as *const Expr, -1);
				self.roots.insert(e, -1);
			},
			ExprKind::Transpose(t) => {
				self.find_kernel_roots(&*t.a);
			},
		}
	}

	fn traverse_children(&mut self, expr: &Expr, parent_inputs: &mut Vec<usize>) -> String {
		match &expr.kind {
			ExprKind::Read(..) | ExprKind::Randn() => {
				// Read and Randn are handled separately, so their cache_key is never used
				String::new()
			},
			ExprKind::Const(c) => format!("{}({})", expr.dtype, c),
			ExprKind::Unary(un) => {
				let a = self.find_postorder(&*un.a, parent_inputs);
				format!("{}_{}({})", un.op.symbol(), expr.dtype, a)
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
					self.swapped_operands.insert(expr);
				}
				format!("{}{}{}", a, bin.op.symbol(), b)
			},
			ExprKind::Reduction(r) => {
				self.find_postorder(&*r.a, parent_inputs);

				// Reductions are handled separately, so their cache_key is never used
				String::new()
			},
			ExprKind::MatMul(m) => {
				self.find_postorder(&*m.a, parent_inputs);
				self.find_postorder(&*m.b, parent_inputs);

				// MatMuls are handled separately, so their cache_key is never used
				String::new()
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
			// - an index into `postorder` if already processed
			let index = *entry;
			if index >= 0 {
				// already processed - just add `expr` as an input to the parent
				self.postorder[index as usize].ref_count += 1;
				parent_inputs.push(index as usize);
			} else {
				// not yet processed - process it
				let mut my_inputs = Vec::new();

				let cache_key = self.traverse_children(expr, &mut my_inputs);
				let cache_key = format!("{}D:{}", expr.shape.ndim(), cache_key);

				parent_inputs.push(self.postorder.len());
				self.roots.insert(expr, self.postorder.len() as isize);
				self.postorder.push(_PostorderItem {
					expr: expr_ptr,
					cache_key,
					inputs: my_inputs,
					ref_count: 1,
				});
			}

			format!("i_{}", expr.dtype)
		} else {
			// `expr` is NOT a root - process its children
			let cache_key = self.traverse_children(expr, parent_inputs);
			cache_key
		}
	}

	pub fn is_root(&self, expr: &Expr) -> bool {
		let expr = expr as *const Expr;
		self.roots.contains_key(&expr)
	}

	pub fn has_swapped_operands(&self, expr: &Expr) -> bool {
		let expr = expr as *const Expr;
		self.swapped_operands.contains(&expr)
	}
}
