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

pub enum BroadcastType {
	Error,
	NoBroadcast,
	Broadcast(bool, bool, Rc<Shape>),
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

	pub fn broadcast_type(&self, other: &Self) -> BroadcastType {
		let ndim = std::cmp::min(self.ndim(), other.ndim());
		let a = &self.__dims[self.ndim() - ndim..];
		let b = &other.__dims[other.ndim() - ndim..];
		let mut a_broadcast = false;
		let mut b_broadcast = false;
		for i in 0..ndim {
			if a[i] != b[i] {
				if a[i] == 1 {
					a_broadcast = true;
				} else if b[i] == 1 {
					b_broadcast = true;
				} else {
					return BroadcastType::Error;
				}
			}
		}
		if a_broadcast || b_broadcast {
			let ndim = self.ndim().max(other.ndim());

			// Number of dimensions of len 1 to add to each shape
			let a1 = ndim - self.ndim();
			let b1 = ndim - other.ndim();

			let mut new_shape = vec![0; ndim];
			let mut new_elems = 1;
			for i in 0..ndim {
				let a_dim = if i < a1 { 1 } else { self.__dims[i - a1] };
				let b_dim = if i < b1 { 1 } else { other.__dims[i - b1] };
				new_shape[i] = std::cmp::max(a_dim, b_dim);
				new_elems *= new_shape[i]; // TODO - check for overflow
			}

			BroadcastType::Broadcast(
				a_broadcast,
				b_broadcast,
				Rc::new(Self {
					__dims: new_shape.into_boxed_slice(),
					__elems: new_elems,
				}),
			)
		} else {
			BroadcastType::NoBroadcast
		}
	}
}

impl fmt::Display for Shape {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "[")?;
		for i in 0..self.__dims.len() {
			if i != 0 {
				write!(f, ", ")?;
			}
			write!(f, "{}", self.__dims[i])?;
		}
		write!(f, "]")
	}
}

pub trait Device {
	fn name(&self) -> &str;

	fn eval(self: Rc<Self>, expr: Rc<Expr>, dotfile: Option<&str>) -> Rc<Tensor>;

	fn owns(&self, tensor: &Tensor) -> bool;

	fn format(
		&self,
		f: &mut fmt::Formatter,
		tensor: &Tensor,
		off: usize,
		len: usize,
		stride: isize,
	) -> fmt::Result;
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

impl Expr {
	fn __draw(&self, file: &mut std::fs::File, roots: Option<&HashMap<*const Expr, isize>>) {
		let color = if let Some(roots) = roots
			&& roots.contains_key(&(self as *const Expr))
		{
			", fillcolor=lightblue, style=filled"
		} else {
			""
		};
		// using node pointers as IDs
		match self.kind {
			ExprKind::Read(ref tensor) => {
				writeln!(file, "\t\"{:p}\" [label=\"in\"{}]", self, color);
			},
			ExprKind::Randn() => {
				writeln!(file, "\t\"{:p}\" [label=\"randn\"{}]", self, color);
			},
			ExprKind::Const(ref c) => {
				writeln!(file, "\t\"{:p}\" [label=\"const({})\"{}]", self, c, color);
			},
			ExprKind::Unary(ref u) => {
				writeln!(
					file,
					"\t\"{:p}\" [label=\"{}\"{}]",
					self,
					u.op.symbol(),
					color
				);
				writeln!(file, "\t\"{:p}\" -> \"{:p}\"", u.a, self);
				u.a.__draw(file, roots);
			},
			ExprKind::Binary(ref b) => {
				writeln!(
					file,
					"\t\"{:p}\" [label=\"{}\"{}]",
					self,
					b.op.symbol(),
					color
				);
				writeln!(file, "\t\"{:p}\" -> \"{:p}\"", b.a, self);
				writeln!(file, "\t\"{:p}\" -> \"{:p}\"", b.b, self);
				b.a.__draw(file, roots);
				b.b.__draw(file, roots);
			},
			ExprKind::Reduction(ref r) => {
				writeln!(
					file,
					"\t\"{:p}\" [label=\"{}\"{}]",
					self,
					r.op.symbol(),
					color
				);
				writeln!(file, "\t\"{:p}\" -> \"{:p}\"", r.a, self);
				r.a.__draw(file, roots);
			},
			ExprKind::MatMul(ref m) => {
				writeln!(file, "\t\"{:p}\" [label=\"MatMul\"{}]", self, color);
				writeln!(file, "\t\"{:p}\" -> \"{:p}\"", m.a, self);
				writeln!(file, "\t\"{:p}\" -> \"{:p}\"", m.b, self);
				m.a.__draw(file, roots);
				m.b.__draw(file, roots);
			},
			ExprKind::Transpose(ref t) => {
				writeln!(
					file,
					"\t\"{:p}\" [label=\"Transpose {} x {}\"{}]",
					self, t.x1, t.x2, color
				);
				writeln!(file, "\t\"{:p}\" -> \"{:p}\"", t.a, self);
				t.a.__draw(file, roots);
			},
			ExprKind::Broadcast(ref b) => {
				writeln!(file, "\t\"{:p}\" [label=\"Broadcast\"{}]", self, color);
				writeln!(file, "\t\"{:p}\" -> \"{:p}\"", b.a, self);
				b.a.__draw(file, roots);
			},
		}
	}

	pub fn draw(&self, filename: &str, roots: Option<&HashMap<*const Expr, isize>>) {
		let mut file = std::fs::File::create(filename).unwrap();
		writeln!(file, "digraph G {{").unwrap();
		writeln!(file, "\trankdir=BT").unwrap();
		self.__draw(&mut file, roots);
		writeln!(file, "}}").unwrap();
	}
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
	Broadcast(BroadcastExpr),
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

pub struct BroadcastExpr {
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
		_ => panic!("{}() requires a float input", op.symbol()),
	}
	Rc::new(Expr {
		shape: a.shape.clone(),
		dtype: a.dtype,
		kind: ExprKind::Unary(UnaryExpr { a, op }),
	})
}

pub fn exp<A: ExprLike>(a: A) -> Rc<Expr> {
	unary_op(a, UnaryOp::Exp)
}

pub fn sqrt<A: ExprLike>(a: A) -> Rc<Expr> {
	unary_op(a, UnaryOp::Sqrt)
}

fn binary_op<A: ExprLike, B: ExprLike>(a: A, b: B, op: BinaryOp) -> Rc<Expr> {
	let mut a = a.get();
	let mut b = b.get();
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
				//		let mut child = std::process::Command::new("c:/data/sw/clang/build11/Release/bin/clang")
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
	indexes: Vec<String>,
	dims: Vec<String>,
	perm: Vec<usize>,
	code: String,
}

impl Index {
	fn __new(indexes: Vec<String>, dims: Vec<String>, perm: Vec<usize>) -> Self {
		let ndim = perm.len();

		let mut code = String::new();
		for i in 0..ndim {
			if i != 0 {
				code.push_str(" + ");
			}
			code.push_str(indexes[perm[i]].as_str());
			for j in i + 1..ndim {
				write!(code, "*{}", dims[perm[j]]).unwrap();
			}
		}

		Index { indexes, dims, perm, code }
	}

	fn new(ndim: usize) -> Self {
		let mut indexes = Vec::new();
		let mut dims = Vec::new();

		for i in 0..ndim {
			indexes.push(format!("i_{}", i));
			dims.push(format!("dim_{}", i));
		}

		Self::__new(indexes, dims, (0..ndim).collect())
	}

	fn new_perm(parent: &Index, perm: &[usize]) -> Self {
		let mut new_perm = vec![0; perm.len()];
		for i in 0..perm.len() {
			new_perm[i] = perm[parent.perm[i]];
		}

		Self::__new(parent.indexes.clone(), parent.dims.clone(), new_perm)
	}

	fn new_transposed(parent: &Index, x1: usize, x2: usize) -> Self {
		let mut perm = (0..parent.perm.len()).collect::<Vec<usize>>();
		perm.swap(x1, x2);
		Self::new_perm(parent, &perm)
	}

	fn new_broadcast(parent: &Index, from_shape: &Shape, to_shape: &Shape) -> Self {
		debug_assert!(to_shape.ndim() >= from_shape.ndim());

		// how many 1s to add to the front of the from_shape
		let prefix = to_shape.ndim() - from_shape.ndim();

		let mut indexes = Vec::new();
		let mut dims = Vec::new();

		for i in 0..to_shape.ndim() {
			let dim = if i < prefix {
				1
			} else {
				from_shape.dims()[i - prefix]
			};

			if dim == 1 {
				indexes.push("0".to_string());
				dims.push("1".to_string());
			} else {
				indexes.push(format!("i_{}", i - prefix));
				dims.push(format!("dim_{}", i - prefix));
			}
		}

		Self::__new(indexes, dims, parent.perm.clone())
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

	fn gen_kernel(
		&self,
		sequence: &ComputeSequence,
		item: &PostorderItem,
	) -> Result<CPUKernel, std::fmt::Error> {
		let root = item.expr;
		let inputs = item.inputs;
		let ndim = root.shape.ndim();

		let mut code = String::new();
		writeln!(code, "#include <cstdint>")?;
		writeln!(code, "#include <cmath>")?;
		writeln!(code, "#include <stdio.h> // stdio")?;
		writeln!(code)?;
		writeln!(code, "using Index = uintptr_t;")?;
		writeln!(code, "using Count = uintptr_t;")?;
		writeln!(code)?;
		writeln!(code, "using f32 = float;")?;
		writeln!(code, "using f64 = double;")?;
		writeln!(code)?;
		writeln!(code, "inline f32 exp_f32(f32 x) {{ return expf(x); }}")?;
		writeln!(code, "inline f64 exp_f64(f64 x) {{ return exp(x); }}")?;
		writeln!(code)?;
		writeln!(code, "inline f32 sqrt_f32(f32 x) {{ return sqrtf(x); }}")?;
		writeln!(code, "inline f64 sqrt_f64(f64 x) {{ return sqrt(x); }}")?;
		writeln!(code)?;
		writeln!(
			code,
			"inline f32 max_f32(f32 x, f32 y) {{ return fmaxf(x, y); }}"
		)?;
		writeln!(
			code,
			"inline f64 max_f64(f64 x, f64 y) {{ return fmax(x, y); }}"
		)?;
		writeln!(code)?;
		writeln!(
			code,
			"inline f32 min_f32(f32 x, f32 y) {{ return fminf(x, y); }}"
		)?;
		writeln!(
			code,
			"inline f64 min_f64(f64 x, f64 y) {{ return fmin(x, y); }}"
		)?;
		writeln!(code)?;
		writeln!(code, "// cache key: `{}`", item.cache_key)?;
		writeln!(code, "extern \"C\" /*__declspec(dllexport)*/ void kernel(")?;
		writeln!(
			code,
			"\tCount const *dims, // array of {} dimension sizes",
			ndim
		)?;
		writeln!(
			code,
			"\tvoid const * const *inputs, // array of {} input tensors",
			inputs.len()
		)?;
		writeln!(code, "\t{} *output // output tensor", root.dtype)?;
		writeln!(code, ") {{")?;
		writeln!(
			code,
			"\tprintf(\"running kernel: {}\\n\"); // stdio",
			item.cache_key
		)?;

		for i in 0..ndim {
			writeln!(code, "\tCount const dim_{} = dims[{}];", i, i)?;
			writeln!(
				code,
				"\tprintf(\"dim_{} = %lu\\n\", dim_{}); // stdio",
				i, i
			)?;
		}
		writeln!(code)?;
		for i in 0..inputs.len() {
			let dtype = sequence.item_dtype(i);
			writeln!(
				code,
				"\t{} const * const input_{} = reinterpret_cast<{} const *>(inputs[{}]);",
				dtype, i, dtype, i
			)?;
		}
		writeln!(code)?;

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

		Ok(CPUKernel::new(
			&code,
			ndim,
			inputs.len(),
			self.tempdir.path(),
		))
	}

	fn gen_kernel_pointwise(&self, code: &mut String, sequence: &ComputeSequence, root: &Expr) {
		let ndim = root.shape.ndim();

		let index = Index::new(ndim);
		writeln!(code, "\t// pointwise kernel");
		for dim in 0..ndim {
			#[rustfmt::skip] writeln!(
				code, "{}\tfor (Index i_{} = 0; i_{} < dim_{}; ++i_{}) {{",
				Indent(dim), dim, dim, dim, dim
			);
		}

		write!(code, "{}\toutput[{}] = ", Indent(ndim), index.code);
		let mut input_counter = 0;
		self.gen_expr(
			code,
			ndim + 1,
			sequence,
			root,
			root,
			&index,
			&mut input_counter,
		);
		writeln!(code, "{}\t;", Indent(ndim));

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

		let index = Index::new(ndim);
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

		write!(code, "{}\t{} t = ", Indent(loop_cnt), root.dtype);
		let mut input_counter = 0;
		self.gen_expr(
			code,
			loop_cnt + 1,
			sequence,
			root,
			&red.a,
			&index,
			&mut input_counter,
		);
		writeln!(code, "{}\t;", Indent(loop_cnt));
		writeln!(code);
		match red.op {
			ReductionOp::Sum => {
				writeln!(code, "{}\tvalue += t;", Indent(loop_cnt));
			},
			ReductionOp::Max => {
				writeln!(
					code,
					"{}\tvalue = max_{}(value, t);",
					Indent(loop_cnt),
					root.dtype
				);
			},
			ReductionOp::Min => {
				writeln!(
					code,
					"{}\tvalue = min_{}(value, t);",
					Indent(loop_cnt),
					root.dtype
				);
			},
		}

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
		indent: usize,
		sequence: &ComputeSequence,
		root: &Expr,
		expr: &Expr,
		index: &Index,
		input_counter: &mut usize,
	) -> Result<(), std::fmt::Error> {
		if (root as *const Expr) != (expr as *const Expr) && sequence.is_root(expr) {
			writeln!(code, "input_{}[{}]", input_counter, index.code)?;
			*input_counter += 1;
			return Ok(());
		}

		match &expr.kind {
			ExprKind::Const(c) => {
				writeln!(code, "{}({})", expr.dtype, c)?;
			},
			ExprKind::Read(t) => {
				writeln!(code, "READ_TENSOR_TODO")?;
			},
			ExprKind::Unary(un) => {
				writeln!(code, "{}_{}(", un.op.symbol(), expr.dtype)?;
				write!(code, "{}", Indent(indent + 1))?;
				self.gen_expr(
					code,
					indent + 1,
					sequence,
					root,
					&un.a,
					index,
					input_counter,
				)?;
				writeln!(code, "{})", Indent(indent))?;
			},
			ExprKind::Binary(bin) => {
				writeln!(code, "(")?;
				write!(code, "{}", Indent(indent + 1))?;
				if bin.op.is_commutative() && sequence.has_swapped_operands(expr) {
					self.gen_expr(
						code,
						indent + 1,
						sequence,
						root,
						&bin.b,
						index,
						input_counter,
					)?;
					writeln!(code, "{}{}", Indent(indent), bin.op.symbol())?;
					write!(code, "{}", Indent(indent + 1))?;
					self.gen_expr(
						code,
						indent + 1,
						sequence,
						root,
						&bin.a,
						index,
						input_counter,
					)?;
				} else {
					self.gen_expr(
						code,
						indent + 1,
						sequence,
						root,
						&bin.a,
						index,
						input_counter,
					)?;
					writeln!(code, "{}{}", Indent(indent), bin.op.symbol())?;
					write!(code, "{}", Indent(indent + 1))?;
					self.gen_expr(
						code,
						indent + 1,
						sequence,
						root,
						&bin.b,
						index,
						input_counter,
					)?;
				}
				writeln!(code, "{})", Indent(indent))?;
			},
			ExprKind::Transpose(t) => {
				let new_index = Index::new_transposed(index, t.x1, t.x2);
				self.gen_expr(
					code,
					indent + 1,
					sequence,
					root,
					&t.a,
					&new_index,
					input_counter,
				)?;
			},
			ExprKind::Broadcast(b) => {
				let new_index = Index::new_broadcast(index, &b.a.shape, &root.shape);
				self.gen_expr(
					code,
					indent + 1,
					sequence,
					root,
					&b.a,
					&new_index,
					input_counter,
				)?;
			},
			ExprKind::Randn() | ExprKind::Reduction(..) | ExprKind::MatMul(..) => {
				panic!("Unsupported expression");
			},
		}
		Ok(())
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

	fn eval(self: Rc<CPUDevice>, expr: Rc<Expr>, dotfile: Option<&str>) -> Rc<Tensor> {
		let sequence = ComputeSequence::new(expr.clone());

		if let Some(dotfile) = dotfile {
			expr.draw(dotfile, Some(&sequence.roots));
		}

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
				ExprKind::MatMul(..) => {
					unimplemented!("MatMul")
				},
				ExprKind::Const(..)
				| ExprKind::Unary(..)
				| ExprKind::Binary(..)
				| ExprKind::Reduction(..)
				| ExprKind::Transpose(..)
				| ExprKind::Broadcast(..) => {
					let rc = item.ref_count;
					buffers.push(unsafe {
						CPUDevice::new_uninit(expr.dtype.bits(), expr.shape.elems())
					});
					rc_buffers.push((rc, buffers.last().unwrap() as *const CPUTensorData));

					// TODO - try to find the kernel in the cache
					let kernel = self.gen_kernel(&sequence, &item).unwrap();

					// For reduce kernel, the dims should be dims of the input
					let dims = match &expr.kind {
						ExprKind::Reduction(red) => red.a.shape.dims(),
						_ => expr.shape.dims(),
					};
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

					for i in item.inputs.iter() {
						let (rc, _) = &mut rc_buffers[*i];
						*rc -= 1;
						if *rc == 0 {
							buffers[*i].__data = Vec::new().into_boxed_slice();
						}
					}
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

	fn format(
		&self,
		f: &mut fmt::Formatter,
		tensor: &Tensor,
		off: usize,
		len: usize,
		stride: isize,
	) -> fmt::Result {
		let tensor = tensor as *const Tensor;
		let tensor = tensor as *const Tensor<CPUTensorData>;
		let tensor = unsafe { &*tensor };
		match tensor.dtype {
			DType::Float(32) => {
				let data = tensor.data.cast::<f32>();
				for i in 0..len {
					if i != 0 {
						write!(f, ", ")?;
					}
					let p = (off as isize) + (i as isize) * stride;
					write!(f, "{:.6}", data[p as usize].get())?;
				}
			},
			_ => unimplemented!(),
		}
		Ok(())
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
			ExprKind::Broadcast(b) => {
				self.find_kernel_roots(&*b.a);

				self.roots.insert(b.a.as_ref() as *const Expr, -1);
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
					self.swapped_operands.insert(expr);
				}
				format!("({}{}{})", a, bin.op.symbol(), b)
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
					"T({},{},{})",
					a,
					std::cmp::min(t.x1, t.x2),
					std::cmp::max(t.x1, t.x2)
				)
			},
			ExprKind::Broadcast(br) => {
				let a = self.find_postorder(&*br.a, parent_inputs);
				format!("B({},{},{})", a, expr.shape, br.a.shape)
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

			format!("{}", expr.dtype)
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

fn fmt_0d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	tensor.device.format(f, tensor, off, 1, 1)
}

fn fmt_1d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	write!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 1];
	tensor.device.format(f, tensor, off, len, 1)?;
	write!(f, "]")
}

fn fmt_2d(tensor: &Tensor, f: &mut fmt::Formatter, off: usize) -> fmt::Result {
	writeln!(f, "[")?;
	let ndim = tensor.shape.ndim();
	let len = tensor.shape.dims()[ndim - 2];
	let stride = tensor.shape.dims()[ndim - 1];
	for i in 0..len {
		write!(f, "\t")?;
		fmt_1d(tensor, f, off + i * stride)?;
		writeln!(f, ",")?;
	}
	write!(f, "]")
}

impl fmt::Display for Tensor {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let ndim = self.shape.ndim();
		match ndim {
			0 => fmt_0d(self, f, 0),
			1 => fmt_1d(self, f, 0),
			2 => fmt_2d(self, f, 0),
			_ => {
				unimplemented!("Tensor with {} dimensions", ndim);
			},
		}
	}
}
