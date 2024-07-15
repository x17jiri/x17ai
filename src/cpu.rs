// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

extern crate libloading;
extern crate tempfile;

#[cfg(unix)]
use libloading::os::unix::Symbol as RawSymbol;
#[cfg(windows)]
use libloading::os::windows::Symbol as RawSymbol;

use crate::*;
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::path::Path;
use std::rc::Rc;

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
			ExprKind::Reduce(red) => {
				self.gen_kernel_reduce(&mut code, sequence, root, red);
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
				Indent::new(dim), dim, dim, dim, dim
			);
		}

		write!(code, "{}\toutput[{}] = ", Indent::new(ndim), index.code);
		let mut input_counter = 0;
		self.gen_expr(
			code,
			Indent::new(ndim + 1),
			sequence,
			root,
			root,
			&index,
			&mut input_counter,
		);
		writeln!(code, "{}\t;", Indent::new(ndim));

		for dim in (0..ndim).rev() {
			for _ in 0..dim {
				code.push('\t');
			}
			writeln!(code, "\t}}");
		}
	}

	fn gen_kernel_reduce(
		&self,
		code: &mut String,
		sequence: &ComputeSequence,
		root: &Expr,
		red: &ReduceExpr,
	) {
		let ndim = root.shape.ndim();

		let index = Index::new(ndim);
		writeln!(code, "\t// reduce kernel");

		let mut loop_cnt = 0;
		for dim in 0..ndim {
			// TODO - use binary_search()? ndim will usually be very small, so contains() may be faster
			if red.dims_to_reduce.contains(dim) {
				continue;
			}
			#[rustfmt::skip] writeln!(
				code, "{}\tfor (Index i_{} = 0; i_{} < dim_{}; ++i_{}) {{",
				Indent::new(loop_cnt), dim, dim, dim, dim
			);
			loop_cnt += 1;
		}

		let outer_loops = loop_cnt;
		// TODO: `0.0` is only correct for `sum`
		writeln!(
			code,
			"{}\t{} value = {}(0);",
			Indent::new(loop_cnt),
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
				Indent::new(loop_cnt), dim, dim, dim, dim
			);
			loop_cnt += 1;
		}

		write!(code, "{}\t{} t = ", Indent::new(loop_cnt), root.dtype);
		let mut input_counter = 0;
		self.gen_expr(
			code,
			Indent::new(loop_cnt + 1),
			sequence,
			root,
			&red.a,
			&index,
			&mut input_counter,
		);
		writeln!(code, "{}\t;", Indent::new(loop_cnt));
		writeln!(code);
		match red.op {
			ReduceOp::Sum => {
				writeln!(code, "{}\tvalue += t;", Indent::new(loop_cnt));
			},
			ReduceOp::Max => {
				writeln!(
					code,
					"{}\tvalue = max_{}(value, t);",
					Indent::new(loop_cnt),
					root.dtype
				);
			},
			ReduceOp::Min => {
				writeln!(
					code,
					"{}\tvalue = min_{}(value, t);",
					Indent::new(loop_cnt),
					root.dtype
				);
			},
		}

		for _ in (outer_loops..ndim).rev() {
			loop_cnt -= 1;
			writeln!(code, "{}\t}}", Indent::new(loop_cnt));
		}

		for i in &red.dims_to_reduce {
			writeln!(
				code,
				"{}\tCount dim_{} = 1; Index i_{} = 0;",
				Indent::new(outer_loops),
				i,
				i
			);
		}

		writeln!(
			code,
			"{}\toutput[{}] = value;",
			Indent::new(loop_cnt),
			index.code
		);

		for _ in (0..outer_loops).rev() {
			loop_cnt -= 1;
			writeln!(code, "{}\t}}", Indent::new(loop_cnt));
		}
	}

	fn gen_kernel_matmul(&self, _code: &mut String, _sequence: &ComputeSequence, _root: &Expr) {
		unimplemented!("gen_kernel_matmul");
	}

	fn gen_expr(
		&self,
		code: &mut String,
		indent: Indent,
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
			ExprKind::Input(..) => {
				writeln!(code, "READ_TENSOR_TODO")?;
			},
			ExprKind::Unary(un) => {
				writeln!(code, "{}_{}(", un.op.symbol(), expr.dtype)?;
				write!(code, "{}", indent + 1)?;
				self.gen_expr(
					code,
					indent + 1,
					sequence,
					root,
					&un.a,
					index,
					input_counter,
				)?;
				writeln!(code, "{})", indent)?;
			},
			ExprKind::Binary(bin) => {
				writeln!(code, "(")?;
				write!(code, "{}", indent + 1)?;
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
					writeln!(code, "{}{}", indent, bin.op.symbol())?;
					write!(code, "{}", indent + 1)?;
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
					writeln!(code, "{}{}", indent, bin.op.symbol())?;
					write!(code, "{}", indent + 1)?;
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
				writeln!(code, "{})", indent)?;
			},
			ExprKind::Transpose(t) => {
				let new_index = Index::new_transposed(index, t.x1, t.x2);
				self.gen_expr(
					code,
					indent,
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
					indent,
					sequence,
					root,
					&b.a,
					&new_index,
					input_counter,
				)?;
			},
			ExprKind::SimpleReshape(sr) => {
				self.gen_expr(code, indent, sequence, root, &sr.a, index, input_counter)?;
			},
			ExprKind::Randn() | ExprKind::Reduce(..) | ExprKind::MatMul(..) => {
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
			sequence.draw_expr(dotfile);
		}

		let mut buffers: Vec<CPUTensorData> = Vec::with_capacity(sequence.len());
		let mut rc_buffers: Vec<(usize, *const CPUTensorData)> = Vec::with_capacity(sequence.len());
		for item in sequence.iter() {
			let expr = item.expr;
			match &expr.kind {
				ExprKind::Input(tensor) => {
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
				| ExprKind::Reduce(..)
				| ExprKind::Transpose(..)
				| ExprKind::SimpleReshape(..)
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
						ExprKind::Reduce(red) => red.a.shape.dims(),
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
