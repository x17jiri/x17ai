import ast
import re
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
kernels_path = os.path.join(script_dir, 'kernels.txt')

with open(kernels_path, 'r') as f:
    kernels = f.read()

source_lines = kernels.splitlines()

CONST_TYPE = 'C'
ELEM_TYPE = 'E'
REDUCE_TYPE = 'R'

class Redirect:
	def __init__(self, target, args):
		self.target = target
		self.args = args

class Arg:
	def __init__(self, name, type, pos):
		self.name = name
		self.type = type
		self.pos = pos
		self.needs_lifetime = type != CONST_TYPE
		self.rust_type = 'f64' if type == CONST_TYPE else "&'a Tensor"

	def dump(self, indent=0):
		print(f"Arg(name={self.name}, type={self.type}), pos={self.pos})", sep="")

	def print_expr(self, indent):
		return self.rust_type

	def print_destructuring(self, indent):
		return self.name

class Temp:
	def __init__(self, name):
		self.name = name

	def dump(self, indent=0):
		print(f"Temp(name={self.name}")

	def print_expr(self, indent):
		return self.rust_type

class Fn:
	def __init__(self, name, type_counts, args, body, redirection):
		arg_map = {}
		needs_lifetime = False
		for arg in args:
			assert arg.name not in arg_map, f"Duplicate argument name: {arg.name}"
			arg_map[arg.name] = arg
			needs_lifetime = needs_lifetime or arg.needs_lifetime

		assert len(body) > 0
		assert isinstance(body[-1], ast.Expr), f"Expected Expr, got {type(body[-1])}"
		self.temp = [parse_temp(arg_map, t) for t in body[:-1]]
		self.return_expr = body[-1].value

		if len(self.temp) == 0:
			self.body_op = parse_body(arg_map, self.return_expr)
		else:
			self.body_op = None

		self.name = name
		self.type_counts = type_counts
		self.cls_name = to_camel_case(name)
		self.args = args
		self.body = body
		self.redirection = redirection
		self.e_args = [arg.name for arg in args if arg.type == ELEM_TYPE]
		self.r_args = [arg.name for arg in args if arg.type == REDUCE_TYPE]
		self.c_args = [arg.name for arg in args if arg.type == CONST_TYPE]
		self.needs_lifetime = needs_lifetime
		self.lifetime = "<'a>" if needs_lifetime else ""

	def dump(self, indent=0):
		print(f"//Fn:", sep="")
		print("//", "\t"*(indent+1), f"name = {self.name}", sep="")
		print("//", "\t"*(indent+1), "args = [", sep="")
		for arg in self.args:
			print("//", "\t"*(indent+2), sep="", end="")
			arg.dump(indent+2)
		print("//", "\t"*(indent+1), "]", sep="")
		print("//", "\t"*(indent+1), f"body = {self.body}", sep="")
		print("//", "\t"*(indent+1), f"redirection = {self.redirection}", sep="")

class Op:
	def __init__(self, cls_name, args):
		self.cls_name = cls_name
		self.args = args

	def print_expr(self, indent):
		result = f"{self.cls_name}<\n";
		nested_indent = indent + "\t"
		for arg in self.args:
			result += nested_indent + arg.print_expr(nested_indent) + ",\n"
		result += indent + ">"
		return result

	def print_destructuring(self, indent):
		result = f"{self.cls_name}(\n";
		nested_indent = indent + "\t"
		for arg in self.args:
			result += nested_indent + arg.print_destructuring(nested_indent) + ",\n"
		result += indent + ")"
		return result

def parse_temp(arg_map, temp):
	match temp:
		case ast.Assign(targets=[ast.Name(id)], value=value):
			return (id, value)
		case _:
			raise ValueError(f"Unexpected temp command: {type(temp)}")

def parse_body(arg_map, body):
	match body:
		case ast.Name(id):
			return arg_map[id]
		case ast.BinOp(left, op, right):
			left = parse_body(arg_map, left)
			right = parse_body(arg_map, right)
			match op:
				case ast.Add():
					return Op('AddLookupExpr', [left, right])
				case ast.Sub():
					return Op('SubLookupExpr', [left, right])
				case ast.Mult():
					return Op('MulLookupExpr', [left, right])
				case _:
					raise ValueError(f"Unexpected binary operator: {type(op)}")
		case ast.Call(func, args):
			match func:
				case ast.Attribute(value, attr):
					value = parse_body(arg_map, value)
					args = [parse_body(arg_map, arg) for arg in args]
					return Op(f"{to_camel_case(attr)}LookupExpr", [value] + args)
				case _:
					raise ValueError(f"Unexpected function call: {type(func)}")
		case _:
			raise ValueError(f"Unexpected body type: {type(body)}")

def to_camel_case(s):
    return ''.join(word.capitalize() for word in s.split('_'))

def expr_as_str(expr):
	ln_b = expr.lineno
	col_b = expr.col_offset
	ln_e = expr.end_lineno
	col_e = expr.end_col_offset

	assert ln_b == ln_e
	return source_lines[ln_b-1][col_b:col_e]

def parse_kernels(kernels):
	tree = ast.parse(kernels)
	print(ast.dump(tree, indent=4), file=sys.stderr)

	kernel_list = []
	for node in tree.body:
		match node:
			case ast.FunctionDef(name, args, body, decorator_list):
				type_counts = {
					ELEM_TYPE: 0,
					REDUCE_TYPE: 0,
					CONST_TYPE: 0,
				}
				fn_name = name
				print("Processing function name:", fn_name, file=sys.stderr)
				assert args.posonlyargs == []
				assert args.kwonlyargs == []
				assert args.kw_defaults == []
				assert args.defaults == []
				arg_list = []
				for arg in args.args:
					arg_name = arg.arg
					match arg.annotation:
						case ast.Name(id):
							pos = type_counts[id]
							type_counts[id] += 1
							arg_list.append(Arg(arg_name, id, pos))
						case _:
							raise ValueError(f"Unexpected annotation type: {type(arg.annotation)}")

				redirection = None
				if decorator_list:
					assert len(decorator_list) == 1
					match decorator_list[0]:
						case ast.Call(func=ast.Name(id), args=args):
							redirection = Redirect(id, [expr_as_str(arg) for arg in args])
						case _:
							raise ValueError(f"Unexpected decorator type: {type(decorator_list[0])}")

				kernel_list.append(Fn(fn_name, type_counts, arg_list, body, redirection))
			case _:
				raise ValueError(f"Unexpected node type: {type(node)}")

	return kernel_list

kernel_list = parse_kernels(kernels)
kernel_map = {kernel.name: kernel for kernel in kernel_list}

print("// Generated file, do not edit")
print()
print("//------------------------------------------------------------------------------")
print("//")
print("// Copyright 2025 Jiri Bobek. All rights reserved.")
print("// License: GPL 3.0 or later. See LICENSE.txt for details.")
print("//")
print("//------------------------------------------------------------------------------")
print()
print("use crate::ErrPack;")
print("use crate::tensor::{Tensor, TensorOpError};")
print()
print("use super::Kernel;")
print("use super::builder::KernelBuilder;")
print("use super::library::KernelLibrary;")
print("use super::lookup::{")
print("	AddLookupExpr,")
print("	KernelCall,")
print("	LnClampedLookupExpr,")
print("	LookupWrapper,")
print("	MulLookupExpr,")
print("	RecipLookupExpr,")
print("	SqrtLookupExpr,")
print("	SubLookupExpr,")
print("	SumLookupExpr,")
print("	SwishLookupExpr,")
print("};")
print()

for kernel in kernel_list:
	kernel.dump()
	print("//--------------------------------------------------------------------------------------------------")
	print()
	if not kernel.redirection:
		print(f"#[derive(Clone)]")
		print(f"pub struct {kernel.cls_name}Kernel {{")
		E = kernel.type_counts[ELEM_TYPE]
		R = kernel.type_counts[REDUCE_TYPE]
		C = kernel.type_counts[CONST_TYPE]
		print(f"\tkernel: Kernel<{E}, {R}, {C}>,")
		print(f"}}")
		print()
		print(f"impl {kernel.cls_name}Kernel {{")
		print(f"	fn new() -> Self {{")
		e_args = ", ".join(kernel.e_args)
		r_args = ", ".join(kernel.r_args)
		c_args = ", ".join(kernel.c_args)
		e_arg_names = ", ".join([f'"{arg}"' for arg in kernel.e_args])
		r_arg_names = ", ".join([f'"{arg}"' for arg in kernel.r_args])
		c_arg_names = ", ".join([f'"{arg}"' for arg in kernel.c_args])
		print(f'		let (builder, [{e_args}], [{r_args}], [{c_args}]) =')
		print(f'			KernelBuilder::new(')
		print(f'				"{kernel.name}", [{e_arg_names}], [{r_arg_names}], [{c_arg_names}]')
		print(f'			);')
		for temp in kernel.temp:
			print(f'		let {temp[0]} = {expr_as_str(temp[1])};')
		print(f'		let kernel = builder.build({expr_as_str(kernel.return_expr)});')
		print(f'		Self {{ kernel }}')
		print(f'	}}')
		print(f"}}")
		print()
		args = ", ".join([arg.name for arg in kernel.args])
		#header = ", ".join([f"{arg.name}: {arg.rust_type}" for arg in kernel.args])
		#if kernel.needs_lifetime:
		#	print(f"	pub fn call<'a>(&'a self, {header}) -> {kernel.cls_name}KernelCall<'a> {{")
		#	print(f"		{kernel.cls_name}KernelCall {{ kernel: self, {args} }}")
		#	print(f"	}}")
		#else:
		#	print(f"	pub fn call<'a>(&'a self, {header}) -> {kernel.cls_name}KernelCall<'a> {{")
		#	print(f"		{kernel.cls_name}KernelCall {{ kernel: self, {args}, phantom: std::marker::PhantomData, }}")
		#	print(f"	}}")
		#print(f"}}")
		#print()
		#print(f"pub struct {kernel.cls_name}KernelCall<'a> {{")
		#print(f"	kernel: &'a {kernel.cls_name}Kernel,")
		#for arg in kernel.args:
		#	print(f"	{arg.name}: {arg.rust_type},")
		#if not kernel.needs_lifetime:
		#	print(f"	phantom: std::marker::PhantomData<&'a ()>,")
		#print(f"}}")
		#print()
		#print(f"impl<'a> EvaluatesToTensor for {kernel.cls_name}KernelCall<'a> {{")
		#print(f"	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {{")
		#e_args = ", ".join('self.'+arg for arg in kernel.e_args)
		#r_args = ", ".join('self.'+arg for arg in kernel.r_args)
		#c_args = ", ".join('self.'+arg for arg in kernel.c_args)
		#print(f"		self.kernel.kernel.run(to, [{e_args}], [{r_args}], [{c_args}])")
		#print(f"	}}")
		#print(f"}}")
		#print()
        #
		#kernel_call_class = f"{kernel.cls_name}KernelCall"
		#kernel_call_expr = f"{kernel.name}.call({args})"
	#else:
		##print(f"// Redirecting {kernel.name} to {kernel.redirection}")
        #
		#kernel_call_func = re.search(r'^\s*(\w+)\s*\(', kernel.redirection).group(1)
		#kernel_call_class = f"{to_camel_case(kernel_call_func)}KernelCall"
		#kernel_call_expr = result = re.sub(r'^\s*(\w+)\s*\(', r'\1.call(', kernel.redirection)

	if kernel.temp:
		header = ", ".join([f"{arg.name}: {re.sub("&'a ", "&", arg.rust_type)}" for arg in kernel.args])
		print(f"pub fn {kernel.name}(")
		print(f"	to: &Tensor,")
		print(f"	{header},")
		print(f") -> Result<(), ErrPack<TensorOpError>> {{")
		print(f"	let library = to.builtin_kernel_library();")
		e_args = ", ".join(kernel.e_args)
		r_args = ", ".join(kernel.r_args)
		c_args = ", ".join(kernel.c_args)
		print(f"	library.data.{kernel.name}.kernel.run(to, [{e_args}], [{r_args}], [{c_args}])")
		print(f"}}")

	else:
		print(f"type {kernel.cls_name}Expr<'a> =")
		print(f"\t{kernel.body_op.print_expr("\t")};")

		print()

		print(f"impl<'a> KernelCall<{kernel.cls_name}Expr<'a>> for KernelLibrary {{")
		print(f"	fn call(")
		print(f"		&self,")
		print(f"		to: &Tensor,")
		print(f"		expr: LookupWrapper<{kernel.cls_name}Expr<'a>>")
		print(f"	) -> Result<(), ErrPack<TensorOpError>> {{")
		print(f"		let {kernel.body_op.print_destructuring("\t\t")} = expr.0;")

		if kernel.redirection:
			zipped = list(zip(kernel.redirection.args, kernel_map[kernel.redirection.target].args))
			e_args = ", ".join(z[0] for z in zipped if z[1].type == ELEM_TYPE)
			r_args = ", ".join(z[0] for z in zipped if z[1].type == REDUCE_TYPE)
			c_args = ", ".join(z[0] for z in zipped if z[1].type == CONST_TYPE)
			print(f"		self.data.{kernel.redirection.target}.kernel.run(to, [{e_args}], [{r_args}], [{c_args}])")
		else:
			e_args = ", ".join(kernel.e_args)
			r_args = ", ".join(kernel.r_args)
			c_args = ", ".join(kernel.c_args)
			print(f"		self.data.{kernel.name}.kernel.run(to, [{e_args}], [{r_args}], [{c_args}])")
		print(f"	}}")
		print(f"}}")

	print()

print("//--------------------------------------------------------------------------------------------------")
print()
print(f"pub struct KernelLibraryData {{")
for kernel in kernel_list:
	if not kernel.redirection:
		print(f"	{kernel.name}: {kernel.cls_name}Kernel,")
print(f"}}")
print()
print(f"impl KernelLibraryData {{")
print(f"	pub fn new() -> Self {{")
print(f"		Self {{")
for kernel in kernel_list:
	if not kernel.redirection:
		print(f"			{kernel.name}: {kernel.cls_name}Kernel::new(),")
print(f"		}}")
print(f"	}}")
print(f"}}")
print()
print("//--------------------------------------------------------------------------------------------------")
