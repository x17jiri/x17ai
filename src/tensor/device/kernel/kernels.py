import ast

kernels = """
def add(a: Scalar, b: Scalar):
	a + b

def weighted_add(a_weight: Const, a: Scalar, b_weight: Const, b: Scalar):
	(a * a_weight) + (b * b_weight)

def dot(a: Vector, b: Vector):
	(a * b).sum()

def dot_scaled(a: Vector, b: Vector, scale: Const):
	(a * b).sum() * scale

def dot_scaled2(a: Vector, b: Vector, scale1: Const, scale2: Const):
	dot_scaled(a, b, scale1 * scale2)
""";

source_lines = kernels.splitlines()

class Arg:
	def __init__(self, name, type, pos):
		self.name = name
		self.type = type
		self.pos = pos
		self.rust_type = 'f64' if type == 'Const' else "&'a Tensor"

	def dump(self, indent=0):
		print(f"Arg(name={self.name}, type={self.type}), pos={self.pos})", sep="")

class Fn:
	def __init__(self, name, type_counts, args, body, redirection):
		self.name = name
		self.type_counts = type_counts
		self.cls_name = to_camel_case(name)
		self.args = args
		self.body = body
		self.redirection = redirection
		self.e_args = [arg.name for arg in args if arg.type == 'Scalar']
		self.r_args = [arg.name for arg in args if arg.type == 'Vector']
		self.c_args = [arg.name for arg in args if arg.type == 'Const']

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

def to_camel_case(s):
    return ''.join(word.capitalize() for word in s.split('_'))

def print_expr(expr, add_parens=False):
	ln_b = expr.lineno
	col_b = expr.col_offset
	ln_e = expr.end_lineno
	col_e = expr.end_col_offset

	assert ln_b == ln_e
	return source_lines[ln_b-1][col_b:col_e]

def parse_kernels(kernels):
	tree = ast.parse(kernels)
	print(ast.dump(tree, indent=4))

	kernel_list = []
	for node in tree.body:
		match node:
			case ast.FunctionDef(name, args, body):
				type_counts = {
					'Scalar': 0,
					'Vector': 0,
					'Const': 0,
				}
				fn_name = name
				print("Processing function name:", fn_name)
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
				assert len(body) == 1

				redirection = None
				match body[0]:
					case ast.Expr(value):
						match value:
							case ast.Call(func, args):
								match func:
									case ast.Name(id):
										redirection = (id, args)

				kernel_list.append(Fn(fn_name, type_counts, arg_list, body[0], redirection))
			case _:
				raise ValueError(f"Unexpected node type: {type(node)}")

	return kernel_list

kernel_list = parse_kernels(kernels)

print("// Generated file, do not edit!")
print()
print("//------------------------------------------------------------------------------")
print("//")
print("// Copyright 2025 Jiri Bobek. All rights reserved.")
print("// License: GPL 3.0 or later. See LICENSE.txt for details.")
print("//")
print("//------------------------------------------------------------------------------")
print()
for kernel in kernel_list:
	#kernel.dump()
	print("//--------------------------------------------------------------------------------------------------")
	print()
	if kernel.redirection:
		print(f"// TODO: Handle redirection")
	else:
		print(f"#[derive(Clone, Copy)]")
		print(f"pub struct {kernel.cls_name}Kernel {{")
		E = kernel.type_counts['Scalar']
		R = kernel.type_counts['Vector']
		C = kernel.type_counts['Const']
		print(f"\tkernel: &'static Kernel<{E}, {R}, {C}>,")
		print(f"}}")
		print()
		print(f"impl {kernel.cls_name}Kernel {{")
		print(f"	pub fn instance() -> Self {{")
		print(f"		static instance: OnceLock<Kernel<{E}, {R}, {C}>> = OnceLock::new();")
		print(f"		let kernel = instance.get_or_init(|| {{")
		e_args = ", ".join(kernel.e_args)
		r_args = ", ".join(kernel.r_args)
		c_args = ", ".join(kernel.c_args)
		e_arg_names = ", ".join([f'"{arg}"' for arg in kernel.e_args])
		r_arg_names = ", ".join([f'"{arg}"' for arg in kernel.r_args])
		c_arg_names = ", ".join([f'"{arg}"' for arg in kernel.c_args])
		print(f'			let (builder, [{e_args}], [{r_args}], [{c_args}]) =')
		print(f'				KernelBuilder::new(')
		print(f'					"{kernel.name}", [{e_arg_names}], [{r_arg_names}], [{c_arg_names}]')
		print(f'				);')
		print(f'			builder.build({print_expr(kernel.body)})')
		print(f'		}});')
		print(f'		Self {{ kernel }}')
		print(f'	}}')
		print()
		header = ", ".join([f"{arg.name}: {arg.rust_type}" for arg in kernel.args])
		args = ", ".join([arg.name for arg in kernel.args])
		print(f"	pub fn call<'a>(self, {header}) -> {kernel.cls_name}KernelCall<'a> {{")
		print(f"		{kernel.cls_name}KernelCall {{ kernel: self, {args} }}")
		print(f"	}}")
		print(f"}}")
		print()
		print(f"pub struct {kernel.cls_name}KernelCall<'a> {{")
		print(f"	kernel: {kernel.cls_name}Kernel,")
		for arg in kernel.args:
			print(f"	{arg.name}: {arg.rust_type},")
		print(f"}}")
		print()
		print(f"impl<'a> EvaluatesToTensor for {kernel.cls_name}KernelCall<'a> {{")
		print(f"	fn eval_to_tensor(self, to: &Tensor) -> Result<(), ErrPack<TensorOpError>> {{")
		e_args = ", ".join('self.'+arg for arg in kernel.e_args)
		r_args = ", ".join('self.'+arg for arg in kernel.r_args)
		c_args = ", ".join('self.'+arg for arg in kernel.c_args)
		print(f"		self.kernel.kernel.run(to, [{e_args}], [{r_args}], [{c_args}])")
		print(f"	}}")
		print(f"}}")
		print()
		print(f"type {kernel.cls_name}Expr<'a> =")
		print(f"	RecipLookupExpr<")
		print(f"		SqrtLookupExpr<&'a Tensor>,")
		print(f"		f64")
		print(f"	>;")
		print()
		print(f"impl<'a> KernelLookup<{kernel.cls_name}Expr<'a>> for KernelLibrary {{")
		print(f"	type CallType = {kernel.cls_name}KernelCall<'a>;")
		print()
		print(f"	fn create_call(&self, expr: LookupWrapper<{kernel.cls_name}Expr<'a>>) -> {kernel.cls_name}KernelCall<'a> {{")
		print(f"		let RecipLookupExpr(SqrtLookupExpr(inp), eps) = expr.0;")
		print(f"		self.data.{kernel.name}.call({args})")
		print(f"	}}")
		print(f"}}")

	print()

print("//--------------------------------------------------------------------------------------------------")
print()
