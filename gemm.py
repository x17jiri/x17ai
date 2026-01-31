from enum import Enum, auto

class Layout(Enum):
	RowMajor = auto()
	ColMajor = auto()

class Storage(Enum):
	Register = auto()
	Shared = auto()
	Global = auto()

class Matrix:
	def __init__(self, name, m, n, storage, layout, transposed=False):
		self.name = name
		self.m = m
		self.n = n
		self.storage = storage
		self.layout = layout
		self.transposed = transposed

	def T(self):
		return Matrix(
			name=self.name,
			m=self.m,
			n=self.n,
			storage=self.storage,
			layout=self.layout,
			transposed=not self.transposed
		)

	def tile(self, row_range, col_range):
		return Tile(self, row_range, col_range)

class Tile:
	def __init__(self, matrix: Matrix, row_range: range, col_range: range):
		self.matrix = matrix
		self.row_range = row_range
		self.col_range = col_range

	def __str__(self):
		if self.matrix.transposed:
			return "{name}.tile({cols_from}..<{cols_to}, {rows_from}..<{rows_to}).T".format(
					name=self.matrix.name,
					rows_from=self.row_range.start,
					rows_to=self.row_range.stop,
					cols_from=self.col_range.start,
					cols_to=self.col_range.stop
				)
		else:
			return "{name}.tile({rows_from}..<{rows_to}, {cols_from}..<{cols_to})".format(
					name=self.matrix.name,
					rows_from=self.row_range.start,
					rows_to=self.row_range.stop,
					cols_from=self.col_range.start,
					cols_to=self.col_range.stop
				)

class GemmAtom:
	def __init__(self, name, m, n, k, layout_a, layout_b, layout_c):
		self.name = name
		self.m = m
		self.n = n
		self.k = k
		self.layout_a = layout_a
		self.layout_b = layout_b
		self.layout_c = layout_c

class AtomCall:
	def __init__(self, atom: GemmAtom, a: Tile, b: Tile, c: Tile):
		self.atom = atom
		self.a = a
		self.b = b
		self.c = c

	def __str__(self):
		return "{name}({a}, {b}, {c})".format(
				name=self.atom.name,
				a=self.a,
				b=self.b,
				c=self.c
			)

class Preload:
	def __init__(self, tile: Tile, register: str):
		self.tile = tile
		self.register = register

	def __str__(self):
		return "{reg} = {tile}".format(
			reg=self.register,
			tile=self.tile
		)

def gemm(a: Matrix, b: Matrix, c: Matrix, atom: GemmAtom):
	m = a.m if not a.transposed else a.n
	n = b.n if not b.transposed else b.m
	k = a.n if not a.transposed else a.m
	k2 = b.m if not b.transposed else b.n
	assert k == k2, "Inner dimensions must match"
	m2 = c.m if not c.transposed else c.n
	n2 = c.n if not c.transposed else c.m
	assert m == m2 and n == n2, "Output dimensions must match"

	m_steps = m // atom.m
	n_steps = n // atom.n
	k_steps = k // atom.k

	result = []
	result.append("// {c} += {a} x {b}".format(
		a=a.name if not a.transposed else a.name + ".T",
		b=b.name if not b.transposed else b.name + ".T",
		c=c.name if not c.transposed else c.name + ".T"
	))
	for p in range(k_steps):
		for i in range(m_steps):
			for j in range(n_steps):
				result.append(AtomCall(
					atom,
					a.tile(range(atom.m*i, atom.m*(i+1)), range(atom.k*p, atom.k*(p+1))),
					b.tile(range(atom.k*p, atom.k*(p+1)), range(atom.n*j, atom.n*(j+1))),
					c.tile(range(atom.m*i, atom.m*(i+1)), range(atom.n*j, atom.n*(j+1)))
				))
			if n_steps > 1: result.append("")
		if m_steps > 1 and k_steps > 1: result.append("")
	return result

def preload(code, a_regs, b_regs):
	result = []
	assigned = {}
	next_a = 0
	next_b = 0
	for line in code:
		if not isinstance(line, AtomCall):
			result.append(line)
			continue

		if line.a in assigned:
			line.a = assigned[line.a]
		else:
			reg = a_regs[next_a]
			next_a = (next_a + 1) % len(a_regs)
			assigned[line.a] = reg
			result.append(Preload(line.a, reg))
			line.a = reg
		if line.b in assigned:
			line.b = assigned[line.b]
		else:
			reg = b_regs[next_b]
			next_b = (next_b + 1) % len(b_regs)
			assigned[line.b] = reg
			result.append(Preload(line.b, reg))
			line.b = reg
		result.append(line)
	return result

def schedule(code):
	for i in range(len(code)):
		if not isinstance(code[i], Preload):
			continue
		new_i = i
		for j in reversed(range(i)):
			# if line j doesn't use the register being preloaded at line i
			if not isinstance(code[j], AtomCall) or (code[j].a != code[i].register and code[j].b != code[i].register):
				new_i = j

		# move line to new_i
		code = code[:new_i] + [code[i]] + code[new_i:i] + code[i+1:]
	return code

atom = GemmAtom(
	"gemm",
	m=16,
	n=8,
	k=16,
	layout_a=Layout.RowMajor,
	layout_b=Layout.ColMajor,
	layout_c=Layout.ColMajor
)

code = []
# rScores=sK * sQ^T
code += gemm(
	Matrix("sK", 32, 192, Storage.Shared, Layout.RowMajor),
	Matrix("sQ", 8, 192, Storage.Shared, Layout.RowMajor).T(),
	Matrix("rScores", 32, 8, Storage.Register, Layout.ColMajor),
	atom
)
#print("---")
code += gemm(
	Matrix("sV", 32, 128, Storage.Shared, Layout.RowMajor).T(),
	Matrix("rScores", 32, 8, Storage.Register, Layout.ColMajor),
	Matrix("rO", 128, 8, Storage.Register, Layout.ColMajor),
	atom
)

code = preload(code, ["r0", "r1", "r2", "r3"], ["u0", "u1"])
code = schedule(code)
for line in code:
	print(line)
