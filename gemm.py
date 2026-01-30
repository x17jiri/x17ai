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

	def tile(self, rows, cols):
		if self.transposed:
			return "{name}.tile({cols_from}..<{cols_to}, {rows_from}..<{rows_to}).T".format(name=self.name, rows_from=rows.start, rows_to=rows.stop, cols_from=cols.start, cols_to=cols.stop)
		else:
			return "{name}.tile({rows_from}..<{rows_to}, {cols_from}..<{cols_to})".format(name=self.name, rows_from=rows.start, rows_to=rows.stop, cols_from=cols.start, cols_to=cols.stop)

class GemmAtom:
	def __init__(self, m, n, k, layout_a, layout_b, layout_c):
		self.m = m
		self.n = n
		self.k = k
		self.layout_a = layout_a
		self.layout_b = layout_b
		self.layout_c = layout_c

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

	for p in range(k_steps):
		for i in range(m_steps):
			for j in range(n_steps):
				print("gemm_atom({a}, {b}, {c})".format(
					a=a.tile(range(atom.m*i, atom.m*(i+1)), range(atom.k*p, atom.k*(p+1))),
					b=b.tile(range(atom.k*p, atom.k*(p+1)), range(atom.n*j, atom.n*(j+1))),
					c=c.tile(range(atom.m*i, atom.m*(i+1)), range(atom.n*j, atom.n*(j+1)))
				))
			if n_steps > 1: print()
		if m_steps > 1 and k_steps > 1: print()

atom = GemmAtom(
	m=16,
	n=8,
	k=16,
	layout_a=Layout.RowMajor,
	layout_b=Layout.ColMajor,
	layout_c=Layout.ColMajor
)

# rScores=sK * sQ^T
gemm(
	Matrix("sK", 32, 192, Storage.Shared, Layout.RowMajor),
	Matrix("sQ", 8, 192, Storage.Shared, Layout.RowMajor).T(),
	Matrix("rScores", 32, 8, Storage.Register, Layout.ColMajor),
	atom
)
print("---")
gemm(
	Matrix("sV", 32, 128, Storage.Shared, Layout.RowMajor).T(),
	Matrix("rScores", 32, 8, Storage.Register, Layout.ColMajor),
	Matrix("rO", 128, 8, Storage.Register, Layout.ColMajor),
	atom
)
