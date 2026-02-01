
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

	def __eq__(self, other):
		if not isinstance(other, Matrix):
			return False
		return (self.name == other.name and
				self.m == other.m and
				self.n == other.n and
				self.storage == other.storage and
				self.layout == other.layout and
				self.transposed == other.transposed)

	def __hash__(self):
		return hash((
			self.name,
			self.m,
			self.n,
			self.storage,
			self.layout,
			self.transposed
		))

class Tile:
	def __init__(self, matrix: Matrix, row_range: range, col_range: range):
		self.matrix = matrix
		self.row_range = row_range
		self.col_range = col_range

	def __str__(self):
		result = self.matrix.name
		m = self.matrix.m
		n = self.matrix.n
		if self.matrix.transposed:
			result += ".transposed()"
			m, n = n, m

		tile_row_cnt = self.row_range.stop - self.row_range.start
		tile_row_idx = self.row_range.start // tile_row_cnt
		tiles_m = m // tile_row_cnt

		tile_col_cnt = self.col_range.stop - self.col_range.start
		tile_col_idx = self.col_range.start // tile_col_cnt
		tiles_n = n // tile_col_cnt

		result += ".tile<{tile_row_cnt}, {tile_col_cnt}>({tile_row_idx} /*..{tiles_m}*/, {tile_col_idx} /*..{tiles_n}*/)".format(
			tile_row_cnt=tile_row_cnt,
			tile_col_cnt=tile_col_cnt,
			tile_row_idx=tile_row_idx,
			tiles_m=tiles_m,
			tile_col_idx=tile_col_idx,
			tiles_n=tiles_n
		)
		return result

	def __eq__(self, other):
		if not isinstance(other, Tile):
			return False
		return (self.matrix == other.matrix and
				self.row_range == other.row_range and
				self.col_range == other.col_range)

	def __hash__(self):
		return hash((
			self.matrix,
			self.row_range.start,
			self.row_range.stop,
			self.col_range.start,
			self.col_range.stop
		))

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
		self.dist = 0

	def __str__(self):
		return "{reg} = {tile} // dist = {dist}".format(
			reg=self.register,
			tile=self.tile,
			dist=self.dist
		)

class Title:
	def __init__(self, title: str):
		self.title = title

	def __str__(self):
		return "//***** " + self.title + " *****//"

class Code:
	def __init__(self):
		self.lines = []

	def gemm(self, title, a: Matrix, b: Matrix, c: Matrix, atom: GemmAtom):
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
		result.append(Title(title))
		result.append("//*** {c} += {a} x {b}".format(
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
		self.lines += result

	def preload(self, a_regs, b_regs):
		self.preload = []
		for _ in self.lines: self.preload.append([])
		assigned = {}
		next_a = 0
		next_b = 0
		for i in range(len(self.lines)):
			line = self.lines[i]
			if not isinstance(line, AtomCall):
				if line == "//++a":
					next_a += 1
				elif line == "//++b":
					next_b += 1
				continue

			if line.a.matrix.storage == Storage.Shared:
				if line.a in assigned:
					line.a = assigned[line.a]
				else:
					reg = a_regs[next_a]
					next_a = (next_a + 1) % len(a_regs)
					assigned[line.a] = reg
					self.preload[i].append(Preload(line.a, reg))
					line.a = reg
			if line.b.matrix.storage == Storage.Shared:
				if line.b in assigned:
					line.b = assigned[line.b]
				else:
					reg = b_regs[next_b]
					next_b = (next_b + 1) % len(b_regs)
					assigned[line.b] = reg
					self.preload[i].append(Preload(line.b, reg))
					line.b = reg

	def schedule_line(self, i, preload_next_loop):
		lines = self.lines
		p_list  = self.preload[i]
		self.preload[i] = []
		insertions = []
		for p in p_list:
			mat_name = p.tile.matrix.name
			new_i = i
			for j in reversed(range(i)):
				# if line j doesn't use the register being preloaded at line i
				register = p.register
				if not isinstance(lines[j], AtomCall):
					new_i = j
					if lines[j] == "//++dist":
						p.dist += 1
				elif lines[j].a != register and lines[j].b != register:
					new_i = j
					p.dist += 1
				else:
					break
				# ...
				if mat_name in preload_next_loop:
					for prev_p in self.preload[i]:
						if prev_p.tile.matrix.name == mat_name:
							break;

			# move line to new_i
			insertions.append((new_i, p))

		for mat_name in preload_next_loop:
			mat_preloads = [i for (i, p) in insertions if p.tile.matrix.name == mat_name]
			if len(mat_preloads) > 0:
				mat_preloads.sort()
				self.preload[mat_preloads[0]].append("")
				self.preload[mat_preloads[0]].append(
					"//*********** Preload {mat} for the next loop ***".format(mat=mat_name)
				)
				self.preload[mat_preloads[0]].append(preload_next_loop[mat_name])
				self.preload[mat_preloads[0]].append("")

		for (i, p) in insertions:
			self.preload[i].append(p)

	def schedule(self):
		lines = self.lines
		preload_next_loop = {}
		for i in range(len(lines)):
			self.schedule_line(i, preload_next_loop)

	def loop(self, preload_next_loop):
		self.lines.append("")
		self.preload.append(self.preload[0][:])
		self.schedule_line(len(self.lines)-1, preload_next_loop)

	def finish(self):
		for i in range(len(self.preload)):
			p_list = self.preload[i]
			new_list = []
			j = 0
			t = 0
			while j < len(p_list):
				t = j
				while t < len(p_list) and isinstance(p_list[t], Preload):
					t += 1
				if t > j:
					to_add = p_list[j:t]
					to_add.sort(key=lambda x: x.dist)
					new_list += to_add
					j = t
				else:
					new_list.append(p_list[t])
					j += 1
			self.preload[i] = new_list

		lines = self.lines
		result = [Title("Init")]
		for i in range(len(lines)):
			result += self.preload[i] + [lines[i]]
		self.lines = result

		current = ""
		self.sections = {}
		for line in result:
			if isinstance(line, Title):
				current = line.title
				self.sections[current] = [line]
			else:
				self.sections[current].append(line)

atom = GemmAtom(
	"gemm",
	m=16,
	n=8,
	k=16,
	layout_a=Layout.RowMajor,
	layout_b=Layout.ColMajor,
	layout_c=Layout.ColMajor
)

code = Code()
# rScores=sK * sQ^T
code.gemm(
	"GEMM 1",
	Matrix("sKV", 32, 192, Storage.Shared, Layout.RowMajor),
	Matrix("sQ", 8, 192, Storage.Shared, Layout.RowMajor).T(),
	Matrix("rScores", 32, 8, Storage.Register, Layout.ColMajor),
	atom
)
code.lines.append("//++a")
code.lines.append("//++a")
#code.lines.append("//++dist")
code.lines.append("//++dist")
#print("---")
code.gemm(
	"GEMM 2",
	Matrix("sKV", 32, 128, Storage.Shared, Layout.RowMajor).T(),
	Matrix("rScores", 32, 8, Storage.Register, Layout.ColMajor),
	Matrix("rO", 128, 8, Storage.Register, Layout.ColMajor),
	atom
)

code.preload(["r0", "r1", "r2"], ["u0", "u1"])
code.schedule()
code.loop({"sKV": "123"})
code.finish()
#for line in code.lines:
#	print(line)
delay = 0.0
for section_name in code.sections:
	section = code.sections[section_name]
	print("//=============================================")
	for line in section:
		print(line)
		if section_name != "Init" and isinstance(line, Preload):
			delay += max(0, 2-line.dist)

print("// delay per loop: {delay}".format(delay=delay))
