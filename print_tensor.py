#!/usr/bin/env python3

import argparse
import itertools
from pathlib import Path

import torch


DEFAULT_VECTOR_SIZE = 32
DEFAULT_MATRIX_TILE = (8, 16)
F8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def load_safetensor(path: str | Path) -> torch.Tensor:
	from safetensors.torch import load_file

	path = Path(path)
	tensors = load_file(str(path), device="cpu")
	if len(tensors) != 1:
		raise ValueError(f"Expected exactly one tensor in {path}, found {len(tensors)}")
	return next(iter(tensors.values())).contiguous()


def load_tensor(path: str | Path) -> torch.Tensor:
	path = Path(path)
	if path.suffix == ".safetensors":
		return load_safetensor(path)
	raw = path.read_bytes()
	return torch.frombuffer(bytearray(raw), dtype=torch.int8)


def infer_shape(path: str, tensor: torch.Tensor, shape_args: list[int] | None) -> tuple[int, ...]:
	elem_count = tensor.numel()
	if elem_count == 0:
		raise ValueError(f"{path} is empty")

	if shape_args is None:
		return tuple(tensor.shape)

	shape = tuple(shape_args)
	if len(shape) == 0:
		raise ValueError("--shape must contain at least one dimension")
	if any(dim <= 0 for dim in shape):
		raise ValueError(f"Shape must be positive, got {shape}")

	shape_elems = 1
	for dim in shape:
		shape_elems *= dim
	if shape_elems != elem_count:
		raise ValueError(f"Provided shape {shape} has {shape_elems} elements, file has {elem_count}")

	return shape


def default_size(shape: tuple[int, ...]) -> tuple[int, ...]:
	if len(shape) == 1:
		return (min(DEFAULT_VECTOR_SIZE, shape[0]),)
	if len(shape) == 2:
		return (
			min(DEFAULT_MATRIX_TILE[0], shape[0]),
			min(DEFAULT_MATRIX_TILE[1], shape[1]),
		)
	return (
		*(1 for _ in shape[:-2]),
		min(DEFAULT_MATRIX_TILE[0], shape[-2]),
		min(DEFAULT_MATRIX_TILE[1], shape[-1]),
	)


def parse_args_for_rank(
	name: str,
	values: list[int] | None,
	rank: int,
	defaults: tuple[int, ...],
) -> tuple[int, ...]:
	if values is None:
		return defaults
	if len(values) != rank:
		raise ValueError(f"{name} expects {rank} value(s) for a rank-{rank} tensor, got {values}")
	return tuple(values)


def clamp_size(shape: tuple[int, ...], offset: tuple[int, ...], size: tuple[int, ...]) -> tuple[int, ...]:
	clamped: list[int] = []
	for dim, start, length in zip(shape, offset, size):
		if start < 0 or start >= dim:
			raise ValueError(f"Offset {offset} is outside shape {shape}")
		if length <= 0:
			raise ValueError(f"Tile size must be positive, got {size}")
		clamped.append(min(length, dim - start))
	return tuple(clamped)


def format_shape(shape: tuple[int, ...]) -> str:
	return f"[{', '.join(str(dim) for dim in shape)}]"


def printable_tensor(values: torch.Tensor) -> torch.Tensor:
	if F8_DTYPE is not None and values.dtype == F8_DTYPE:
		return values.to(torch.float32)
	if values.dtype in (torch.float16, torch.bfloat16):
		return values.to(torch.float32)
	return values


def format_value(value) -> str:
	if isinstance(value, float):
		return f"{value:.8g}"
	return str(value)


def print_vector(values: torch.Tensor, offset: int, length: int) -> None:
	indices = list(range(offset, offset + length))
	data = printable_tensor(values[offset : offset + length]).tolist()
	cells = [format_value(value) for value in data]
	cell_width = max(
		5,
		max(len(str(index)) for index in indices),
		max(len(cell) for cell in cells),
	)

	print(f"Slice: [{offset}:{offset + length})")
	print("index " + " ".join(f"{index:>{cell_width}}" for index in indices))
	print("value " + " ".join(f"{cell:>{cell_width}}" for cell in cells))


def print_matrix(values: torch.Tensor, row: int, col: int, height: int, width: int) -> None:
	tile = values[row : row + height, col : col + width]
	tile_rows = printable_tensor(tile).tolist()
	cell_rows = [[format_value(value) for value in tile_row] for tile_row in tile_rows]
	row_indices = list(range(row, row + height))
	col_indices = list(range(col, col + width))
	cell_width = max(
		4,
		max(len(str(index)) for index in col_indices),
		max(len(cell) for cell_row in cell_rows for cell in cell_row),
	)
	row_label_width = max(4, max(len(str(index)) for index in row_indices))

	print(f"Tile: rows [{row}:{row + height}), cols [{col}:{col + width})")
	header = " " * (row_label_width + 2) + " ".join(f"{index:>{cell_width}}" for index in col_indices)
	print(header)
	print("-" * len(header))
	for row_index, cell_row in zip(row_indices, cell_rows):
		formatted = " ".join(f"{cell:>{cell_width}}" for cell in cell_row)
		print(f"{row_index:>{row_label_width}}: {formatted}")


def print_tensor(values: torch.Tensor, offset: tuple[int, ...], size: tuple[int, ...]) -> None:
	rank = len(offset)
	if rank == 1:
		print_vector(values, offset[0], size[0])
		return
	if rank == 2:
		print_matrix(values, offset[0], offset[1], size[0], size[1])
		return

	prefix_ranges = [
		range(start, start + length)
		for start, length in zip(offset[:-2], size[:-2])
	]
	for slice_idx, prefix in enumerate(itertools.product(*prefix_ranges)):
		if slice_idx > 0:
			print()
		print(f"Slice prefix: {format_shape(prefix)}")
		print_matrix(values[prefix], offset[-2], offset[-1], size[-2], size[-1])


def main() -> None:
	parser = argparse.ArgumentParser(description="Print a tile from a tensor .bin or .safetensors file")
	parser.add_argument("tensor_file", help="Tensor .bin or .safetensors file to inspect")
	parser.add_argument(
		"--shape",
		type=int,
		nargs="+",
		default=None,
		help="Optional tensor shape. Raw .bin files are treated as flat int8 without this.",
	)
	parser.add_argument(
		"--offset",
		type=int,
		nargs="+",
		default=None,
		help="Tile start, one value per tensor dimension.",
	)
	parser.add_argument(
		"--size",
		type=int,
		nargs="+",
		default=None,
		help="Tile size, one value per tensor dimension.",
	)
	args = parser.parse_args()

	raw = load_tensor(args.tensor_file)
	shape = infer_shape(args.tensor_file, raw, args.shape)
	rank = len(shape)
	values = raw.reshape(shape)
	offset = parse_args_for_rank("--offset", args.offset, rank, (0,) * rank)
	size = parse_args_for_rank("--size", args.size, rank, default_size(shape))
	size = clamp_size(shape, offset, size)

	print(f"File: {args.tensor_file}")
	print(f"DType: {raw.dtype}")
	print(f"Shape: {format_shape(shape)}")
	print(f"Elements: {raw.numel()}")
	print()
	print_tensor(values, offset, size)


if __name__ == "__main__":
	main()
