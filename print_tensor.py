#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


DEFAULT_VECTOR_SIZE = 32
DEFAULT_MATRIX_TILE = (8, 16)


def load_i8(path: str) -> torch.Tensor:
	raw = Path(path).read_bytes()
	return torch.frombuffer(bytearray(raw), dtype=torch.int8)


def infer_shape(path: str, shape_args: list[int] | None) -> tuple[int, ...]:
	elem_count = Path(path).stat().st_size
	if elem_count == 0:
		raise ValueError(f"{path} is empty")

	if shape_args is None:
		return (elem_count,)

	shape = tuple(shape_args)
	if len(shape) not in (1, 2):
		raise ValueError("--shape currently supports 1D or 2D tensors only")
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
	return (
		min(DEFAULT_MATRIX_TILE[0], shape[0]),
		min(DEFAULT_MATRIX_TILE[1], shape[1]),
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


def print_vector(values: torch.Tensor, offset: int, length: int) -> None:
	indices = list(range(offset, offset + length))
	data = values[offset : offset + length].tolist()
	cell_width = max(
		5,
		max(len(str(index)) for index in indices),
		max(len(str(value)) for value in data),
	)

	print(f"Slice: [{offset}:{offset + length})")
	print("index " + " ".join(f"{index:>{cell_width}}" for index in indices))
	print("value " + " ".join(f"{value:>{cell_width}}" for value in data))


def print_matrix(values: torch.Tensor, row: int, col: int, height: int, width: int) -> None:
	tile = values[row : row + height, col : col + width]
	tile_rows = tile.tolist()
	row_indices = list(range(row, row + height))
	col_indices = list(range(col, col + width))
	cell_width = max(
		4,
		max(len(str(index)) for index in col_indices),
		max(len(str(value)) for tile_row in tile_rows for value in tile_row),
	)
	row_label_width = max(4, max(len(str(index)) for index in row_indices))

	print(f"Tile: rows [{row}:{row + height}), cols [{col}:{col + width})")
	print(" " * (row_label_width + 2) + " ".join(f"{index:>{cell_width}}" for index in col_indices))
	for row_index, tile_row in zip(row_indices, tile_rows):
		formatted = " ".join(f"{value:>{cell_width}}" for value in tile_row)
		print(f"{row_index:>{row_label_width}}: {formatted}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Print a tile from a raw int8 tensor .bin file")
	parser.add_argument("tensor_file", help="Tensor .bin file to inspect")
	parser.add_argument(
		"--shape",
		type=int,
		nargs="+",
		default=None,
		help="Optional tensor shape. Supports 1D or 2D tensors.",
	)
	parser.add_argument(
		"--offset",
		type=int,
		nargs="+",
		default=None,
		help="Tile start. For 1D: index. For 2D: row col.",
	)
	parser.add_argument(
		"--size",
		type=int,
		nargs="+",
		default=None,
		help="Tile size. For 1D: length. For 2D: height width.",
	)
	args = parser.parse_args()

	shape = infer_shape(args.tensor_file, args.shape)
	rank = len(shape)
	raw = load_i8(args.tensor_file)
	values = raw.reshape(shape)
	offset = parse_args_for_rank("--offset", args.offset, rank, (0,) * rank)
	size = parse_args_for_rank("--size", args.size, rank, default_size(shape))
	size = clamp_size(shape, offset, size)

	print(f"File: {args.tensor_file}")
	print(f"Shape: {format_shape(shape)}")
	print(f"Elements: {raw.numel()}")
	print()

	if rank == 1:
		print_vector(values, offset[0], size[0])
		return

	print_matrix(values, offset[0], offset[1], size[0], size[1])


if __name__ == "__main__":
	main()
