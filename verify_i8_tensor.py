#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def load_i8(path: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
	raw = Path(path).read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.int8)
	if shape is not None:
		return data.reshape(shape)
	return data


def infer_shape(file_a: str, file_b: str, shape_args: list[int] | None) -> tuple[int, ...]:
	bytes_a = Path(file_a).stat().st_size
	bytes_b = Path(file_b).stat().st_size
	if bytes_a != bytes_b:
		raise ValueError(f"File sizes differ: {file_a}={bytes_a} bytes, {file_b}={bytes_b} bytes")

	elem_count = bytes_a
	if shape_args is None:
		return (elem_count,)

	shape = tuple(shape_args)
	shape_elems = 1
	for dim in shape:
		shape_elems *= dim
	if shape_elems != elem_count:
		raise ValueError(f"Provided shape {shape} has {shape_elems} elements, file has {elem_count}")
	return shape


def format_shape(shape: tuple[int, ...]) -> str:
	return f"[{', '.join(str(dim) for dim in shape)}]"


def verify_i8(file_a: str, file_b: str, shape: tuple[int, ...]) -> None:
	a = load_i8(file_a, shape)
	b = load_i8(file_b, shape)

	total = a.numel()
	exact_match = int((a == b).sum().item())
	mismatch_count = total - exact_match
	diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
	max_abs_diff = int(diff.max().item())
	min_a = int(a.min().item())
	max_a = int(a.max().item())
	min_b = int(b.min().item())
	max_b = int(b.max().item())

	out_of_range_a = int(((a < -126) | (a > 126)).sum().item())
	out_of_range_b = int(((b < -126) | (b > 126)).sum().item())
	outside_100_a = int(((a < -100) | (a > 100)).sum().item())
	outside_100_b = int(((b < -100) | (b > 100)).sum().item())

	print("\n--- I8 Tensor Compare ---")
	print(f"A: {file_a}")
	print(f"B: {file_b}")
	print(f"Shape: {format_shape(shape)}")
	print(f"A min/max:      {min_a} / {max_a}")
	print(f"B min/max:      {min_b} / {max_b}")
	print(f"Mismatched i8:  {mismatch_count}/{total} ({100.0 * mismatch_count / total:.2f}%)")
	print(f"Max abs diff:   {max_abs_diff}")
	print(f"A outside [-100, +100]: {outside_100_a}")
	print(f"B outside [-100, +100]: {outside_100_b}")
	print(f"A outside [-126, +126]: {out_of_range_a}")
	print(f"B outside [-126, +126]: {out_of_range_b}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Compare two int8 tensor .bin files")
	parser.add_argument("file_a", help="Reference tensor .bin file")
	parser.add_argument("file_b", help="Other tensor .bin file")
	parser.add_argument("--shape", type=int, nargs="+", default=None, help="Optional tensor shape")
	args = parser.parse_args()

	shape = infer_shape(args.file_a, args.file_b, args.shape)
	verify_i8(args.file_a, args.file_b, shape)


if __name__ == "__main__":
	main()
