#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def load_i8(path: str | Path, shape: tuple[int, ...] | None = None) -> torch.Tensor:
	path = Path(path)
	if path.suffix == ".safetensors":
		from safetensors.torch import load_file

		tensors = load_file(str(path), device="cpu")
		if len(tensors) != 1:
			raise ValueError(f"Expected exactly one tensor in {path}, found {len(tensors)}")
		data = next(iter(tensors.values())).contiguous()
		if data.dtype != torch.int8:
			raise ValueError(f"Expected {path} to contain int8 tensor, got {data.dtype}")
		if shape is not None:
			return data.reshape(shape)
		return data

	raw = path.read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.int8)
	if shape is not None:
		return data.reshape(shape)
	return data


def elem_count(shape: tuple[int, ...]) -> int:
	count = 1
	for dim in shape:
		count *= dim
	return count


def file_shape(path: str) -> tuple[int, ...]:
	path_obj = Path(path)
	if path_obj.suffix == ".safetensors":
		return tuple(load_i8(path_obj).shape)
	return (path_obj.stat().st_size,)


def infer_shape(file_a: str, file_b: str, shape_args: list[int] | None) -> tuple[int, ...]:
	shape_a = file_shape(file_a)
	shape_b = file_shape(file_b)
	elems_a = elem_count(shape_a)
	elems_b = elem_count(shape_b)
	if elems_a != elems_b:
		raise ValueError(f"Element counts differ: {file_a}={elems_a}, {file_b}={elems_b}")

	if shape_args is None:
		if shape_a == shape_b:
			return shape_a
		if Path(file_a).suffix == ".safetensors":
			return shape_a
		if Path(file_b).suffix == ".safetensors":
			return shape_b
		return (elems_a,)

	shape = tuple(shape_args)
	shape_elems = elem_count(shape)
	if shape_elems != elems_a:
		raise ValueError(f"Provided shape {shape} has {shape_elems} elements, files have {elems_a}")
	return shape


def format_shape(shape: tuple[int, ...]) -> str:
	return f"[{', '.join(str(dim) for dim in shape)}]"


def verify_i8(file_a: str, file_b: str, shape: tuple[int, ...]) -> None:
	a = load_i8(file_a, shape)
	b = load_i8(file_b, shape)
	a_f64 = a.to(torch.float64)
	b_f64 = b.to(torch.float64)

	total = a.numel()
	exact_match = int((a == b).sum().item())
	mismatch_count = total - exact_match
	diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
	max_abs_diff = int(diff.max().item())
	diff_counts = torch.bincount(diff.flatten().to(torch.int64), minlength=11)
	min_a = int(a.min().item())
	max_a = int(a.max().item())
	min_b = int(b.min().item())
	max_b = int(b.max().item())
	mean_a = a_f64.mean().item()
	var_a = a_f64.var(unbiased=False).item()
	mean_b = b_f64.mean().item()
	var_b = b_f64.var(unbiased=False).item()

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
	print(f"A mean/var:     {mean_a:.6e} / {var_a:.6e}")
	print(f"B mean/var:     {mean_b:.6e} / {var_b:.6e}")
	print(f"Mismatched i8:  {mismatch_count}/{total} ({100.0 * mismatch_count / total:.2f}%)")
	print(f"Max abs diff:   {max_abs_diff}")
	for abs_diff in range(1, 1+max_abs_diff):
		count = int(diff_counts[abs_diff].item())
		if count != 0:
			print(f"Abs diff == {abs_diff}: {count}/{total} ({100.0 * count / total:.2f}%)")
	print(f"A outside [-100, +100]: {outside_100_a}")
	print(f"B outside [-100, +100]: {outside_100_b}")
	print(f"A outside [-126, +126]: {out_of_range_a}")
	print(f"B outside [-126, +126]: {out_of_range_b}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Compare two int8 tensor .bin or .safetensors files")
	parser.add_argument("file_a", help="Reference tensor file")
	parser.add_argument("file_b", help="Other tensor file")
	parser.add_argument("--shape", type=int, nargs="+", default=None, help="Optional tensor shape")
	args = parser.parse_args()

	shape = infer_shape(args.file_a, args.file_b, args.shape)
	verify_i8(args.file_a, args.file_b, shape)


if __name__ == "__main__":
	main()
