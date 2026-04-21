#!/usr/bin/env python3

import argparse
import math
import os
from pathlib import Path

import torch

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
	import matplotlib
	matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_bf16(path: str) -> torch.Tensor:
	raw = Path(path).read_bytes()
	return torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)


def load_f32(path: str) -> torch.Tensor:
	raw = Path(path).read_bytes()
	return torch.frombuffer(bytearray(raw), dtype=torch.float32)


def load_tensor(path: str, dtype: str) -> torch.Tensor:
	if dtype == "bf16":
		return load_bf16(path).to(torch.float32)
	if dtype == "f32":
		return load_f32(path)
	raise ValueError(f"Unsupported dtype: {dtype}")


def infer_dtype_from_path(path: str) -> str:
	return "f32" if "_f32" in Path(path).stem else "bf16"


def default_output_path(input_path: str) -> Path:
	path = Path(input_path)
	if path.suffix:
		return path.with_suffix(".stats.png")
	return Path(f"{path}.stats.png")


def standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
	return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def finite_values(tensor: torch.Tensor) -> torch.Tensor:
	return tensor[torch.isfinite(tensor)]


def plot_tensor_stats(values: torch.Tensor, bins: int, title: str) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(10, 6))
	values_np = values.numpy()
	data_min = float(values.min().item())
	data_max = float(values.max().item())
	x_min = min(data_min, -4.0)
	x_max = max(data_max, 4.0)
	if x_min == x_max:
		x_min -= 1.0
		x_max += 1.0
	x = torch.linspace(x_min, x_max, 512)
	y = standard_normal_pdf(x)

	ax.hist(values_np, bins=bins, density=True, alpha=0.65, color="#4C72B0", label="tensor values")
	ax.plot(x.numpy(), y.numpy(), color="#C44E52", linewidth=2.0, label="N(0, 1)")
	ax.set_title(title)
	ax.set_xlabel("value")
	ax.set_ylabel("density")
	ax.grid(True, alpha=0.2)
	ax.legend()
	fig.tight_layout()
	return fig


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot histogram statistics for a tensor .bin file")
	parser.add_argument("tensor_file", help="Tensor .bin file to inspect")
	parser.add_argument("--dtype", choices=["bf16", "f32"], default=None, help="Tensor element type")
	parser.add_argument("--bins", type=int, default=200, help="Histogram bin count")
	parser.add_argument("--title", default=None, help="Plot title")
	parser.add_argument("--output", default=None, help="Optional output .png path")
	args = parser.parse_args()

	dtype = args.dtype or infer_dtype_from_path(args.tensor_file)
	tensor = load_tensor(args.tensor_file, dtype).reshape(-1)
	finite = finite_values(tensor)
	if finite.numel() == 0:
		raise ValueError(f"{args.tensor_file} does not contain any finite values")

	mean = float(finite.mean().item())
	std = float(finite.std(unbiased=False).item())
	var = float(finite.var(unbiased=False).item())
	print(f"Loaded {args.tensor_file}")
	print(f"dtype: {dtype}")
	print(f"elements: {tensor.numel()}")
	print(f"finite elements: {finite.numel()}")
	print(f"mean: {mean:.6e}")
	print(f"std: {std:.6e}")
	print(f"var: {var:.6e}")

	title = args.title or f"{Path(args.tensor_file).name}  mean={mean:.4f}  var={var:.4f}"
	fig = plot_tensor_stats(finite, args.bins, title)

	output_path = Path(args.output) if args.output is not None else None
	if output_path is None and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
		output_path = default_output_path(args.tensor_file)

	if output_path is not None:
		fig.savefig(output_path, dpi=150)
		print(f"Saved plot to {output_path}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
