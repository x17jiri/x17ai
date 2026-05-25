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


def load_i8(path: str) -> torch.Tensor:
	raw = Path(path).read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.int8)
	scaled = data.to(torch.float32) / 8.0
	return torch.where(data == -128, torch.full_like(scaled, math.nan), scaled)


def load_tensor(path: str, dtype: str) -> torch.Tensor:
	if dtype == "bf16":
		return load_bf16(path).to(torch.float32)
	if dtype == "f32":
		return load_f32(path)
	if dtype == "i8":
		return load_i8(path)
	raise ValueError(f"Unsupported dtype: {dtype}")


def infer_dtype_from_path(path: str) -> str:
	stem = Path(path).stem
	if "_f32" in stem:
		return "f32"
	if "_i8" in stem:
		return "i8"
	return "bf16"


def default_output_path(input_path: str) -> Path:
	path = Path(input_path)
	if path.suffix:
		return path.with_suffix(".stats.png")
	return Path(f"{path}.stats.png")


def read_expected_variance(path: str) -> float:
	return float(Path(path).read_text(encoding="ascii").strip())


def standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
	return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
	return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def normalize_values(values: torch.Tensor, reference_variance: float) -> torch.Tensor:
	if reference_variance <= 0.0:
		return values
	return values / math.sqrt(reference_variance)


def finite_values(tensor: torch.Tensor) -> torch.Tensor:
	return tensor[torch.isfinite(tensor)]


def summarize_values(values: torch.Tensor) -> tuple[float, float, float]:
	mean = float(values.mean().item())
	std = float(values.std(unbiased=False).item())
	var = float(values.var(unbiased=False).item())
	return mean, std, var


def plotted_quantization_step(dtype: str, reference_variance: float) -> float | None:
	if dtype != "i8":
		return None
	step = 1.0 / 8.0
	if reference_variance > 0.0:
		step /= math.sqrt(reference_variance)
	return step


def discrete_density(values: torch.Tensor, step: float) -> tuple[torch.Tensor, torch.Tensor]:
	centers, counts = torch.unique(values, sorted=True, return_counts=True)
	density = counts.to(torch.float32) / (values.numel() * step)
	return centers.to(torch.float32), density


def discrete_reference_density(centers: torch.Tensor, step: float) -> torch.Tensor:
	half_step = 0.5 * step
	mass = standard_normal_cdf(centers + half_step) - standard_normal_cdf(centers - half_step)
	return mass / step


def reference_centers(x_min: float, x_max: float, step: float) -> torch.Tensor:
	start = math.floor(x_min / step) * step
	stop = math.ceil(x_max / step) * step
	count = int(round((stop - start) / step)) + 1
	return start + step * torch.arange(count, dtype=torch.float32)


def load_tensor_summary(tensor_file: str, dtype: str, var_file: str | None) -> dict[str, object]:
	tensor = load_tensor(tensor_file, dtype).reshape(-1)
	finite = finite_values(tensor)
	if finite.numel() == 0:
		raise ValueError(f"{tensor_file} does not contain any finite values")

	reference_variance = 1.0
	if var_file is not None:
		reference_variance = read_expected_variance(var_file)
	plotted_values = normalize_values(finite, reference_variance)
	plotted_step = plotted_quantization_step(dtype, reference_variance)
	mean, std, var = summarize_values(finite)
	plotted_mean, plotted_std, plotted_var = summarize_values(plotted_values)
	return {
		"tensor_file": tensor_file,
		"name": Path(tensor_file).name,
		"dtype": dtype,
		"elements": tensor.numel(),
		"finite_elements": finite.numel(),
		"mean": mean,
		"std": std,
		"var": var,
		"reference_variance": reference_variance,
		"plotted_values": plotted_values,
		"plotted_step": plotted_step,
		"plotted_mean": plotted_mean,
		"plotted_std": plotted_std,
		"plotted_var": plotted_var,
	}


def print_tensor_summary(prefix: str, summary: dict[str, object]) -> None:
	print(f"{prefix} {summary['tensor_file']}")
	print(f"dtype: {summary['dtype']}")
	print(f"elements: {summary['elements']}")
	print(f"finite elements: {summary['finite_elements']}")
	print(f"mean: {summary['mean']:.6e}")
	print(f"std: {summary['std']:.6e}")
	print(f"var: {summary['var']:.6e}")
	print(f"reference variance: {summary['reference_variance']:.6e}")
	if summary["reference_variance"] > 0.0:
		print(f"normalized mean: {summary['plotted_mean']:.6e}")
		print(f"normalized std: {summary['plotted_std']:.6e}")
		print(f"normalized var: {summary['plotted_var']:.6e}")


def plot_tensor_stats(
	series: list[tuple[str, torch.Tensor, float | None]],
	bins: int,
	title: str,
	show_standard_normal: bool,
	show_zero_reference: bool,
	x_label: str,
) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(10, 6))
	x_min = -4.0
	x_max = 4.0
	x = torch.linspace(x_min, x_max, 512)

	colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
	all_discrete = all(step is not None for _, _, step in series)
	shared_step = None
	if all_discrete:
		first_step = series[0][2]
		assert first_step is not None
		if all(abs(step - first_step) < 1e-12 for _, _, step in series):
			shared_step = first_step
	if len(series) == 1:
		label, values, step = series[0]
		if step is None:
			ax.hist(values.numpy(), bins=bins, density=True, alpha=0.65, color=colors[0], label=label)
		else:
			centers, density = discrete_density(values, step)
			ax.bar(
				centers.numpy(),
				density.numpy(),
				width=0.9 * step,
				alpha=0.65,
				color=colors[0],
				align="center",
				label=label,
			)
	else:
		for idx, (label, values, step) in enumerate(series):
			if step is None:
				ax.hist(
					values.numpy(),
					bins=bins,
					density=True,
					histtype="step",
					linewidth=2.0,
					color=colors[idx % len(colors)],
					label=label,
				)
			else:
				centers, density = discrete_density(values, step)
				ax.plot(
					centers.numpy(),
					density.numpy(),
					drawstyle="steps-mid",
					linewidth=2.0,
					color=colors[idx % len(colors)],
					label=label,
				)
	if show_standard_normal:
		if shared_step is not None:
			centers = reference_centers(x_min, x_max, shared_step)
			y = discrete_reference_density(centers, shared_step)
			ax.plot(
				centers.numpy(),
				y.numpy(),
				color="#DD8452",
				linewidth=2.0,
				label="N(0, 1) avg/bin",
			)
		else:
			y = standard_normal_pdf(x)
			ax.plot(
				x.numpy(),
				y.numpy(),
				color="#DD8452",
				linewidth=2.0,
				label="N(0, 1)",
			)
	elif show_zero_reference:
		ax.axvline(0.0, color="#C44E52", linewidth=2.0, label="expected variance = 0")
	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel("density")
	ax.set_xlim(x_min, x_max)
	ax.grid(True, alpha=0.2)
	ax.legend()
	fig.tight_layout()
	return fig


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot histogram statistics for a tensor .bin file")
	parser.add_argument("tensor_file", help="Tensor .bin file to inspect")
	parser.add_argument("var_file", nargs="?", default=None, help="Optional .var file containing the expected variance")
	parser.add_argument("--dtype", choices=["bf16", "f32", "i8"], default=None, help="Tensor element type")
	parser.add_argument("--overlay", default=None, help="Optional second tensor .bin file to overlay")
	parser.add_argument("--overlay-var", default=None, help="Optional .var file for the overlay tensor")
	parser.add_argument("--overlay-dtype", choices=["bf16", "f32", "i8"], default=None, help="Overlay tensor element type")
	parser.add_argument("--bins", type=int, default=150, help="Histogram bin count")
	parser.add_argument("--title", default=None, help="Plot title")
	parser.add_argument("--output", default=None, help="Optional output .png path")
	args = parser.parse_args()

	dtype = args.dtype or infer_dtype_from_path(args.tensor_file)
	primary = load_tensor_summary(args.tensor_file, dtype, args.var_file)
	print_tensor_summary("Loaded", primary)

	summaries = [primary]
	if args.overlay is not None:
		overlay_dtype = args.overlay_dtype or infer_dtype_from_path(args.overlay)
		overlay = load_tensor_summary(args.overlay, overlay_dtype, args.overlay_var)
		print()
		print_tensor_summary("Overlay", overlay)
		summaries.append(overlay)

	positive_reference_count = sum(summary["reference_variance"] > 0.0 for summary in summaries)
	show_standard_normal = positive_reference_count == len(summaries)
	show_zero_reference = positive_reference_count == 0
	if show_standard_normal:
		x_label = "value / sqrt(ref_var)"
	elif show_zero_reference:
		x_label = "value"
	else:
		x_label = "value (normalized per tensor when ref_var > 0)"

	series = []
	for summary in summaries:
		label = summary["name"]
		if summary["reference_variance"] > 0.0:
			label = f"{label} (norm_var={summary['plotted_var']:.4f})"
		series.append((label, summary["plotted_values"], summary["plotted_step"]))

	if args.overlay is None:
		if primary["reference_variance"] > 0.0:
			title = args.title or (
				f"{primary['name']}  norm_mean={primary['plotted_mean']:.4f}"
				f"  norm_var={primary['plotted_var']:.4f}  raw_var={primary['var']:.4f}"
			)
		else:
			title = args.title or (
				f"{primary['name']}  mean={primary['mean']:.4f}  var={primary['var']:.4f}"
				f"  ref_var={primary['reference_variance']:.4f}"
			)
	else:
		overlay = summaries[1]
		title = args.title or (
			f"{primary['name']} vs {overlay['name']}  "
			f"norm_var={primary['plotted_var']:.4f}/{overlay['plotted_var']:.4f}"
		)

	fig = plot_tensor_stats(series, args.bins, title, show_standard_normal, show_zero_reference, x_label)

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
