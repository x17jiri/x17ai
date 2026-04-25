#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import torch

import block

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
	import matplotlib
	matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TENSOR_DIR = ROOT / "tmp" / "block_torch"


def load_bf16(path: Path) -> torch.Tensor:
	raw = path.read_bytes()
	return torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16).to(torch.float32)


def infer_q_len_from_attn_out(path: Path) -> int:
	element_count = path.stat().st_size // 2
	row_width = block.Q_ROWS
	if element_count % row_width != 0:
		raise ValueError(
			f"Expected {path} to contain rows of width {row_width}, got {element_count} bf16 values"
		)
	return element_count // row_width


def read_attn_out_rows(handle, row_start: int, row_count: int) -> torch.Tensor:
	row_width = block.Q_ROWS
	byte_offset = row_start * row_width * 2
	handle.seek(byte_offset)
	raw = handle.read(row_count * row_width * 2)
	if len(raw) != row_count * row_width * 2:
		raise ValueError(
			f"Failed to read rows [{row_start}, {row_start + row_count}) from attn_out_pregate"
		)
	data = torch.frombuffer(bytearray(raw), dtype=torch.int16)
	return data.view(torch.bfloat16).to(torch.float32).view(row_count, block.N_HEADS, block.HEAD_DIM)


def compute_windowed_variances(
	attn_out_path: Path,
	head: int | None,
	window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
	q_len = infer_q_len_from_attn_out(attn_out_path)
	if head is not None and (head < 0 or head >= block.N_HEADS):
		raise ValueError(f"Head index {head} is out of range for {block.N_HEADS} heads")

	positions = []
	variances = []
	windows = []

	with attn_out_path.open("rb") as handle:
		for row_start in range(0, q_len, window_size):
			row_end = min(row_start + window_size, q_len)
			row_count = row_end - row_start
			attn_out = read_attn_out_rows(handle, row_start, row_count)
			if head is not None:
				window_values = attn_out[:, head, :]
			else:
				window_values = attn_out

			positions.append(row_start + 0.5 * (row_count - 1))
			variances.append(window_values.var(unbiased=False).item())
			windows.append((row_start, row_end))

	return torch.tensor(positions), torch.tensor(variances), windows


def default_output_path(head: int | None, window_size: int) -> Path:
	suffix = f"head{head}" if head is not None else "all_heads"
	return TENSOR_DIR / f"attn_out_pregate_{suffix}_window{window_size}.png"


def plot_variances(positions: torch.Tensor, variances: torch.Tensor, head: int | None, window_size: int) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(10, 6))
	mean_variance = variances.mean().item()
	head_label = f"head {head}" if head is not None else "all heads"

	ax.plot(
		positions.numpy(),
		variances.numpy(),
		marker="o",
		markersize=4.0,
		linewidth=1.5,
		color="#4C72B0",
		label="window variance",
	)
	ax.axhline(mean_variance, color="#C44E52", linestyle="--", linewidth=1.5, label=f"mean={mean_variance:.4f}")
	ax.set_title(f"attn_out_pregate variance by Q position ({head_label}, window {window_size})")
	ax.set_xlabel("Q position (window center)")
	ax.set_ylabel("variance")
	ax.set_ylim(bottom=0.0)
	ax.grid(True, alpha=0.2)
	ax.legend()
	fig.tight_layout()
	return fig


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot per-window variance of attn_out_pregate to test SSMax stabilization across Q positions"
	)
	parser.add_argument("--head", type=int, default=None, help="Optional attention head to inspect; defaults to all heads")
	parser.add_argument("--window", type=int, default=32, help="Number of Q rows per variance window")
	parser.add_argument(
		"--tensor-file",
		default=str(TENSOR_DIR / "attn_out_pregate.bin"),
		help="Path to attn_out_pregate.bin",
	)
	parser.add_argument("--output", default=None, help="Optional output image path")
	args = parser.parse_args()

	if args.window <= 0:
		raise ValueError(f"Expected a positive window size, got {args.window}")

	tensor_path = Path(args.tensor_file)

	if not tensor_path.exists():
		raise FileNotFoundError(f"Missing attn_out_pregate artifact: {tensor_path}")

	positions, variances, windows = compute_windowed_variances(
		tensor_path,
		args.head,
		args.window,
	)
	q_len = infer_q_len_from_attn_out(tensor_path)

	print(f"tensor file: {tensor_path}")
	print(f"head: {'all' if args.head is None else args.head}")
	print(f"q_len: {q_len}")
	print(f"window size: {args.window}")
	print(f"windows: {len(windows)}")
	print(f"variance mean: {variances.mean().item():.6e}")
	print(f"variance min:  {variances.min().item():.6e}")
	print(f"variance max:  {variances.max().item():.6e}")
	print()
	print("row_start row_end variance")
	for (row_start, row_end), variance in zip(windows, variances.tolist()):
		print(f"{row_start:8d} {row_end:7d} {variance:.6e}")

	fig = plot_variances(positions, variances, args.head, args.window)
	output_path = Path(args.output) if args.output is not None else None
	if output_path is None and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
		output_path = default_output_path(args.head, args.window)

	if output_path is not None:
		fig.savefig(output_path, dpi=150)
		print()
		print(f"Saved plot to {output_path}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
