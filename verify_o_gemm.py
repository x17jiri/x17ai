#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import torch

DEFAULT_A_ROWS = 2048
DEFAULT_K = 4096
DEFAULT_B_COLS = 32768
CONFIG_PATH = "tmp/o_gemm.config.json"
INPUT_A_PATH = "tmp/o_a.bin"
INPUT_B_PATH = "tmp/o_b.bin"
OUTPUT_PATH = "tmp/o_out_cpu.bin"


def load_config():
	if not os.path.exists(CONFIG_PATH):
		return DEFAULT_A_ROWS, DEFAULT_K, DEFAULT_B_COLS
	try:
		with open(CONFIG_PATH, "r", encoding="ascii") as config_file:
			config = json.load(config_file)
	except (OSError, ValueError, TypeError, json.JSONDecodeError):
		return DEFAULT_A_ROWS, DEFAULT_K, DEFAULT_B_COLS
	return (
		int(config.get("A_ROWS", DEFAULT_A_ROWS)),
		int(config.get("K", DEFAULT_K)),
		int(config.get("B_COLS", DEFAULT_B_COLS)),
	)


A_ROWS, K, B_COLS = load_config()


def create_inputs(a_rows, b_cols):
	rng = np.random.default_rng(42)
	a_f64 = rng.standard_normal((a_rows, K))
	b_t_f64 = rng.standard_normal((b_cols, K))
	a_bf16 = torch.from_numpy(a_f64).to(torch.bfloat16)
	b_t_bf16 = torch.from_numpy(b_t_f64).to(torch.bfloat16)
	a_bf16.view(torch.int16).numpy().view(np.uint16).tofile(INPUT_A_PATH)
	b_t_bf16.view(torch.int16).numpy().view(np.uint16).tofile(INPUT_B_PATH)
	print(f"Created {INPUT_A_PATH}: A [{a_rows}, {K}] bf16  ({a_rows * K * 2} bytes)")
	print(f"Created {INPUT_B_PATH}: B^T [{b_cols}, {K}] bf16  ({b_cols * K * 2} bytes)")


def load_bf16(path, shape):
	raw = np.fromfile(path, dtype=np.uint16).reshape(shape)
	return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)


def geglu(x):
	return x * torch.sigmoid(1.702 * x)


def print_matrix(name, tensor):
	print(f"\n{name}:")
	for row in range(tensor.shape[0]):
		print("".join(f" {tensor[row, col].item():10.4f}" for col in range(tensor.shape[1])))


def verify(a_rows, b_cols):
	if a_rows % 2 != 0:
		raise ValueError(f"A_ROWS must be even for GeGLU, got {a_rows}")
	a_bf16 = load_bf16(INPUT_A_PATH, (a_rows, K))
	b_t_bf16 = load_bf16(INPUT_B_PATH, (b_cols, K))
	if not os.path.exists(OUTPUT_PATH):
		print(f"Missing CUDA output file: {OUTPUT_PATH}")
		return

	cuda_bf16 = load_bf16(OUTPUT_PATH, (a_rows // 2, b_cols))
	a_f32 = a_bf16.float()
	b_t_f32 = b_t_bf16.float()
	pre_geglu = a_f32 @ b_t_f32.transpose(0, 1)
	ref_f32 = geglu(pre_geglu[0::2]) * pre_geglu[1::2]
	ref_bf16 = ref_f32.to(torch.bfloat16)
	ref = ref_bf16.float()
	cuda = cuda_bf16.float()

	diff = (ref - cuda).abs()
	max_abs_diff = diff.max().item()
	mean_abs_diff = diff.mean().item()
	idx = (diff == diff.max()).nonzero(as_tuple=False)[0]
	worst_row = idx[0].item()
	worst_col = idx[1].item()
	worst_ref = ref_bf16[idx[0], idx[1]].float().item()
	worst_cuda = cuda_bf16[idx[0], idx[1]].float().item()
	ref_np = ref_bf16.view(torch.int16).numpy().view(np.uint16)
	cuda_np = cuda_bf16.view(torch.int16).numpy().view(np.uint16)
	exact_match = int((ref_np == cuda_np).sum())
	total = ref.numel()

	print("\n--- O GEMM + GeGLU vs CUDA ---")
	print(f"A=[{a_rows}, {K}], B=[{K}, {b_cols}] stored as B^T=[{b_cols}, {K}], out=[{a_rows // 2}, {b_cols}]")
	print(f"Max abs diff:     {max_abs_diff:.6e}")
	print(f"Mean abs diff:    {mean_abs_diff:.6e}")
	print(f"Exact bf16 match: {exact_match}/{total} ({100.0 * exact_match / total:.2f}%)")
	if max_abs_diff > 0.0:
		print(f"Worst mismatch at [{worst_row}, {worst_col}]: ref={worst_ref:.6e}, cuda={worst_cuda:.6e}")

	print_matrix("First 4x4 (CUDA)", cuda[:4, :4])
	print_matrix("First 4x4 (ref bf16)", ref[:4, :4])


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Output projection GEMM verification")
	parser.add_argument("--create-inputs", action="store_true", help="Generate random O projection inputs in tmp/")
	parser.add_argument("--m", type=int, default=None)
	parser.add_argument("--n", type=int, default=None)
	args = parser.parse_args()

	a_rows = args.m or A_ROWS
	b_cols = args.n or B_COLS

	if args.create_inputs:
		create_inputs(a_rows, b_cols)
	else:
		verify(a_rows, b_cols)
