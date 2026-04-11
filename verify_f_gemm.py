#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import torch

DEFAULT_A_ROWS = 1024
DEFAULT_K = 2048
DEFAULT_B_COLS = 32768
CONFIG_PATH = "tmp/f_gemm.config.json"
INPUT_A_PATH = "tmp/f_a.bin"
INPUT_B_PATH = "tmp/f_b.bin"
OUTPUT_PATH = "tmp/f_out_cpu.bin"
L2_PATH = "tmp/f_L2.bin"


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


def load_f32(path, shape):
	raw = np.fromfile(path, dtype=np.float32).reshape(shape)
	return torch.from_numpy(raw.copy())


def print_matrix(name, tensor):
	print(f"\n{name}:")
	for row in range(tensor.shape[0]):
		print("".join(f" {tensor[row, col].item():10.4f}" for col in range(tensor.shape[1])))


def verify(a_rows, b_cols):
	a_bf16 = load_bf16(INPUT_A_PATH, (a_rows, K))
	b_t_bf16 = load_bf16(INPUT_B_PATH, (b_cols, K))
	if not os.path.exists(OUTPUT_PATH):
		print(f"Missing CUDA output file: {OUTPUT_PATH}")
		return

	cuda_bf16 = load_bf16(OUTPUT_PATH, (b_cols, a_rows))
	a_f32 = a_bf16.float()
	b_t_f32 = b_t_bf16.float()
	print(f"a = {a_f32.shape}")
	print(f"b^T = {b_t_f32.shape}")
	ref_bf16 = (a_f32 @ b_t_f32.transpose(0, 1)).to(torch.bfloat16).transpose(0, 1)
	print(f"ref = {ref_bf16.shape}")
	ref = ref_bf16.float()
	cuda = cuda_bf16.float()
	print(f"cuda = {cuda.shape}")

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

	print("\n--- F GEMM vs CUDA ---")
	print(f"A=[{a_rows}, {K}], B=[{K}, {b_cols}] stored as B^T=[{b_cols}, {K}], out=[{b_cols}, {a_rows}]")
	print(f"Max abs diff:     {max_abs_diff:.6e}")
	print(f"Mean abs diff:    {mean_abs_diff:.6e}")
	print(f"Exact bf16 match: {exact_match}/{total} ({100.0 * exact_match / total:.2f}%)")
	if max_abs_diff > 0.0:
		print(f"Worst mismatch at [{worst_row}, {worst_col}]: ref={worst_ref:.6e}, cuda={worst_cuda:.6e}")

	print_matrix("First 4x4 (CUDA)", cuda[:4, :4])
	print_matrix("First 4x4 (ref bf16)", ref[:4, :4])

	if os.path.exists(L2_PATH):
		cuda_l2 = load_f32(L2_PATH, (b_cols,))
		ref_l2 = (a_f32 @ b_t_f32.transpose(0, 1)).square().sum(dim=0)
		l2_diff = (ref_l2 - cuda_l2).abs()
		l2_max_abs_diff = l2_diff.max().item()
		l2_mean_abs_diff = l2_diff.mean().item()
		l2_idx = int(torch.argmax(l2_diff).item())
		l2_exact_match = int((ref_l2 == cuda_l2).sum().item())
		l2_exact_bf16_match = int((ref_l2.to(torch.bfloat16) == cuda_l2.to(torch.bfloat16)).sum().item())
		l2_rel_diff = l2_diff / ref_l2.abs().clamp_min(1.0)
		l2_max_rel_diff = l2_rel_diff.max().item()
		l2_close = torch.isclose(ref_l2, cuda_l2, rtol=1e-5, atol=512.0)
		l2_close_match = int(l2_close.sum().item())
		print("\n--- F GEMM L2 vs CUDA ---")
		print(f"Max abs diff:     {l2_max_abs_diff:.6e}")
		print(f"Mean abs diff:    {l2_mean_abs_diff:.6e}")
		print(f"Exact f32 match:  {l2_exact_match}/{b_cols} ({100.0 * l2_exact_match / b_cols:.2f}%)")
		print(f"Exact bf16 match: {l2_exact_bf16_match}/{b_cols} ({100.0 * l2_exact_bf16_match / b_cols:.2f}%)")
		print(f"Max rel diff:     {l2_max_rel_diff:.6e}")
		print(f"Rows within tol:  {l2_close_match}/{b_cols} ({100.0 * l2_close_match / b_cols:.2f}%)")
		if l2_max_abs_diff > 0.0:
			print(f"Worst mismatch at row {l2_idx}: ref={ref_l2[l2_idx].item():.6e}, cuda={cuda_l2[l2_idx].item():.6e}")
		print("\nFirst 4 L2 values (CUDA):")
		print("".join(f" {cuda_l2[i].item():12.6f}" for i in range(min(4, b_cols))))
		print("First 4 L2 values (ref):")
		print("".join(f" {ref_l2[i].item():12.6f}" for i in range(min(4, b_cols))))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Full-width GEMM verification")
	parser.add_argument("--create-inputs", action="store_true", help="Generate random F GEMM inputs in tmp/")
	parser.add_argument("--m", type=int, default=None)
	parser.add_argument("--n", type=int, default=None)
	args = parser.parse_args()

	a_rows = args.m or A_ROWS
	b_cols = args.n or B_COLS

	if args.create_inputs:
		create_inputs(a_rows, b_cols)
	else:
		verify(a_rows, b_cols)
