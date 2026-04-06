#!/usr/bin/env python3
"""
Verification script for gemm.cu.

Usage:
python verify_gemm.py --create-inputs
python verify_gemm.py
"""

import argparse
import json
import os

import numpy as np
import torch

DEFAULT_A_M = 4096
DEFAULT_A_N = 512
DEFAULT_B_M = 1024
DEFAULT_B_N = 32768
DEFAULT_HEAD_DIM = 32
DEFAULT_ROPE_DIM = DEFAULT_HEAD_DIM
DEFAULT_ROPE_BASE = 10000.0
DEFAULT_INPUT_STEP = 16
GEMM_CONFIG_PATH = "tmp/gemm.config.json"


def load_gemm_config():
	if not os.path.exists(GEMM_CONFIG_PATH):
		return DEFAULT_A_M, DEFAULT_A_N, DEFAULT_B_M, DEFAULT_B_N, DEFAULT_HEAD_DIM, DEFAULT_ROPE_DIM, DEFAULT_ROPE_BASE, DEFAULT_INPUT_STEP
	try:
		with open(GEMM_CONFIG_PATH, "r", encoding="ascii") as config_file:
			config = json.load(config_file)
	except (OSError, ValueError, TypeError, json.JSONDecodeError):
		return DEFAULT_A_M, DEFAULT_A_N, DEFAULT_B_M, DEFAULT_B_N, DEFAULT_HEAD_DIM, DEFAULT_ROPE_DIM, DEFAULT_ROPE_BASE, DEFAULT_INPUT_STEP
	return (
		int(config.get("A_M", DEFAULT_A_M)),
		int(config.get("A_N", DEFAULT_A_N)),
		int(config.get("B_M", DEFAULT_B_M)),
		int(config.get("B_N", DEFAULT_B_N)),
		int(config.get("HEAD_DIM", DEFAULT_HEAD_DIM)),
		int(config.get("ROPE_DIM", DEFAULT_ROPE_DIM)),
		float(config.get("ROPE_BASE", DEFAULT_ROPE_BASE)),
		int(config.get("INPUT_STEP", DEFAULT_INPUT_STEP)),
	)


A_M, A_N, B_M, B_N, HEAD_DIM, ROPE_DIM, ROPE_BASE, INPUT_STEP = load_gemm_config()


def create_inputs(m, n):
	rng = np.random.default_rng(42)
	a_f64 = rng.standard_normal((m, A_N))
	b_t_f64 = rng.standard_normal((n, B_M))
	a_bf16 = torch.from_numpy(a_f64).to(torch.bfloat16)
	b_t_bf16 = torch.from_numpy(b_t_f64).to(torch.bfloat16)
	a_bf16.view(torch.int16).numpy().view(np.uint16).tofile("tmp/a.bin")
	b_t_bf16.view(torch.int16).numpy().view(np.uint16).tofile("tmp/b.bin")
	print(f"Created tmp/a.bin: A [{m}, {A_N}] bf16  ({m * A_N * 2} bytes)")
	print(f"Created tmp/b.bin: B^T [{n}, {B_M}] bf16  ({n * B_M * 2} bytes)")



def load_bf16(path, shape):
	raw = np.fromfile(path, dtype=np.uint16).reshape(shape)
	return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)



def load_inputs(m, n):
	a = load_bf16("tmp/a.bin", (m, A_N))
	b = load_bf16("tmp/b.bin", (n, B_M))
	return a, b


def make_rope_tables(m):
	if ROPE_DIM == 0:
		shape = (m, 0)
		return torch.empty(shape, dtype=torch.float32), torch.empty(shape, dtype=torch.float32)
	inv_freq = torch.pow(
		torch.tensor(ROPE_BASE, dtype=torch.float32),
		-2.0 * torch.arange(ROPE_DIM // 2, dtype=torch.float32) / float(ROPE_DIM),
	)
	positions = torch.arange(m, dtype=torch.float32).unsqueeze(1)
	theta = positions * inv_freq.unsqueeze(0)
	return torch.cos(theta), torch.sin(theta)



def shift_b_rows(b_f32_chunk, row_start):
	row_offsets = torch.arange(row_start, row_start + b_f32_chunk.shape[0], dtype=torch.long)
	row_offsets = (row_offsets * INPUT_STEP) % B_M
	k_idx = torch.arange(A_N, dtype=torch.long).unsqueeze(0)
	gather_idx = (k_idx + row_offsets.unsqueeze(1)) % B_M
	return b_f32_chunk.gather(1, gather_idx)


def apply_rope(chunk, cos_table, sin_table):
	if ROPE_DIM == 0:
		return chunk
	result = chunk.clone()
	even = chunk[:, :ROPE_DIM:2]
	odd = chunk[:, 1:ROPE_DIM:2]
	result[:, :ROPE_DIM:2] = even * cos_table - odd * sin_table
	result[:, 1:ROPE_DIM:2] = even * sin_table + odd * cos_table
	return result


def reference_chunk(a_f32, b_f32_chunk, row_start, cos_table, sin_table):
	b_shifted = shift_b_rows(b_f32_chunk, row_start)
	chunk = a_f32 @ b_shifted.T
	norm = torch.linalg.vector_norm(chunk, ord=2, dim=1, keepdim=True)
	chunk = chunk / norm.clamp_min(1e-30)
	return apply_rope(chunk, cos_table, sin_table)



def print_matrix(name, t):
	print(f"\n{name}:")
	for r in range(t.shape[0]):
		print("".join(f" {t[r, c].item():10.4f}" for c in range(t.shape[1])))




def verify(m, n):
	if A_N <= 0 or B_M <= 0 or A_N > B_M:
		raise ValueError(f"Expected 0 < A_N <= B_M, got A_N={A_N}, B_M={B_M}")
	if A_N % 16 != 0 or B_M % 16 != 0:
		raise ValueError(f"A_N and B_M must be multiples of 16, got A_N={A_N}, B_M={B_M}")
	if HEAD_DIM <= 0 or HEAD_DIM % 16 != 0:
		raise ValueError(f"HEAD_DIM must be a positive multiple of 16, got {HEAD_DIM}")
	if n % HEAD_DIM != 0:
		raise ValueError(f"N must be divisible by HEAD_DIM: N={n}, HEAD_DIM={HEAD_DIM}")
	if ROPE_DIM < 0 or ROPE_DIM > HEAD_DIM or ROPE_DIM % 16 != 0:
		raise ValueError(f"ROPE_DIM must satisfy 0 <= ROPE_DIM <= HEAD_DIM and be a multiple of 16, got {ROPE_DIM}")

	a_bf16, b_bf16 = load_inputs(m, n)
	cuda_path = "tmp/out_cpu.bin"
	if not os.path.exists(cuda_path):
		print(f"Missing CUDA output file: {cuda_path}")
		return

	cuda_bf16 = load_bf16(cuda_path, (m, n))
	a_f32 = a_bf16.float()
	b_f32 = b_bf16.float()
	cuda_f32 = cuda_bf16.float()
	cos_table, sin_table = make_rope_tables(m)

	num_nan = torch.isnan(cuda_f32).sum().item()
	num_inf = torch.isinf(cuda_f32).sum().item()
	if num_nan > 0 or num_inf > 0:
		print(f"\n*** CUDA output: {num_nan} NaN, {num_inf} Inf out of {cuda_f32.numel()} values ***")

	total = m * n
	exact_match = 0
	sum_abs_diff = 0.0
	max_abs_diff = 0.0
	worst_row = 0
	worst_col = 0
	worst_ref = 0.0
	worst_cuda = 0.0
	sign_flips = 0
	flip_max_mag = 0.0
	max_pct_ratios = [1.0] * 6
	min_mags = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
	ref_preview = None

	for n_start in range(0, n, HEAD_DIM):
		n_end = min(n_start + HEAD_DIM, n)
		ref_chunk_f32 = reference_chunk(a_f32, b_f32[n_start:n_end], n_start, cos_table, sin_table)
		ref_chunk_bf16 = ref_chunk_f32.to(torch.bfloat16)
		cuda_chunk_bf16 = cuda_bf16[:, n_start:n_end]
		ref_chunk_f = ref_chunk_bf16.float()
		cuda_chunk_f = cuda_chunk_bf16.float()
		if ref_preview is None:
			ref_preview = ref_chunk_f[:4, :4].clone()

		diff = (ref_chunk_f - cuda_chunk_f).abs()
		sum_abs_diff += diff.sum().item()
		chunk_max = diff.max().item()
		if chunk_max > max_abs_diff:
			idx = (diff == diff.max()).nonzero(as_tuple=False)[0]
			worst_row = idx[0].item()
			worst_col = n_start + idx[1].item()
			worst_ref = ref_chunk_bf16[idx[0], idx[1]].float().item()
			worst_cuda = cuda_chunk_bf16[idx[0], idx[1]].float().item()
			max_abs_diff = chunk_max

		ref_np = ref_chunk_bf16.view(torch.int16).numpy().view(np.uint16)
		cuda_np = cuda_chunk_bf16.view(torch.int16).numpy().view(np.uint16)
		exact_match += int((ref_np == cuda_np).sum())

		sign_flip_mask = (ref_chunk_f * cuda_chunk_f) < 0
		sign_flips += int(sign_flip_mask.sum().item())
		if sign_flip_mask.any():
			flip_max_mag = max(
				flip_max_mag,
				torch.maximum(ref_chunk_f.abs(), cuda_chunk_f.abs())[sign_flip_mask].max().item(),
			)

		a_abs = ref_chunk_f.abs()
		b_abs = cuda_chunk_f.abs()
		for i, min_mag in enumerate(min_mags):
			mask = (a_abs > min_mag) & (b_abs > min_mag)
			if mask.any():
				hi = torch.maximum(a_abs[mask], b_abs[mask])
				lo = torch.minimum(a_abs[mask], b_abs[mask])
				max_pct_ratios[i] = max(max_pct_ratios[i], (hi / lo).max().item())

	mean_abs_diff = sum_abs_diff / total
	pct_strs = [f"{(ratio - 1.0) * 100.0:.4f}%" for ratio in max_pct_ratios]

	print(f"\n--- GEMM + L2 norm + RoPE vs CUDA ---")
	print(f"A=[{m}, {A_N}], B=[{B_M}, {n}] stored as B^T=[{n}, {B_M}], INPUT_STEP={INPUT_STEP}")
	print(f"Max abs diff:     {max_abs_diff:.6e}")
	print(f"Mean abs diff:    {mean_abs_diff:.6e}")
	print(f"Max pct diff:     {', '.join(pct_strs)}")
	print(f"  (MIN_MAG:       1e-6,     1e-5,     1e-4,     1e-3,     1e-2,     1e-1)")
	print(f"Exact bf16 match: {exact_match}/{total} ({100.0 * exact_match / total:.2f}%)")
	if sign_flips > 0:
		print(f"WARNING: {sign_flips} values flipped sign! max magnitude: {flip_max_mag:.6e}")
	if max_abs_diff > 0.0:
		print(f"Worst mismatch at [{worst_row}, {worst_col}]: ref={worst_ref:.6e}, cuda={worst_cuda:.6e}")

	print_matrix("First 4x4 (CUDA)", cuda_f32[:4, :4])
	if ref_preview is not None:
		print_matrix("First 4x4 (ref bf16)", ref_preview)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="GEMM verification")
	parser.add_argument("--create-inputs", action="store_true", help="Generate random GEMM inputs in tmp/")
	parser.add_argument("--m", type=int, default=None)
	parser.add_argument("--n", type=int, default=None)
	args = parser.parse_args()

	m = args.m or A_M
	n = args.n or B_N

	if args.create_inputs:
		create_inputs(m, n)
	else:
		verify(m, n)
