#!/usr/bin/env python3
"""
Verification script for the flash attention kernel in ldmatrix_example.cu.

Usage:
python verify_attn.py --create-inputs          # generate q.bin, kv.bin
python verify_attn.py                           # compute reference & compare
"""

import numpy as np
import torch
import argparse
import os

QK_DIM = 128
V_DIM = 128

import math


def print_matrix(t):
	"""Print a 2D tensor matching C++ printf("%12.6e ", ...) format."""
	for r in range(t.shape[0]):
		print("".join(f"{t[r, c].item():12.6e} " for c in range(t.shape[1])))


def create_inputs(q_len, kv_len):
	rng = np.random.default_rng(42)
	q_f32 = (rng.standard_normal((q_len, QK_DIM)) * 0.1).astype(np.float32)
	kv_f32 = (rng.standard_normal((kv_len, QK_DIM)) * 0.1).astype(np.float32)

	# Convert to bf16 via torch (proper rounding)
	q_bf16 = torch.from_numpy(q_f32).to(torch.bfloat16)
	kv_bf16 = torch.from_numpy(kv_f32).to(torch.bfloat16)

	q_bf16.view(torch.int16).numpy().view(np.uint16).tofile("tmp/q.bin")
	kv_bf16.view(torch.int16).numpy().view(np.uint16).tofile("tmp/kv.bin")
	print(f"Created tmp/q.bin:  [{q_len}, {QK_DIM}] bf16  ({q_len * QK_DIM * 2} bytes)")
	print(f"Created tmp/kv.bin: [{kv_len}, {QK_DIM}] bf16  ({kv_len * QK_DIM * 2} bytes)")


def load_inputs(q_len, kv_len):
	q_raw = np.fromfile("tmp/q.bin", dtype=np.uint16).reshape(q_len, QK_DIM)
	kv_raw = np.fromfile("tmp/kv.bin", dtype=np.uint16).reshape(kv_len, QK_DIM)
	Q = torch.from_numpy(q_raw.view(np.int16)).view(torch.bfloat16)
	KV = torch.from_numpy(kv_raw.view(np.int16)).view(torch.bfloat16)
	return Q, KV


def reference_exact(Q, KV):
	"""Scalable-Softmax causal attention in f64.
	SSMax_n(x)_i = n^(x_i) / sum_j n^(x_j), which is equivalent to
	standard softmax with scores pre-scaled by ln(n).
	For row i (0-indexed), n = i+1 (causal: attends to positions 0..i).
	"""
	q_len = Q.shape[0]
	kv_len = KV.shape[0]
	K = KV[:, :QK_DIM].double()   # [kv_len, QK_DIM]
	V = KV[:, :V_DIM].double()    # [kv_len, V_DIM]
	scores = Q.double() @ K.T     # [q_len, kv_len]
	scores /= math.sqrt(QK_DIM)

	# SSMax: multiply each row's scores by ln(n) where n = row_index + 1
	n = torch.arange(1, q_len + 1, dtype=torch.float64).unsqueeze(1)  # [q_len, 1]
	scores *= torch.log(n)

	# Causal mask: mask out kv_pos > q_pos
	causal_mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=1)
	scores.masked_fill_(causal_mask, float('-inf'))
	attn = torch.softmax(scores, dim=-1)
	out = attn @ V
	return out.to(torch.bfloat16)


def compare(name, ref_bf16, cuda_bf16, q_len):
	ref_f = ref_bf16.float()
	cuda_f = cuda_bf16.float()
	diff = (ref_f - cuda_f).abs()
	ref_np = ref_bf16.view(torch.int16).numpy().view(np.uint16).reshape(q_len, V_DIM)
	cuda_np = cuda_bf16.view(torch.int16).numpy().view(np.uint16).reshape(q_len, V_DIM)
	exact_match = (ref_np == cuda_np).sum()
	total = q_len * V_DIM

	# Sign flips: both nonzero and opposite signs
	sign_flip_mask = (ref_f * cuda_f) < 0
	sign_flips = sign_flip_mask.sum().item()
	if sign_flips > 0:
		flip_max_mag = torch.maximum(ref_f.abs(), cuda_f.abs())[sign_flip_mask].max().item()
	else:
		flip_max_mag = 0.0

	# Max percentage difference: max(a,b)/min(a,b), skipping near-zero elements
	a, b = ref_f.abs(), cuda_f.abs()
	pct_strs = []
	for MIN_MAG in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
		mask = (a > MIN_MAG) & (b > MIN_MAG)
		if mask.any():
			hi = torch.maximum(a[mask], b[mask])
			lo = torch.minimum(a[mask], b[mask])
			max_pct = ((hi / lo).max().item() - 1.0) * 100.0
		else:
			max_pct = 0.0
		pct_strs.append(f"{max_pct:.4f}%")

	print(f"\n--- {name} vs CUDA ---")
	print(f"Max abs diff:     {diff.max().item():.6e}")
	print(f"Mean abs diff:    {diff.mean().item():.6e}")
	print(f"Max pct diff:     {', '.join(pct_strs)}")
	print(f"  (MIN_MAG:       1e-6,     1e-5,     1e-4,     1e-3,     1e-2,     1e-1)")
	print(f"Exact bf16 match: {exact_match}/{total} ({100.0 * exact_match / total:.2f}%)")
	if sign_flips > 0:
		print(f"WARNING: {sign_flips} values flipped sign! max magnitude: {flip_max_mag:.6e}")

	if diff.max().item() > 0:
		idx = (diff == diff.max()).nonzero(as_tuple=False)[0]
		r, c = idx[0].item(), idx[1].item()
		print(f"Worst mismatch at [{r}, {c}]: "
			f"ref={ref_bf16[r, c].float().item():.6e}, "
			f"cuda={cuda_bf16[r, c].float().item():.6e}")


def verify(q_len, kv_len):
	Q, KV = load_inputs(q_len, kv_len)

	print("Computing exact reference (f64 softmax)...")
	ref_exact = reference_exact(Q, KV)

	print(f"\nFirst 4 rows, first 8 cols:")
	print_matrix(ref_exact[:4, :8].float())
	print(f"\nLast 4 rows, last 8 cols:")
	print_matrix(ref_exact[-4:, -8:].float())

	# Compare with CUDA output if available
	cuda_path = "tmp/out_cpu.bin"
	if os.path.exists(cuda_path):
		cuda_raw = np.fromfile(cuda_path, dtype=np.uint16)
		expected = q_len * V_DIM
		if cuda_raw.size != expected:
			print(f"\nWarning: tmp/out_cpu.bin has {cuda_raw.size} uint16 elements, "
				f"expected {expected} ({q_len}x{V_DIM}). "
				f"Make sure Q_LEN/KV_LEN match in the CUDA code.")
			return

		cuda_bf16 = torch.from_numpy(cuda_raw.view(np.int16)).view(torch.bfloat16).reshape(q_len, V_DIM)

		compare("Exact f64", ref_exact, cuda_bf16, q_len)
	else:
		print(f"\nNo {cuda_path} found — run the CUDA kernel first to compare.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Flash attention verification")
	parser.add_argument("--create-inputs", action="store_true",
						help="Generate random tmp/q.bin and tmp/kv.bin")
	parser.add_argument("--q-len", type=int, default=1024)
	parser.add_argument("--kv-len", type=int, default=1024)
	args = parser.parse_args()

	if args.create_inputs:
		create_inputs(args.q_len, args.kv_len)
	else:
		verify(args.q_len, args.kv_len)
