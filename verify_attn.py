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
V_DIM = 64

import math


def print_matrix(t):
	"""Print a 2D tensor matching C++ printf("%12.6e ", ...) format."""
	for r in range(t.shape[0]):
		print("".join(f"{t[r, c].item():12.6e} " for c in range(t.shape[1])))


def create_inputs(q_len, kv_len):
	rng = np.random.default_rng(42)
	q_f64 = rng.standard_normal((q_len, QK_DIM))
	kv_f64 = rng.standard_normal((kv_len, QK_DIM))
	dO_f64 = rng.standard_normal((q_len, V_DIM))

	# Convert to bf16 via torch (proper rounding)
	q_bf16 = torch.from_numpy(q_f64).to(torch.bfloat16)
	kv_bf16 = torch.from_numpy(kv_f64).to(torch.bfloat16)
	dO_bf16 = torch.from_numpy(dO_f64).to(torch.bfloat16)

	q_bf16.view(torch.int16).numpy().view(np.uint16).tofile("tmp/q.bin")
	kv_bf16.view(torch.int16).numpy().view(np.uint16).tofile("tmp/kv.bin")
	dO_bf16.view(torch.int16).numpy().view(np.uint16).tofile("tmp/dO.bin")
	print(f"Created tmp/q.bin:  [{q_len}, {QK_DIM}] bf16  ({q_len * QK_DIM * 2} bytes)")
	print(f"Created tmp/kv.bin: [{kv_len}, {QK_DIM}] bf16  ({kv_len * QK_DIM * 2} bytes)")
	print(f"Created tmp/dO.bin: [{q_len}, {V_DIM}] bf16  ({q_len * V_DIM * 2} bytes)")


def load_inputs(q_len, kv_len, large=False):
	q_file = "tmp/large_q.bin" if large else "tmp/q.bin"
	kv_file = "tmp/large_kv.bin" if large else "tmp/kv.bin"
	q_raw = np.fromfile(q_file, dtype=np.uint16).reshape(q_len, QK_DIM)
	kv_raw = np.fromfile(kv_file, dtype=np.uint16).reshape(kv_len, QK_DIM)
	Q = torch.from_numpy(q_raw.view(np.int16)).view(torch.bfloat16)
	KV = torch.from_numpy(kv_raw.view(np.int16)).view(torch.bfloat16)
	return Q, KV


def load_bf16(path, shape):
	raw = np.fromfile(path, dtype=np.uint16).reshape(shape)
	return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)


def load_f32(path, shape):
	return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(shape))


def reference_exact(Q, K, V, sink, gate):
	"""Scalable-Softmax causal attention with optional sink token, in f64.
	Q, K, V are f64 tensors. Q may have requires_grad=True.
	Returns output (f64, with grad_fn if Q requires grad).
	"""
	q_len = Q.shape[0]
	kv_len = K.shape[0]
	scores = Q @ K.T     # [q_len, kv_len]
	sink_col = torch.full((q_len, 1), sink, dtype=torch.double)
	scores = torch.cat([sink_col, scores], dim=1)  # [q_len, kv_len+1]
	scores = scores / math.sqrt(QK_DIM)

	# SSMax: multiply ALL scores (including sink) by ln(n) where n = row_index + 2
	n = torch.arange(2, q_len + 2, dtype=torch.float64).unsqueeze(1)  # [q_len, 1]
	scores = scores * torch.log(n)

	# Causal mask: mask out kv_pos > q_pos (sink column is never masked)
	sink_mask = torch.zeros(q_len, 1, dtype=torch.bool)
	mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=1)
	mask = torch.cat([sink_mask, mask], dim=1)
	scores = scores.masked_fill(mask, float('-inf'))

	attn = torch.softmax(scores, dim=-1)
	# Only the real tokens contribute to output (skip sink at column 0)
	out = attn[:, 1:] @ V
	out = out * gate
	return out


def compare(name, ref_bf16, cuda_bf16, q_len, dim):
	ref_f = ref_bf16.float()
	cuda_f = cuda_bf16.float()

	# Count NaN and Inf in CUDA output
	num_nan = torch.isnan(cuda_f).sum().item()
	num_inf = torch.isinf(cuda_f).sum().item()
	if num_nan > 0 or num_inf > 0:
		print(f"\n*** CUDA output: {num_nan} NaN, {num_inf} Inf out of {cuda_f.numel()} values ***")
		nan_mask = torch.isnan(cuda_f) | torch.isinf(cuda_f)
		bad_rows = nan_mask.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
		if bad_rows.numel() > 0:
			print(f"    Bad rows (first 20): {bad_rows[:20].tolist()}")

	diff = (ref_f - cuda_f).abs()
	ref_np = ref_bf16.view(torch.int16).numpy().view(np.uint16).reshape(q_len, dim)
	cuda_np = cuda_bf16.view(torch.int16).numpy().view(np.uint16).reshape(q_len, dim)
	exact_match = (ref_np == cuda_np).sum()
	total = q_len * dim

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


def verify(q_len, kv_len, large=False, sink_val=-0.3, gate_val=0.5):
	Q_bf16, KV_bf16 = load_inputs(q_len, kv_len, large=large)

	if large:
		sink_arg = -math.inf
		gate_arg = 1.0
		prefix = "tmp/large_"
	else:
		sink_arg = sink_val
		gate_arg = gate_val
		prefix = "tmp/"

	# Build f64 tensors with requires_grad for autograd
	Q = Q_bf16.double().requires_grad_(True)
	K = KV_bf16[:, :QK_DIM].double().requires_grad_(True)
	V = KV_bf16[:, :V_DIM].double().requires_grad_(True)

	print(f"Computing exact reference (f64 softmax) q_len={q_len}, kv_len={kv_len}...")
	ref_out = reference_exact(Q, K, V, sink=sink_arg, gate=gate_arg)
	ref_out_bf16 = ref_out.detach().to(torch.bfloat16)

	print(f"\nFirst 4 rows, first 8 cols:")
	print_matrix(ref_out_bf16[:4, :8].float())
	print(f"\nLast 4 rows, last 8 cols:")
	print_matrix(ref_out_bf16[-4:, -8:].float())

	# Compare forward output with CUDA
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
		compare("Forward (exact f64)", ref_out_bf16, cuda_bf16, q_len, V_DIM)
	else:
		print(f"\nNo {cuda_path} found — run the CUDA kernel first to compare.")

	# Backward: compute reference dQ via autograd
	dO_path = f"{prefix}dO.bin"
	if not os.path.exists(dO_path):
		print(f"\nNo {dO_path} found — skipping backward verification.")
		return

	dO_bf16 = load_bf16(dO_path, (q_len, V_DIM))
	ref_out.backward(dO_bf16.double())
	ref_dQ = Q.grad  # f64
	ref_dQ_bf16 = ref_dQ.to(torch.bfloat16)
	ref_dK = K.grad  # f64
	ref_dV = V.grad  # f64
	ref_dK_bf16 = ref_dK.to(torch.bfloat16)
	ref_dV_bf16 = ref_dV.to(torch.bfloat16)

	print(f"\n=== Backward verification ===")

	# Verify D = rowsum(dO ⊙ O)
	D_path = "tmp/D.bin"
	if os.path.exists(D_path):
		O_cuda = load_bf16("tmp/out_cpu.bin", (q_len, V_DIM))
		D_cuda = load_f32(D_path, (q_len,))
		D_ref = (dO_bf16.float() * O_cuda.float()).sum(dim=-1) / gate_arg
		diff = (D_ref - D_cuda).abs()
		exact = (D_ref == D_cuda).sum().item()
		print(f"\n--- D' = rowsum(dO ⊙ O) / gate ---")
		print(f"Max abs diff:  {diff.max().item():.6e}")
		print(f"Exact match:   {exact}/{q_len} ({100*exact/q_len:.2f}%)")

	# Verify dQ
	dQ_path = "tmp/dQ.bin"
	if os.path.exists(dQ_path):
		dQ_cuda = load_bf16(dQ_path, (q_len, QK_DIM))
		compare("dQ (autograd f64)", ref_dQ_bf16, dQ_cuda, q_len, QK_DIM)

	# Verify dK
	dK_path = "tmp/dK.bin"
	if os.path.exists(dK_path):
		dK_cuda = load_bf16(dK_path, (kv_len, QK_DIM))
		compare("dK (autograd f64)", ref_dK_bf16, dK_cuda, kv_len, QK_DIM)

	# Verify dV
	dV_path = "tmp/dV.bin"
	if os.path.exists(dV_path):
		dV_cuda = load_bf16(dV_path, (kv_len, V_DIM))
		compare("dV (autograd f64)", ref_dV_bf16, dV_cuda, kv_len, V_DIM)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Flash attention verification")
	parser.add_argument("--create-inputs", action="store_true",
						help="Generate random tmp/q.bin and tmp/kv.bin")
	parser.add_argument("--large", action="store_true",
						help="Use large generated data (tmp/large_q.bin, tmp/large_kv.bin)")
	parser.add_argument("--q-len", type=int, default=None)
	parser.add_argument("--kv-len", type=int, default=None)
	args = parser.parse_args()

	if args.large:
		q_len = args.q_len or 32768
		kv_len = args.kv_len or 32768
	else:
		q_len = args.q_len or 1024
		kv_len = args.kv_len or 1024

	if args.create_inputs:
		create_inputs(q_len, kv_len)
	else:
		verify(q_len, kv_len, large=args.large)
