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
import math

HEAD_CNT = 16
QK_DIM = 64
V_DIM = 64
# Must match WINDOW_SIZE in attn.cu.
WINDOW_SIZE = 0
SCORE_SCALE = 1.0 / math.sqrt(QK_DIM)



def ssmax_n(q_len, 	window_size=0):
	"""Compute SSMax scale factor n for each query position.

	n[i] = min(window_size, i + 1) + (e + 1)

	where:
	  i + 1       = number of real tokens visible (causal: tokens 0..i)
	  window_size = caps the visible count when sliding window is enabled
	  e           = ensures log2(n) >= ~1.89, so SSMax scale >= 1
	  1           = accounts for the sink token

	When window_size == 0 (disabled), min is a no-op: n[i] = i + 2 + e.
	"""
	E_PLUS_1 = math.e + 1.0
	pos_plus_1 = torch.arange(1, q_len + 1, dtype=torch.float32)
	if window_size > 0:
		n = torch.clamp(pos_plus_1, max=float(window_size)) + E_PLUS_1
	else:
		n = pos_plus_1 + E_PLUS_1
	return n


def print_matrix(t):
	"""Print a 2D tensor matching C++ printf("%12.6e ", ...) format."""
	for r in range(t.shape[0]):
		print("".join(f"{t[r, c].item():12.6e} " for c in range(t.shape[1])))


def flatten_heads(t):
	return t.reshape(t.shape[0], -1)


def create_inputs(q_len, kv_len):
	rng = np.random.default_rng(42)
	q_f64 = rng.standard_normal((q_len, HEAD_CNT, QK_DIM))
	kv_f64 = rng.standard_normal((kv_len, HEAD_CNT, QK_DIM))
	dO_f64 = rng.standard_normal((q_len, HEAD_CNT, V_DIM))

	# Convert to bf16 via torch (proper rounding)
	q_bf16 = torch.from_numpy(q_f64).to(torch.bfloat16)
	kv_bf16 = torch.from_numpy(kv_f64).to(torch.bfloat16)
	dO_bf16 = torch.from_numpy(dO_f64).to(torch.bfloat16)

	flatten_heads(q_bf16).view(torch.int16).numpy().view(np.uint16).tofile("tmp/q.bin")
	flatten_heads(kv_bf16).view(torch.int16).numpy().view(np.uint16).tofile("tmp/kv.bin")
	flatten_heads(dO_bf16).view(torch.int16).numpy().view(np.uint16).tofile("tmp/dO.bin")
	print(f"Created tmp/q.bin:  [{q_len}, {HEAD_CNT * QK_DIM}] bf16  ({q_len * HEAD_CNT * QK_DIM * 2} bytes)")
	print(f"Created tmp/kv.bin: [{kv_len}, {HEAD_CNT * QK_DIM}] bf16  ({kv_len * HEAD_CNT * QK_DIM * 2} bytes)")
	print(f"Created tmp/dO.bin: [{q_len}, {HEAD_CNT * V_DIM}] bf16  ({q_len * HEAD_CNT * V_DIM * 2} bytes)")


def load_inputs(q_len, kv_len, large=False):
	q_file = "tmp/large_q.bin" if large else "tmp/q.bin"
	kv_file = "tmp/large_kv.bin" if large else "tmp/kv.bin"
	q_raw = np.fromfile(q_file, dtype=np.uint16).reshape(q_len, HEAD_CNT * QK_DIM)
	kv_raw = np.fromfile(kv_file, dtype=np.uint16).reshape(kv_len, HEAD_CNT * QK_DIM)
	Q = torch.from_numpy(q_raw.view(np.int16)).view(torch.bfloat16).reshape(q_len, HEAD_CNT, QK_DIM)
	KV = torch.from_numpy(kv_raw.view(np.int16)).view(torch.bfloat16).reshape(kv_len, HEAD_CNT, QK_DIM)
	return Q, KV


def load_bf16(path, shape):
	raw = np.fromfile(path, dtype=np.uint16).reshape(shape)
	return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)


def load_f32(path, shape):
	return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(shape))


def compute_attn_real(Q, K, sink, window_size=0):
	"""Return real-token attention probabilities in f32.

	Masking rules for query at position i attending to key at position j:
	  - Causal:  mask when j > i           (no future tokens)
	  - Window:  mask when i - j >= W      (at most W tokens back, W = window_size)
	The sink token is never masked. When window_size == 0, window masking is disabled.
	"""
	q_len = Q.shape[0]
	kv_len = K.shape[0]
	n = ssmax_n(q_len, window_size).unsqueeze(1)  # [q_len, 1]
	ssmax_scale = torch.log2(n)
	real_scores = (Q @ K.T) * (SCORE_SCALE * ssmax_scale)
	sink_col = torch.full((q_len, 1), sink, dtype=torch.float32) * SCORE_SCALE * ssmax_scale
	scores = torch.cat([sink_col, real_scores], dim=1)  # [q_len, kv_len+1]

	# Causal mask: mask when j > i (sink column is never masked)
	causal_mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=1)

	# Window mask: mask when i - j >= window_size (keys too far in the past)
	if window_size > 0:
		# tril(., diagonal=-W) gives True where row - col >= W
		window_mask = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=-window_size)
		mask = causal_mask | window_mask
	else:
		mask = causal_mask

	sink_mask = torch.zeros(q_len, 1, dtype=torch.bool)
	mask = torch.cat([sink_mask, mask], dim=1)
	scores = scores.masked_fill(mask, float('-inf'))

	# Manual base-2 softmax to get closer to the kernel's exp2/log2 path.
	row_max = torch.amax(scores, dim=-1, keepdim=True)
	exp_scores = torch.exp2(scores - row_max)
	attn = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
	return attn[:, 1:]


def reference_exact(Q, K, V, sink, gate, window_size=0):
	"""Reference used for backward pass.
	Uses dense accumulation once probabilities have been bf16-rounded.
	"""
	attn_real = compute_attn_real(Q, K, sink, window_size=window_size)
	# Only the real tokens contribute to output (skip sink at column 0)
	out = attn_real @ V
	out = out * gate
	return out


def reference_matching(Q, K, V, sink, gate, kv_tile=16, window_size=0):
	"""Forward-only reference that tries to match kernel accumulation order better.
	Accumulates the second GEMM in KV tiles, similar to the kernel's tiled loop.
	"""
	attn_real = compute_attn_real(Q, K, sink, window_size=window_size)
	out = torch.zeros((Q.shape[0], V.shape[1]), dtype=torch.float32)
	for start in range(0, K.shape[0], kv_tile):
		end = min(start + kv_tile, K.shape[0])
		out = out + attn_real[:, start:end] @ V[start:end, :]
	return out * gate


def reference_online_softmax(Q, K, V, sink, gate, kv_tile=16, window_size=0):
	"""Reference that simulates the kernel's online softmax and MMA accumulation.

	Key matching details vs the CUDA kernel:
	- Score dot product: accumulated in 8 tiles of k=16 (matching MMA m16n8k16 structure)
	- Online softmax: per-row running max, same update/rescale logic
	- P cast to bf16 before P×V (matching cast before MMA)
	- P×V: inner dim = 16 (matches MMA's k=16 for this GEMM)
	- combine_and_store: sink folding, normalization, gate
	"""
	q_len = Q.shape[0]
	kv_len = K.shape[0]
	v_dim = V.shape[1]

	# Compute scores with tiled k-accumulation matching the kernel's MMA structure.
	# The kernel does 8 mma_a_bt calls (QK_TILES=QK_DIM/16=8), each processing
	# 16 columns of Q and K. We replicate this by accumulating partial matmuls.
	scores = torch.zeros(q_len, kv_len, dtype=torch.float32)
	for d in range(0, QK_DIM, 16):
		scores = scores + Q[:, d:d+16] @ K[:, d:d+16].T

	# Per-row SSMax scale is applied directly to scaled scores.
	n = ssmax_n(q_len, window_size)
	ssmax_scale = torch.log2(n)
	scores = scores  * SCORE_SCALE * ssmax_scale.unsqueeze(1)

	# Causal mask
	mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=1)
	# Window mask: mask when i - j >= window_size
	if window_size > 0:
		window_mask = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=-window_size)
		mask = mask | window_mask
	scores = scores.masked_fill(mask, float('-inf'))

	# V as bf16 (matching kernel's bf16 MMA input)
	V_bf16 = V.to(torch.bfloat16).float()

	# Online softmax state
	FLT_LOWEST = torch.finfo(torch.float32).min  # matches C++ numeric_limits<f32>::lowest()
	row_max = torch.full((q_len,), FLT_LOWEST, dtype=torch.float32)
	row_sum = torch.zeros(q_len, dtype=torch.float32)
	O = torch.zeros(q_len, v_dim, dtype=torch.float32)

	for kv_start in range(0, kv_len, kv_tile):
		kv_end = min(kv_start + kv_tile, kv_len)
		S_tile = scores[:, kv_start:kv_end]

		# Per-row max of this tile
		tile_max = S_tile.max(dim=1).values
		new_max = torch.maximum(row_max, tile_max)

		# Rescale previous O and sum
		rescale = torch.exp2(row_max - new_max)
		O = O * rescale.unsqueeze(1)
		row_sum = row_sum * rescale
		row_max = new_max

		# Unnormalized P = exp2(score - running_max)
		P = torch.exp2(S_tile - row_max.unsqueeze(1))
		row_sum = row_sum + P.sum(dim=1)

		# Cast P to bf16 (kernel casts before MMA)
		P_bf16 = P.to(torch.bfloat16).float()

		# Accumulate O += P @ V
		O = O + P_bf16 @ V_bf16[kv_start:kv_end, :]

	# combine_and_store: include sink and normalize
	sink_scaled = torch.full((q_len,), sink, dtype=torch.float32) * SCORE_SCALE * ssmax_scale
	global_max = torch.maximum(row_max, sink_scaled)
	global_sum = (
		row_sum * torch.exp2(row_max - global_max)
		+ torch.exp2(sink_scaled - global_max)
	)

	# Final rescale: exp2(row_max - global_max) * (gate / global_sum)
	final_rescale = torch.exp2(row_max - global_max) * (gate / global_sum)
	O = O * final_rescale.unsqueeze(1)

	return O


def reference_matching_from_L(Q, K, V, L, gate, kv_tile=16, v_tile=16, window_size=0):
	"""Forward-only reference reconstructed from kernel-produced L.
	Uses exp2(S - L), rounds both P and V through bf16, and accumulates the
	second GEMM in small tiles to get closer to the kernel's MMA schedule.
	"""
	q_len = Q.shape[0]
	kv_len = K.shape[0]
	scores = Q @ K.T
	n = ssmax_n(q_len, window_size).unsqueeze(1)
	scores = scores * SCORE_SCALE * torch.log2(n)
	mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=1)
	if window_size > 0:
		window_mask = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=-window_size)
		mask = mask | window_mask
	scores = scores.masked_fill(mask, float('-inf'))
	attn_real = torch.exp2(scores - L.unsqueeze(1)).to(torch.bfloat16).to(torch.float32)
	V_match = V.to(torch.bfloat16).to(torch.float32)
	out = torch.zeros((q_len, V.shape[1]), dtype=torch.float32)
	for v_start in range(0, V.shape[1], v_tile):
		v_end = min(v_start + v_tile, V.shape[1])
		out_tile = torch.zeros((q_len, v_end - v_start), dtype=torch.float32)
		for kv_start in range(0, kv_len, kv_tile):
			kv_end = min(kv_start + kv_tile, kv_len)
			p_tile = attn_real[:, kv_start:kv_end]
			v_tile_data = V_match[kv_start:kv_end, v_start:v_end]
			out_tile = out_tile + p_tile @ v_tile_data
		out[:, v_start:v_end] = out_tile
	return out * gate


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


def verify(q_len, kv_len, large=False, sink_val=-0.3, gate_val=0.5, window_size=WINDOW_SIZE):
	Q_bf16, KV_bf16 = load_inputs(q_len, kv_len, large=large)

	if large:
		sink_arg = -math.inf
		gate_arg = 1.0
		prefix = "tmp/large_"
	else:
		sink_arg = sink_val
		gate_arg = gate_val
		prefix = "tmp/"

	dO_path = f"{prefix}dO.bin"
	if not os.path.exists(dO_path):
		print(f"\nNo {dO_path} found — skipping backward verification.")
		return

	dO_bf16 = load_bf16(dO_path, (q_len, HEAD_CNT * V_DIM)).reshape(q_len, HEAD_CNT, V_DIM)

	print(f"Computing online softmax reference q_len={q_len}, kv_len={kv_len}, heads={HEAD_CNT}...")
	match_out_heads = []
	ref_out_heads = []
	ref_dQ_heads = []
	ref_dK_heads = []
	ref_dV_heads = []
	for i_head in range(HEAD_CNT):
		Q = Q_bf16[:, i_head, :].float().requires_grad_(True)
		K = KV_bf16[:, i_head, :QK_DIM].float().requires_grad_(True)
		V = KV_bf16[:, i_head, :V_DIM].float().requires_grad_(True)

		match_out = reference_online_softmax(Q.detach(), K.detach(), V.detach(), sink=sink_arg, gate=gate_arg, window_size=window_size)
		ref_out = reference_exact(Q, K, V, sink=sink_arg, gate=gate_arg, window_size=window_size)
		ref_out.backward(dO_bf16[:, i_head, :].float())

		match_out_heads.append(match_out.to(torch.bfloat16))
		ref_out_heads.append(ref_out.to(torch.bfloat16))
		ref_dQ_heads.append(Q.grad.to(torch.bfloat16))
		ref_dK_heads.append(K.grad.to(torch.bfloat16))
		ref_dV_heads.append(V.grad.to(torch.bfloat16))

	match_out_bf16 = torch.stack(match_out_heads, dim=1)
	ref_out_bf16 = torch.stack(ref_out_heads, dim=1)
	ref_dQ_bf16 = torch.stack(ref_dQ_heads, dim=1)
	ref_dK_bf16 = torch.stack(ref_dK_heads, dim=1)
	ref_dV_bf16 = torch.stack(ref_dV_heads, dim=1)

	print(f"\nFirst 4 rows, first 8 cols of head 0:")
	print_matrix(match_out_bf16[:4, 0, :8].float())
	print(f"\nLast 4 rows, last 8 cols of head 0:")
	print_matrix(match_out_bf16[-4:, 0, -8:].float())

	# Compare forward output with CUDA
	cuda_path = "tmp/out_cpu.bin"
	if os.path.exists(cuda_path):
		cuda_raw = np.fromfile(cuda_path, dtype=np.uint16)
		expected = q_len * HEAD_CNT * V_DIM
		if cuda_raw.size != expected:
			print(f"\nWarning: tmp/out_cpu.bin has {cuda_raw.size} uint16 elements, "
				f"expected {expected} ({q_len}x{HEAD_CNT * V_DIM}). "
				f"Make sure Q_LEN/KV_LEN match in the CUDA code.")
			return

		cuda_bf16 = torch.from_numpy(cuda_raw.view(np.int16)).view(torch.bfloat16).reshape(q_len, HEAD_CNT * V_DIM)
		compare("Forward (online softmax)", flatten_heads(match_out_bf16), cuda_bf16, q_len, HEAD_CNT * V_DIM)
		compare("Forward (exact)", flatten_heads(ref_out_bf16), cuda_bf16, q_len, HEAD_CNT * V_DIM)
	else:
		print(f"\nNo {cuda_path} found — run the CUDA kernel first to compare.")

	# Load CUDA's L and O for the matching backward reference
	L_cuda = load_f32("tmp/L.bin", (HEAD_CNT, q_len)) if os.path.exists("tmp/L.bin") else None
	O_cuda = load_bf16("tmp/out_cpu.bin", (q_len, HEAD_CNT * V_DIM)).reshape(q_len, HEAD_CNT, V_DIM) if os.path.exists("tmp/out_cpu.bin") else None

	print(f"\n=== Backward verification ===")

	# Verify D = rowsum(dO ⊙ O)
	D_path = "tmp/D.bin"
	if os.path.exists(D_path) and O_cuda is not None:
		D_cuda = load_f32(D_path, (HEAD_CNT, q_len))
		D_ref = ((dO_bf16.float() * O_cuda.float()).sum(dim=-1) / gate_arg).transpose(0, 1)
		diff = (D_ref - D_cuda).abs()
		exact = (D_ref == D_cuda).sum().item()
		print(f"\n--- D' = rowsum(dO ⊙ O) / gate ---")
		print(f"Max abs diff:  {diff.max().item():.6e}")
		total = HEAD_CNT * q_len
		print(f"Exact match:   {exact}/{total} ({100*exact/total:.2f}%)")

	# Verify dQ
	dQ_path = "tmp/dQ.bin"
	if os.path.exists(dQ_path):
		dQ_cuda = load_bf16(dQ_path, (q_len, HEAD_CNT * QK_DIM))
		compare("dQ (autograd f32)", flatten_heads(ref_dQ_bf16), dQ_cuda, q_len, HEAD_CNT * QK_DIM)

	# Verify dK
	dK_path = "tmp/dK.bin"
	if os.path.exists(dK_path):
		dK_cuda = load_bf16(dK_path, (kv_len, HEAD_CNT * QK_DIM))
		compare("dK (autograd f32)", flatten_heads(ref_dK_bf16), dK_cuda, kv_len, HEAD_CNT * QK_DIM)

	# Verify dV
	dV_path = "tmp/dV.bin"
	if os.path.exists(dV_path):
		dV_cuda = load_bf16(dV_path, (kv_len, HEAD_CNT * V_DIM))
		compare("dV (autograd f32)", flatten_heads(ref_dV_bf16), dV_cuda, kv_len, HEAD_CNT * V_DIM)

	# Matching backward reference using CUDA's L and O
	# This reconstructs P from the kernel's stored L and uses D from CUDA's O,
	# eliminating forward-pass divergence as a source of backward mismatch.
	if L_cuda is not None and O_cuda is not None:
		print(f"\n=== Matching backward (from CUDA L and O) ===")
		match_dQ_heads = []
		match_dK_heads = []
		match_dV_heads = []
		for i_head in range(HEAD_CNT):
			dO_f = dO_bf16[:, i_head, :].float()
			Q_f = Q_bf16[:, i_head, :].float()
			K_f = KV_bf16[:, i_head, :QK_DIM].float()
			V_f = KV_bf16[:, i_head, :V_DIM].float()

			# Recompute scores (multiply by combined score_scale in one step to match
			# the kernel's multiplication order and avoid f32 rounding differences)
			n = ssmax_n(q_len, window_size)
			ssmax_scale = torch.log2(n)
			score_scale = SCORE_SCALE * ssmax_scale
			scores = Q_f @ K_f.T
			scores = scores * score_scale.unsqueeze(1)

			causal = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=1)
			if window_size > 0:
				window = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=-window_size)
				causal = causal | window
			scores = scores.masked_fill(causal, float('-inf'))

			D_prime = (dO_f * O_cuda[:, i_head, :].float()).sum(dim=-1) / gate_arg
			dP = dO_f @ V_f.T
			logb_e = math.log2(math.e)

			grad_scale = (1.0 / logb_e) * gate_arg * score_scale
			L_prime = L_cuda[i_head] - torch.log2(grad_scale)
			P_prime = torch.exp2(scores - L_prime.unsqueeze(1))
			dS_dq = P_prime * (dP - D_prime.unsqueeze(1))
			match_dQ_heads.append((dS_dq.to(torch.bfloat16).to(torch.float32) @ K_f).to(torch.bfloat16))

			L_g = L_cuda[i_head] - math.log2(gate_arg)
			P = torch.exp2(scores - L_g.unsqueeze(1))
			match_dV_heads.append((P.T.to(torch.bfloat16).to(torch.float32) @ dO_f).to(torch.bfloat16))

			dS_dkv = P * (dP - D_prime.unsqueeze(1))
			dk_scale = score_scale / logb_e
			dS_dkv = dS_dkv * dk_scale.unsqueeze(1)
			match_dK_heads.append((dS_dkv.T.to(torch.bfloat16).to(torch.float32) @ Q_f).to(torch.bfloat16))

		match_dQ = torch.stack(match_dQ_heads, dim=1)
		match_dK = torch.stack(match_dK_heads, dim=1)
		match_dV = torch.stack(match_dV_heads, dim=1)

		if os.path.exists("tmp/dQ.bin"):
			dQ_cuda = load_bf16("tmp/dQ.bin", (q_len, HEAD_CNT * QK_DIM))
			compare("dQ (from CUDA L, O)", flatten_heads(match_dQ), dQ_cuda, q_len, HEAD_CNT * QK_DIM)

		if os.path.exists("tmp/dK.bin"):
			dK_cuda = load_bf16("tmp/dK.bin", (kv_len, HEAD_CNT * QK_DIM))
			compare("dK (from CUDA L, O)", flatten_heads(match_dK), dK_cuda, kv_len, HEAD_CNT * QK_DIM)

		if os.path.exists("tmp/dV.bin"):
			dV_cuda = load_bf16("tmp/dV.bin", (kv_len, HEAD_CNT * V_DIM))
			compare("dV (from CUDA L, O)", flatten_heads(match_dV), dV_cuda, kv_len, HEAD_CNT * V_DIM)


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
