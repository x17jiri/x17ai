#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import torch

from jsonc import load_jsonc

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "block.config.json"
TENSOR_DIR = ROOT / "tmp" / "block_torch"

my_device = torch.device("cpu") # or torch.device("cuda")
torch.set_default_device(my_device)
torch.set_default_dtype(torch.float32)

config = load_jsonc(CONFIG_PATH)
N_INPUTS = int(config["n_inputs"])
D_MODEL = int(config["d_model"])
N_HEADS = int(config["n_heads"])
HEAD_DIM = int(config["head_dim"])
ROPE_DIM = int(config["rope_dim"])
QKV_FAN_IN = int(config["qkv_fan_in"])
WINDOW_SIZE = int(config["window_size"])
L2_NORM_EPS = float(config["l2_norm_eps"])
ROPE_BASE = float(config["rope_base"])
QKV_ROWS = 4 * N_HEADS * HEAD_DIM
Q_ROWS = N_HEADS * HEAD_DIM
V_SCALE = math.sqrt(D_MODEL / QKV_FAN_IN)
V_SCALE_FIX = 1.5

def tensor_path(name: str) -> Path:
	return TENSOR_DIR / name

def load_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	raw = path.read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.int16)
	return data.view(torch.bfloat16).reshape(rows, cols).to(my_device).to(torch.float32)

def load_f32_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	raw = path.read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.float32)
	return data.reshape(rows, cols).to(my_device)

def store_tensor(tensor: torch.Tensor, file_name: str) -> None:
	path = tensor_path(file_name)
	path.parent.mkdir(parents=True, exist_ok=True)
	data = tensor.contiguous().to(torch.bfloat16).cpu()
	with path.open("wb") as output_file:
		output_file.write(data.view(torch.int16).numpy().tobytes())
	shape_str = ", ".join(str(dim) for dim in data.shape)
	print(f"Created {path}: [{shape_str}] bfloat16")

def store_f32_tensor(tensor: torch.Tensor, file_name: str) -> None:
	path = tensor_path(file_name)
	path.parent.mkdir(parents=True, exist_ok=True)
	data = tensor.contiguous().cpu().to(torch.float32)
	with path.open("wb") as output_file:
		output_file.write(data.numpy().tobytes())
	shape_str = ", ".join(str(dim) for dim in data.shape)
	print(f"Created {path}: [{shape_str}] float32")

def create_inputs() -> None:
	generator = torch.Generator(device=my_device)
	generator.manual_seed(42)
	inputs = torch.randn((N_INPUTS, D_MODEL), generator=generator)
	inputs_l2 = l2_norm_last_dim(inputs, L2_NORM_EPS)
	qkv_weights = torch.randn((QKV_ROWS, QKV_FAN_IN), generator=generator)
	qk_norm_scales = torch.full((1, Q_ROWS), 1.0)
	sink_k = torch.randn((N_HEADS, HEAD_DIM), generator=generator)
	sink_k = l2_norm_last_dim(sink_k, L2_NORM_EPS)
	sinks_v = torch.randn((N_HEADS, HEAD_DIM), generator=generator)
	store_tensor(inputs, "inputs.bin")
	store_tensor(inputs_l2, "inputs_l2.bin")
	store_tensor(qkv_weights, "qkv_weights.bin")
	store_tensor(qk_norm_scales, "qk_norm_scales.bin")
	store_tensor(sink_k, "sinks_k.bin")
	store_tensor(sinks_v, "sinks_v.bin")

def expand_qkv_weights(qkv_weights: torch.Tensor) -> torch.Tensor:
	if D_MODEL % HEAD_DIM != 0:
		raise ValueError(f"Expected d_model={D_MODEL} to be divisible by head_dim={HEAD_DIM}")
	step = D_MODEL // HEAD_DIM
	expanded = torch.zeros(
		(qkv_weights.shape[0], D_MODEL)
	)
	cols = torch.arange(qkv_weights.shape[1], dtype=torch.int64)
	for row in range(qkv_weights.shape[0]):
		indices = (cols + row * step) % D_MODEL
		expanded[row, indices] = qkv_weights[row]
	return expanded

def l2_norm_last_dim(tensor: torch.Tensor, eps: float) -> torch.Tensor:
	norm = torch.linalg.vector_norm(tensor, ord=2, dim=-1, keepdim=True)
	return tensor / (norm + eps)

def apply_rope(tensor: torch.Tensor, rope_base: float) -> torch.Tensor:
	head_dim = tensor.shape[-1]
	if head_dim % 2 != 0:
		raise ValueError(f"Expected even head_dim for RoPE, got {head_dim}")
	half_dim = head_dim // 2
	device = tensor.device
	inv_freq = torch.pow(
		torch.tensor(rope_base, device=device),
		-2.0 * torch.arange(half_dim, device=device) / float(head_dim),
	)
	positions = torch.arange(tensor.shape[0], device=device)
	theta = positions[:, None] * inv_freq[None, :]
	cos = torch.cos(theta).unsqueeze(1)
	sin = torch.sin(theta).unsqueeze(1)
	even = tensor[..., 0::2]
	odd = tensor[..., 1::2]
	result = tensor.clone()
	result[..., 0::2] = even * cos - odd * sin
	result[..., 1::2] = even * sin + odd * cos
	return result

def apply_rope_rows(tensor: torch.Tensor, positions: torch.Tensor, rope_base: float) -> torch.Tensor:
	head_dim = tensor.shape[-1]
	if head_dim % 2 != 0:
		raise ValueError(f"Expected even head_dim for RoPE, got {head_dim}")
	device = tensor.device
	half_dim = head_dim // 2
	inv_freq = torch.pow(
		torch.tensor(rope_base, device=device),
		-2.0 * torch.arange(half_dim, device=device) / float(head_dim),
	)
	theta = positions.to(device=device, dtype=tensor.dtype)[:, None] * inv_freq[None, :]
	cos = torch.cos(theta)
	sin = torch.sin(theta)
	even = tensor[..., 0::2]
	odd = tensor[..., 1::2]
	result = tensor.clone()
	result[..., 0::2] = even * cos - odd * sin
	result[..., 1::2] = even * sin + odd * cos
	return result

def quantize_(tensor: torch.Tensor) -> torch.Tensor:
	return tensor.to(torch.bfloat16).to(torch.float32)

def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
	ck = math.sqrt(2.0 / math.pi)
	ck3 = 0.044715 * ck
	y = ck3 * x * x * x + ck * x
	return 0.5 * x * torch.tanh(y) + 0.5 * x

def geglu(gate: torch.Tensor, lin: torch.Tensor) -> torch.Tensor:
	return gelu_tanh_approx(gate) * lin

def zig_zag_geglu(attn_out: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
	out = torch.empty_like(attn_out)
	out[..., 0::2] = geglu(g[..., 0::2], attn_out[..., 0::2])
	out[..., 1::2] = geglu(attn_out[..., 1::2], g[..., 1::2])
	return out

def calculate_sink_scores(q: torch.Tensor, sinks_k: torch.Tensor) -> torch.Tensor:
	# sink score calculation is part of the qkv kernel. It has access to precise q, but it loads
	# sinks_k from global memory in bf16
	#q = q.to(torch.bfloat16).to(torch.float32)
	sinks_k = sinks_k.to(torch.bfloat16).to(torch.float32)
	return torch.einsum("qhd,hd->qh", q, sinks_k)

def qkv_proj(
	inputs: torch.Tensor,
	qkv_weights: torch.Tensor,
	qk_norm_scales: torch.Tensor,
	sinks_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	expanded_qkv_weights = expand_qkv_weights(qkv_weights)
	print(f"qkv_weights shape: {qkv_weights.shape}")
	print(f"Expanded qkv_weights shape: {expanded_qkv_weights.shape[0]} x {expanded_qkv_weights.shape[1]}")
	qkv = torch.matmul(inputs, expanded_qkv_weights.transpose(0, 1))
	qkv_cols = Q_ROWS

	q = qkv[:, 0 * qkv_cols:1 * qkv_cols].reshape(N_INPUTS, N_HEADS, HEAD_DIM)
	k = qkv[:, 1 * qkv_cols:2 * qkv_cols].reshape(N_INPUTS, N_HEADS, HEAD_DIM)
	v = qkv[:, 2 * qkv_cols:3 * qkv_cols].reshape(N_INPUTS, N_HEADS, HEAD_DIM)
	g = qkv[:, 3 * qkv_cols:4 * qkv_cols].reshape(N_INPUTS, N_HEADS, HEAD_DIM)

	q = l2_norm_last_dim(q, L2_NORM_EPS)
	k = l2_norm_last_dim(k, L2_NORM_EPS)

	q_scales = qk_norm_scales.reshape(1, N_HEADS, HEAD_DIM)
	q = q * q_scales

	sink_scores = calculate_sink_scores(q, sinks_k)
	q = apply_rope(q, ROPE_BASE)
	k = apply_rope(k, ROPE_BASE)
	return q, k, v, g, sink_scores

def ssmax(q_len, window_size=0):
	"""Compute SSMax scale factor n for each query position.

	n[i] = min(window_size, i) + (1 + e_approx)

	where:
	  i           = number of real tokens visible (causal: tokens 0..i-1)
	  window_size = caps the visible count when sliding window is enabled
	  e_approx    = integer approximation of e used by the CUDA kernel
	  1           = accounts for the sink token

	When window_size == 0 (disabled), min is a no-op: n[i] = i + 1 + e_approx.
	"""
	E_APPROX_PLUS_1 = 4.0
	visible_real_tokens = torch.arange(q_len, dtype=torch.float32)
	if window_size > 0:
		n = torch.clamp(visible_real_tokens, max=float(window_size)) + E_APPROX_PLUS_1
	else:
		n = visible_real_tokens + E_APPROX_PLUS_1
	return torch.log(n)

def attn_one_head(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sink_scores: torch.Tensor,
	sink_v: torch.Tensor,
	score_file_name: str | None,
) -> torch.Tensor:
	seq_len = q.shape[0]
	q = q.to(torch.bfloat16).to(torch.float32)
	k = k.to(torch.bfloat16).to(torch.float32)
	v = v.to(torch.bfloat16).to(torch.float32)
	sink_v = sink_v.to(torch.bfloat16).to(torch.float32)
	TEMPERATURE = math.sqrt(q.shape[1])
	sink_scores = sink_scores.to(torch.float32).unsqueeze(1)
	real_scores = torch.matmul(q, k.transpose(0, 1))
#	real_scores = quantize_(real_scores)
	if score_file_name is not None:
		store_f32_tensor(real_scores, score_file_name)
	scale = ssmax(seq_len, WINDOW_SIZE).unsqueeze(1) * TEMPERATURE
	scores = torch.cat((sink_scores, real_scores), dim=1) * scale
	values = torch.cat((sink_v.unsqueeze(0) * V_SCALE_FIX, v * (V_SCALE_FIX * V_SCALE)), dim=0)
	real_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
	if WINDOW_SIZE > 0:
		real_mask = real_mask | torch.tril(
			torch.ones(seq_len, seq_len, dtype=torch.bool),
			diagonal=-(WINDOW_SIZE + 1),
		)
	sink_mask = torch.zeros(seq_len, 1, dtype=torch.bool)
	scores = scores.masked_fill(torch.cat((sink_mask, real_mask), dim=1), float("-inf"))
	softmax = torch.softmax(scores, dim=1).to(torch.bfloat16).to(torch.float32)
	del scores
	return softmax @ values

def attn(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sink_scores: torch.Tensor,
	sinks_v: torch.Tensor,
) -> torch.Tensor:
	_, n_heads, _ = q.shape
	out = torch.empty_like(v)
	for h in range(n_heads):
		out[:, h, :] = attn_one_head(
			q[:, h, :],
			k[:, h, :],
			v[:, h, :],
			sink_scores[:, h],
			sinks_v[h, :],
			f"scores_{h}_f32.bin" if h < 1 else None,
		)
	return out

def join_qkvg(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
	return torch.cat(
		(
			q.reshape(q.shape[0], -1),
			k.reshape(k.shape[0], -1),
			v.reshape(v.shape[0], -1),
			g.reshape(g.shape[0], -1),
		),
		dim=1,
	)

def run_block() -> None:
	inputs = load_tensor("inputs_l2.bin", N_INPUTS, D_MODEL)
	qkv_weights = load_tensor("qkv_weights.bin", QKV_ROWS, QKV_FAN_IN)
	qk_norm_scales = load_tensor("qk_norm_scales.bin", 1, Q_ROWS)
	sinks_k = load_tensor("sinks_k.bin", N_HEADS, HEAD_DIM)
	sinks_v = load_tensor("sinks_v.bin", N_HEADS, HEAD_DIM)
	q, k, v, g, sink_scores = qkv_proj(inputs, qkv_weights, qk_norm_scales, sinks_k)
	attn_out = attn(q, k, v, sink_scores, sinks_v)
	attn_out = zig_zag_geglu(attn_out, g * V_SCALE)
	#attn_match = attn_matching(q, k, v, sinks_k, window_size, rope_base)
	qkvg = join_qkvg(q, k, v, g)

	print("inputs shape:", inputs.shape)
	print("qkvg shape:", qkvg.shape)
	print("q shape:", q.shape)
	print("k shape:", k.shape)
	print("v shape:", v.shape)
	print("sinks_k shape:", sinks_k.shape)
	print("sinks_v shape:", sinks_v.shape)
	print("sink_scores shape:", sink_scores.shape)
	print("attn_out shape:", attn_out.shape)
	#print("attn_match shape:", attn_match.shape)

	store_f32_tensor(q, "q_f32.bin")
	store_f32_tensor(k, "k_f32.bin")
	store_f32_tensor(v, "v_f32.bin")
	store_f32_tensor(g, "g_f32.bin")

	store_tensor(q, "q.bin")
	store_tensor(k, "k.bin")
	store_tensor(v, "v.bin")
	store_tensor(g, "g.bin")

	store_f32_tensor(sink_scores.transpose(0, 1), "sink_scores_f32.bin")
	store_tensor(attn_out, "attn_out.bin")
	#store_tensor(attn_match, "attn_matching.bin")
	store_tensor(qkvg, "qkvg.bin")

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--create-inputs", action="store_true")
	args = parser.parse_args()

	if args.create_inputs:
		create_inputs()
	else:
		run_block()

if __name__ == "__main__":
	main()
