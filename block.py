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

def load_config() -> dict:
	return load_jsonc(CONFIG_PATH)

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

def create_inputs(config: dict) -> None:
	n_inputs = int(config["n_inputs"])
	d_model = int(config["d_model"])
	n_heads = int(config["n_heads"])
	head_dim = int(config["head_dim"])
	qkv_fan_in = int(config["qkv_fan_in"])
	l2_norm_eps = float(config["l2_norm_eps"])
	qkv_rows = 3 * n_heads * head_dim
	qk_rows = 2 * n_heads * head_dim

	generator = torch.Generator(device=my_device)
	generator.manual_seed(42)
	inputs = torch.randn((n_inputs, d_model), generator=generator)
	qkv_weights = torch.randn((qkv_rows, qkv_fan_in), generator=generator)
	qk_norm_scales = torch.full((1, qk_rows), math.sqrt(head_dim))
	sinks = l2_norm_last_dim(torch.randn((n_heads, head_dim), generator=generator), l2_norm_eps)
	store_tensor(inputs, "inputs.bin")
	store_tensor(qkv_weights, "qkv_weights.bin")
	store_tensor(qk_norm_scales, "qk_norm_scales.bin")
	store_tensor(sinks, "sinks.bin")

def expand_qkv_weights(qkv_weights: torch.Tensor, d_model: int, n_heads: int) -> torch.Tensor:
	if d_model % n_heads != 0:
		raise ValueError(f"Expected d_model={d_model} to be divisible by n_heads={n_heads}")
	step = d_model // n_heads
	expanded = torch.zeros(
		(qkv_weights.shape[0], d_model)
	)
	cols = torch.arange(qkv_weights.shape[1], dtype=torch.int64)
	for row in range(qkv_weights.shape[0]):
		indices = (cols + row * step) % d_model
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

def qkv_proj(
	inputs: torch.Tensor,
	qkv_weights: torch.Tensor,
	qk_norm_scales: torch.Tensor,
	config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	n_inputs = int(config["n_inputs"])
	n_heads = int(config["n_heads"])
	head_dim = int(config["head_dim"])
	qkv_fan_in = int(config["qkv_fan_in"])
	l2_norm_eps = float(config["l2_norm_eps"])
	rope_base = float(config["rope_base"])
	d_model = int(config["d_model"])
	n_heads = int(config["n_heads"])

	expanded_qkv_weights = expand_qkv_weights(qkv_weights, d_model, n_heads)
	print(f"qkv_weights shape: {qkv_weights.shape}")
	print(f"Expanded qkv_weights shape: {expanded_qkv_weights.shape[0]} x {expanded_qkv_weights.shape[1]}")
	qkv = torch.matmul(inputs, expanded_qkv_weights.transpose(0, 1))
	qkv_cols = n_heads * head_dim

	q = qkv[:, :qkv_cols].reshape(n_inputs, n_heads, head_dim)
	k = qkv[:, qkv_cols:2 * qkv_cols].reshape(n_inputs, n_heads, head_dim)
	v = qkv[:, 2 * qkv_cols:3 * qkv_cols].reshape(n_inputs, n_heads, head_dim)

	qk_parts = [q, k]
	qk_scales = qk_norm_scales.reshape(1, 2, n_heads, head_dim)
	v_scale = 1.0 / math.sqrt(qkv_fan_in)
	for idx in range(len(qk_parts)):
		qk_parts[idx] = l2_norm_last_dim(qk_parts[idx], l2_norm_eps)
		qk_parts[idx] = qk_parts[idx] * qk_scales[:, idx]
	q, k = qk_parts
	v = v * v_scale
	#q = apply_rope(q, rope_base)
	k = apply_rope(k, rope_base)
	return q, k, v

def ssmax(q_len, window_size=0):
	"""Compute SSMax scale factor n for each query position.

	n[i] = min(window_size, i + 1) + (e + 1)

	where:
	  i + 1       = number of real tokens visible (causal: tokens 0..i)
	  window_size = caps the visible count when sliding window is enabled
	  e           = ensures ln(n) >= 1
	  1           = accounts for the sink token

	When window_size == 0 (disabled), min is a no-op: n[i] = i + 2 + e.
	"""
	E_PLUS_1 = math.e + 1.0
	pos_plus_1 = torch.arange(1, q_len + 1)
	if window_size > 0:
		n = torch.clamp(pos_plus_1, max=float(window_size)) + E_PLUS_1
	else:
		n = pos_plus_1 + E_PLUS_1
	return torch.log(n)

def ssmax_log2(q_len, window_size=0):
	"""SSMax scale in base-2, matching the CUDA kernel's logb (= log2)."""
	E_PLUS_1 = math.e + 1.0
	pos_plus_1 = torch.arange(1, q_len + 1)
	if window_size > 0:
		n = torch.clamp(pos_plus_1, max=float(window_size)) + E_PLUS_1
	else:
		n = pos_plus_1 + E_PLUS_1
	return torch.log2(n)

def attn_one_head_online(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sink: torch.Tensor,
	window_size: int,
	rope_base: float,
) -> torch.Tensor:
	"""Online softmax attention matching the CUDA kernel's algorithm.

	Processes keys in chunks of 16, uses base-2 exp/log,
	casts P to bf16 before P*V, folds in the sink term at the end,
	and normalizes at the end.
	"""
	KV_CHUNK = 16
	THRESHOLD = 5.0

	seq_len = q.shape[0]
	head_dim = q.shape[1]
	q = q.to(torch.bfloat16).to(torch.float32)
	k = k.to(torch.bfloat16).to(torch.float32)
	v = v.to(torch.bfloat16).to(torch.float32)
	sink = sink.to(torch.bfloat16).to(torch.float32)
	q_rope = apply_rope_rows(q, torch.arange(seq_len, device=q.device), rope_base)

	score_scale = ssmax_log2(seq_len, window_size)

	O = torch.zeros(seq_len, head_dim)

	for qi in range(seq_len):
		#print(f"Processing query {qi}/{seq_len}")
		q_row = q[qi:qi+1]
		q_row_rope = q_rope[qi:qi+1]
		q_sc = score_scale[qi].item()
		sink_scaled = torch.matmul(q_row, sink.unsqueeze(1)).item() * q_sc

		kv_first = max(0, qi - window_size + 1) if window_size > 0 else 0
		kv_begin = (kv_first // KV_CHUNK) * KV_CHUNK
		kv_end = min(((qi // KV_CHUNK) + 1) * KV_CHUNK, seq_len)

		running_max = -3.4e38 + THRESHOLD
		running_sum = 0.0
		O_acc = torch.zeros(1, head_dim)

		first_step = True
		for ki_start in range(kv_begin, kv_end, KV_CHUNK):
			ki_end = min(ki_start + KV_CHUNK, seq_len)
			k_chunk = k[ki_start:ki_end]
			v_chunk = v[ki_start:ki_end]

			S = (q_row_rope @ k_chunk.T) * q_sc

			for ki in range(ki_end - ki_start):
				k_pos = ki_start + ki
				if k_pos > qi:
					S[0, ki] = float("-inf")
				if window_size > 0 and qi - k_pos >= window_size:
					S[0, ki] = float("-inf")

			chunk_max = S.max().item()

			rescale = 1.0
			if (chunk_max - running_max) > THRESHOLD:
				new_max = chunk_max + THRESHOLD
				if not first_step:
					rescale = 2.0 ** (running_max - new_max)
					O_acc = O_acc * rescale
				running_max = new_max

			P = torch.pow(2.0, torch.clamp(S - running_max, min=-150.0))
			running_sum = running_sum * rescale + P.sum().item()

			P = P.to(torch.bfloat16).to(torch.float32)
			O_acc = O_acc + P @ v_chunk
			first_step = False

		if running_sum > 0:
			global_max = max(running_max, sink_scaled)
			real_rescale = 2.0 ** max(running_max - global_max, -150.0)
			sink_contrib = 2.0 ** max(sink_scaled - global_max, -150.0)
			total_sum = running_sum * real_rescale + sink_contrib
			O_acc = O_acc * (real_rescale / total_sum)
		O[qi] = O_acc[0]

	return O

def attn_one_head(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sink: torch.Tensor,
	window_size: int,
	rope_base: float,
) -> torch.Tensor:
	seq_len = q.shape[0]
	q = q.to(torch.bfloat16).to(torch.float32)
	k = k.to(torch.bfloat16).to(torch.float32)
	v = v.to(torch.bfloat16).to(torch.float32)
	sink = sink.to(torch.bfloat16).to(torch.float32)
	sink_scores = torch.matmul(q, sink.unsqueeze(1))
	q_rope = apply_rope_rows(q, torch.arange(seq_len, device=q.device), rope_base)
	real_scores = torch.matmul(q_rope, k.transpose(0, 1))
	scale = ssmax(seq_len, window_size).unsqueeze(1)
	scores = torch.cat((sink_scores, real_scores), dim=1) * scale
	real_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
	if window_size > 0:
		real_mask = real_mask | torch.tril(
			torch.ones(seq_len, seq_len, dtype=torch.bool),
			diagonal=-window_size,
		)
	sink_mask = torch.zeros(seq_len, 1, dtype=torch.bool)
	scores = scores.masked_fill(torch.cat((sink_mask, real_mask), dim=1), float("-inf"))
	softmax = torch.softmax(scores, dim=1).to(torch.bfloat16).to(torch.float32)
	return softmax[:, 1:] @ v

def attn(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sinks: torch.Tensor,
	window_size: int,
	rope_base: float,
) -> torch.Tensor:
	_, n_heads, _ = q.shape
	out = torch.empty_like(v)
	for h in range(n_heads):
		out[:, h, :] = attn_one_head(q[:, h, :], k[:, h, :], v[:, h, :], sinks[h], window_size, rope_base)
	return out

def attn_matching(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sinks: torch.Tensor,
	window_size: int,
	rope_base: float,
) -> torch.Tensor:
	_, n_heads, _ = q.shape
	out = torch.empty_like(v)
	for h in range(n_heads):
		print(f"Processing head {h}/{n_heads}")
		out[:, h, :] = attn_one_head_online(q[:, h, :], k[:, h, :], v[:, h, :], sinks[h], window_size, rope_base)
	return out

def join_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
	return torch.cat(
		(
			q.reshape(q.shape[0], -1),
			k.reshape(k.shape[0], -1),
			v.reshape(v.shape[0], -1),
		),
		dim=1,
	)

def run_block(config: dict) -> None:
	n_inputs = int(config["n_inputs"])
	d_model = int(config["d_model"])
	n_heads = int(config["n_heads"])
	head_dim = int(config["head_dim"])
	qkv_fan_in = int(config["qkv_fan_in"])
	qkv_rows = 3 * n_heads * head_dim
	qk_rows = 2 * n_heads * head_dim
	window_size = int(config["window_size"])
	rope_base = float(config["rope_base"])

	inputs = load_tensor("inputs.bin", n_inputs, d_model)
	qkv_weights = load_tensor("qkv_weights.bin", qkv_rows, qkv_fan_in)
	qk_norm_scales = load_tensor("qk_norm_scales.bin", 1, qk_rows)
	sinks = load_tensor("sinks.bin", n_heads, head_dim)
	q, k, v = qkv_proj(inputs, qkv_weights, qk_norm_scales, config)
	attn_out = attn(q, k, v, sinks, window_size, rope_base)
	#attn_match = attn_matching(q, k, v, sinks, window_size, rope_base)
	qkv = join_qkv(q, k, v)

	print("inputs shape:", inputs.shape)
	print("qkv shape:", qkv.shape)
	print("q shape:", q.shape)
	print("k shape:", k.shape)
	print("v shape:", v.shape)
	print("sinks shape:", sinks.shape)
	print("attn_out shape:", attn_out.shape)
	#print("attn_match shape:", attn_match.shape)

	store_tensor(q, "q.bin")
	store_tensor(k, "k.bin")
	store_tensor(v, "v.bin")
	store_tensor(attn_out, "attn_out.bin")
	#store_tensor(attn_match, "attn_matching.bin")
	store_tensor(qkv, "qkv.bin")

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--create-inputs", action="store_true")
	args = parser.parse_args()

	config = load_config()

	if args.create_inputs:
		create_inputs(config)
	else:
		run_block(config)

if __name__ == "__main__":
	main()
