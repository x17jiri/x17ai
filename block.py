#!/usr/bin/env python3

import argparse
import math

import torch

from block_utils import *

# Expected variances at initialization (after `create_inputs()`):
#
# - Each row of `inputs_l2` has unit norm, so each coordinate contributes variance about 1 / D_MODEL
#
# - Raw projected q/k/v/g coordinates therefore have variance about QKV_FAN_IN / D_MODEL.
#
# - q and k are then L2-normalized per head; with qk_norm_scales = 1, each coordinate has
#   variance about 1 / HEAD_DIM, and RoPE preserves that variance.
#
# - v and g are not L2-normalized, so their raw projection variance stays at
#   QKV_FAN_IN / D_MODEL.
#
# - We therefore choose `SPARSE_SCALE = sqrt(D_MODEL / QKV_FAN_IN)` so that
#   Var(v * SPARSE_SCALE) ~= (QKV_FAN_IN / D_MODEL) * (D_MODEL / QKV_FAN_IN) = 1,
#   and the same correction makes `g * SPARSE_SCALE` unit-variance as well.
#
# - sink_scores = dot(q, sinks_k) has variance about 1 / HEAD_DIM because both inputs are
#   per-head unit vectors after L2 normalization.
#
# - real_scores = q @ k^T also has variance about 1 / HEAD_DIM before masking.
#
# - TEMPERATURE = sqrt(HEAD_DIM), so sink_scores * TEMPERATURE, real_scores * TEMPERATURE,
#   and the concatenated `scores` tensor before SSMax all have variance about 1.
#
# - sink_v is drawn directly from N(0, 1), so each coordinate has variance about 1.
#
# - The sink row of `values` is sink_v * V_SCALE_FIX, so its variance is about V_SCALE_FIX^2.
#
# - The real-token rows of `values` are v * (V_SCALE_FIX * SPARSE_SCALE). Since
#   Var(v) ~= QKV_FAN_IN / D_MODEL and SPARSE_SCALE^2 = D_MODEL / QKV_FAN_IN,
#   those rows also have variance about V_SCALE_FIX^2.
#
# - Therefore the concatenated `values` tensor fed into attention has variance about
#   V_SCALE_FIX^2.
#
# - V_SCALE_FIX was chosen empirically to get the variance of `attn_out_pregate` to around 1.
#   The same value works regardless of sequence length thanks to SSMax.
#   You can run `python ssmax_stats.py` to plot the variance of `attn_out_pregate`
#   depending on position and verify that there is no visible drift.
#
# - In `attn_out = zig_zag_geglu(attn_out_pregate, g * SPARSE_SCALE)`, both GeGLU inputs have
#   variance about 1: `attn_out_pregate` by the choice of `V_SCALE_FIX`, and `g * SPARSE_SCALE`
#   because Var(g) ~= QKV_FAN_IN / D_MODEL and SPARSE_SCALE^2 = D_MODEL / QKV_FAN_IN.
#
# - For independent unit-variance inputs, exact version of GeGLU would have variance
#   `1/3 + 1/(2*pi*sqrt(3))` ~ 0.4252. We use the same variance-fix expression as CUDA:
#   `GELU_VAR_FIX_2 = 1 / (1/3 + 1/(2*pi*sqrt(3)))`,
#   `GELU_VAR_FIX = sqrt(GELU_VAR_FIX_2)`.
#
# - We scale each GeGLU branch so its own output projection has variance about 1:
#   `ATTN_GEGLU_SCALE = GELU_VAR_FIX / sqrt(ATTN_WIDTH)` and
#   `F_GEGLU_SCALE = GELU_VAR_FIX / sqrt(F_WIDTH)`.
#
# - Therefore each coordinate of `attn_out` has variance about `1 / ATTN_WIDTH`, so the attention
#   output projection contributes total variance about `ATTN_WIDTH * (1 / ATTN_WIDTH) = 1`.
#
# - `f_pregate = inputs_l2 @ f_weights^T` has variance about 1 because it is a dense projection
#   from a unit-norm input with unit-variance weights. After the same pairwise GeGLU and
#   `F_GEGLU_SCALE`, each coordinate of `f` has variance about `1 / F_WIDTH`, so the forward
#   output projection also contributes total variance about `F_WIDTH * (1 / F_WIDTH) = 1`.
#
# - The output projection is now split into two dense projections: `o_attn` from attention and
#   `o_ffn` from the FFN branch. With unit-variance `w_attn` and `w_ffn`, each branch output has
#   variance about 1.

def quantize_(tensor: torch.Tensor) -> torch.Tensor:
	return tensor.to(torch.bfloat16).to(torch.float32)

def new_randn(*shape, generator):
	return torch.randn(shape, generator=generator)

def new_ones(*shape):
	return torch.full(shape, 1.0)

def l2_norm(tensor: torch.Tensor, eps: float = L2_NORM_EPS) -> torch.Tensor:
	norm = torch.linalg.vector_norm(tensor, ord=2, dim=-1, keepdim=True)
	return tensor / (norm + eps)

def create_inputs() -> None:
	generator = torch.Generator(device=my_device)
	generator.manual_seed(42)

	# randn init
	inputs = new_randn(N_INPUTS, D_MODEL, generator=generator)
	qkvg_weights = new_randn(QKVG_ROWS, SPARSE_FAN_IN, generator=generator)
	f_weights = new_randn(F_PROJ_OUTPUTS, SPARSE_FAN_IN, generator=generator)
	w_attn = new_randn(D_MODEL, ATTN_WIDTH, generator=generator)
	w_ffn = new_randn(D_MODEL, F_WIDTH, generator=generator)
	sink_k = new_randn(N_HEADS, HEAD_DIM, generator=generator)
	sinks_v = new_randn(N_HEADS, HEAD_DIM, generator=generator)

	# constant value init
	qk_norm_scales = new_ones(1, HEAD_DIM * N_HEADS)

	inputs_l2 = l2_norm(inputs)
	sink_k = l2_norm(sink_k)

	store_tensor(inputs, "inputs.bin", expected_variance=1.0)
	store_tensor(inputs_l2, "inputs_l2.bin", expected_variance=1.0 / D_MODEL)
	store_tensor(qkvg_weights, "qkvg_weights.bin", expected_variance=1.0)
	store_tensor(f_weights, "f_weights.bin", expected_variance=1.0)
	store_tensor(w_attn, "w_attn.bin", expected_variance=1.0)
	store_tensor(w_ffn, "w_ffn.bin", expected_variance=1.0)
	store_tensor(qk_norm_scales, "qk_norm_scales.bin", expected_variance=0.0)
	store_tensor(sink_k, "sinks_k.bin", expected_variance=1.0 / HEAD_DIM)
	store_tensor(sinks_v, "sinks_v.bin", expected_variance=1.0)

def sparse_weights(w: torch.Tensor, d_input, repeat_after) -> torch.Tensor:
	d_output = w.shape[0]
	fan_in = w.shape[1]
	step = d_input // repeat_after
	sparse = torch.zeros((d_output, d_input))
	cols = torch.arange(fan_in, dtype=torch.int64)
	for row in range(d_output):
		indices = (cols + row * step) % d_input
		sparse[row, indices] = w[row]
	return sparse

def apply_rope(tensor: torch.Tensor, rope_base: float) -> torch.Tensor:
	head_dim = tensor.shape[-1]
	assert head_dim % 2 == 0

	half_dim = head_dim // 2
	device = tensor.device
	inv_freq = torch.pow(
		torch.tensor(rope_base, device=device),
		-2.0 * torch.arange(half_dim, device=device) / float(head_dim),
	)
	positions = torch.arange(tensor.shape[0], device=device)
	theta = positions[:, None] * inv_freq[None, :]
	broadcast_shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 2) + (half_dim,)
	cos = torch.cos(theta).reshape(broadcast_shape)
	sin = torch.sin(theta).reshape(broadcast_shape)
	pairs = tensor.reshape(*tensor.shape[:-1], half_dim, 2)
	even, odd = pairs.unbind(dim=-1)
	rotated = torch.stack(
		(
			even * cos - odd * sin,
			even * sin + odd * cos,
		),
		dim=-1,
	)
	return rotated.reshape_as(tensor)

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
	broadcast_shape = (positions.shape[0],) + (1,) * (tensor.ndim - 2) + (half_dim,)
	cos = torch.cos(theta).reshape(broadcast_shape)
	sin = torch.sin(theta).reshape(broadcast_shape)
	pairs = tensor.reshape(*tensor.shape[:-1], half_dim, 2)
	even, odd = pairs.unbind(dim=-1)
	rotated = torch.stack(
		(
			even * cos - odd * sin,
			even * sin + odd * cos,
		),
		dim=-1,
	)
	return rotated.reshape_as(tensor)

def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
	ck = math.sqrt(2.0 / math.pi)
	ck3 = 0.044715 * ck
	y = ck3 * x * x * x + ck * x
	return 0.5 * x * torch.tanh(y) + 0.5 * x

def d_gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
	ck = math.sqrt(2.0 / math.pi)
	ck3 = 0.044715 * ck
	x2 = x * x
	y = ck3 * x * x2 + ck * x
	tanh_y = torch.tanh(y)
	dy_dx = 3.0 * ck3 * x2 + ck
	return 0.5 * (1.0 + tanh_y + x * (1.0 - tanh_y * tanh_y) * dy_dx)

def geglu(gate: torch.Tensor, lin: torch.Tensor) -> torch.Tensor:
	return gelu_tanh_approx(gate) * lin

def calculate_sink_scores(q: torch.Tensor, sinks_k: torch.Tensor) -> torch.Tensor:
	# sink score calculation is part of the qkvg kernel. It has access to precise q, but it loads
	# sinks_k from global memory in bf16
	#q = q.to(torch.bfloat16).to(torch.float32)
	sinks_k = sinks_k.to(torch.bfloat16).to(torch.float32)
	return torch.einsum("qhd,hd->qh", q, sinks_k)

def qkvg_proj(
	inputs: torch.Tensor,
	qkvg_weights: torch.Tensor,
	qk_norm_scales: torch.Tensor,
	sinks_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	qkvg = torch.matmul(inputs, qkvg_weights.transpose(0, 1))
	qkvg_cols = HEAD_DIM * N_HEADS

	q = qkvg[:, 0 * qkvg_cols:1 * qkvg_cols].reshape(N_INPUTS, N_HEADS, HEAD_DIM)
	k = qkvg[:, 1 * qkvg_cols:2 * qkvg_cols].reshape(N_INPUTS, N_HEADS, HEAD_DIM)
	v = qkvg[:, 2 * qkvg_cols:3 * qkvg_cols].reshape(N_INPUTS, N_HEADS, HEAD_DIM)
	g = qkvg[:, 3 * qkvg_cols:4 * qkvg_cols].reshape(N_INPUTS, N_HEADS, HEAD_DIM)

	q = l2_norm(q)
	k = l2_norm(k)

	q_scales = qk_norm_scales.reshape(1, N_HEADS, HEAD_DIM)
	q = q * q_scales

	sink_scores = calculate_sink_scores(q, sinks_k)
	q = apply_rope(q, ROPE_BASE)
	k = apply_rope(k, ROPE_BASE)
	return q, k, v, g, sink_scores

def ssmax_n(q_len, window_size=0):
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
	return n.unsqueeze(1)

def attn_one_head(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sink_scores: torch.Tensor,
	sink_v: torch.Tensor,
	score_file_name: str | None,
	prob_file_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
	seq_len = q.shape[0]
	QK_DIM = q.shape[1]

	q = quantize_(q)
	k = quantize_(k)
	v = quantize_(v) * (SPARSE_SCALE * V_SCALE_FIX)

	sink_scores = sink_scores.unsqueeze(1)
	sink_v = quantize_(sink_v.unsqueeze(0)) * V_SCALE_FIX

	S = q @ k.transpose(0, 1)

	BASE_TEMPERATURE = math.sqrt(QK_DIM)
	temperature = BASE_TEMPERATURE * torch.log(ssmax_n(seq_len, WINDOW_SIZE))

	mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
	if WINDOW_SIZE > 0:
		mask = mask | torch.tril(
			torch.ones(seq_len, seq_len, dtype=torch.bool),
			diagonal = -(WINDOW_SIZE + 1),
		)
	S = S.masked_fill(mask, float("-inf"))
	S = torch.cat((sink_scores, S), dim=1)

	S *= temperature
	max = S.amax(dim=1, keepdim=True)
	S -= max
	P = torch.exp(S)
	sum = P.sum(dim=1, keepdim=True)

	P = torch.cat((P[:, :1], quantize_(P[:, 1:])), dim=1)
	v = torch.cat((sink_v, v), dim=0)

	o = P @ v

	sum_recip = torch.reciprocal(sum)
	o *= sum_recip

	LOG2_E = 1.0 / math.log(2.0)
	return o, max.squeeze(1) * LOG2_E

def attn(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sink_scores: torch.Tensor,
	sinks_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	_, n_heads, _ = q.shape
	out = torch.empty_like(v)
	max_values = torch.empty((q.shape[0], n_heads), dtype=torch.float32)
	for h in range(n_heads):
		out[:, h, :], max_values[:, h] = attn_one_head(
			q[:, h, :],
			k[:, h, :],
			v[:, h, :],
			sink_scores[:, h],
			sinks_v[h, :],
			f"scores_{h}_f32.bin" if h < 1 else None,
			f"attn_probs_{h}.bin" if h < 1 else None,
		)
	return out, max_values

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

def pairwise_geglu(tensor: torch.Tensor, output_scale: float) -> torch.Tensor:
	if tensor.shape[-1] % 2 != 0:
		raise ValueError(f"Expected even projection width for GeGLU, got {tensor.shape[-1]}")
	return geglu(tensor[..., 0::2], tensor[..., 1::2]) * output_scale

def pairwise_geglu_backward_multipliers(
	tensor: torch.Tensor,
	input_scale: float,
	output_scale: float,
	) -> torch.Tensor:
	if tensor.shape[-1] % 2 != 0:
		raise ValueError(f"Expected even projection width for GeGLU, got {tensor.shape[-1]}")
	gate = tensor[..., 0::2]
	lin = tensor[..., 1::2]
	# Store d(GeGLU) / d(raw pregate) so multiplying by d_out feeds directly into the input GEMM backward.
	scaled_gate = gate * input_scale
	backvec = torch.empty_like(tensor)
	backvec[..., 0::2] = lin * d_gelu_tanh_approx(scaled_gate) * output_scale * input_scale * input_scale
	backvec[..., 1::2] = gelu_tanh_approx(scaled_gate) * output_scale * input_scale
	return backvec

def f_proj_pregate(inputs: torch.Tensor, f_weights: torch.Tensor) -> torch.Tensor:
	if inputs.shape[-1] != D_MODEL:
		raise ValueError(f"Expected input width {D_MODEL}, got {inputs.shape[-1]}")
	if f_weights.shape != (F_PROJ_OUTPUTS, D_MODEL):
		raise ValueError(f"Expected f_weights shape {(F_PROJ_OUTPUTS, D_MODEL)}, got {tuple(f_weights.shape)}")
	inputs = quantize_(inputs)
	f_weights = quantize_(f_weights)
	return torch.matmul(inputs, f_weights.transpose(0, 1))

def f_proj(inputs: torch.Tensor, f_weights: torch.Tensor) -> torch.Tensor:
	return pairwise_geglu(f_proj_pregate(inputs, f_weights), F_GEGLU_SCALE)

def o_proj_attn(attn_out: torch.Tensor, w_attn: torch.Tensor) -> torch.Tensor:
	flat_attn_out = attn_out.reshape(attn_out.shape[0], -1)
	if flat_attn_out.shape[1] != ATTN_WIDTH:
		raise ValueError(
			f"Expected flattened attn_out width {ATTN_WIDTH}, got {flat_attn_out.shape[1]}"
		)
	if w_attn.shape != (D_MODEL, ATTN_WIDTH):
		raise ValueError(
			f"Expected w_attn shape {(D_MODEL, ATTN_WIDTH)}, got {tuple(w_attn.shape)}"
		)
	flat_attn_out = quantize_(flat_attn_out)
	w_attn = quantize_(w_attn)
	return torch.matmul(flat_attn_out, w_attn.transpose(0, 1))

def o_proj_ffn(f: torch.Tensor, w_ffn: torch.Tensor) -> torch.Tensor:
	if f.ndim != 2:
		raise ValueError(f"Expected f to be rank-2, got rank {f.ndim}")
	if f.shape[1] != F_WIDTH:
		raise ValueError(f"Expected f width {F_WIDTH}, got {f.shape[1]}")
	if w_ffn.shape != (D_MODEL, F_WIDTH):
		raise ValueError(
			f"Expected w_ffn shape {(D_MODEL, F_WIDTH)}, got {tuple(w_ffn.shape)}"
		)
	f = quantize_(f)
	w_ffn = quantize_(w_ffn)
	return torch.matmul(f, w_ffn.transpose(0, 1))

def o_proj_ffn_backward(
	f: torch.Tensor,
	w_ffn: torch.Tensor,
	d_o_ffn: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
	f_leaf = f.detach().clone().requires_grad_(True)
	w_ffn_leaf = w_ffn.detach().clone().requires_grad_(True)
	o_ffn = o_proj_ffn(f_leaf, w_ffn_leaf)
	o_ffn.backward(d_o_ffn)
	if f_leaf.grad is None:
		raise RuntimeError("Expected gradient for f after o_proj_ffn backward")
	if w_ffn_leaf.grad is None:
		raise RuntimeError("Expected gradient for w_ffn after o_proj_ffn backward")
	return f_leaf.grad, w_ffn_leaf.grad

def run_block() -> None:
	inputs = load_tensor("inputs_l2.bin", N_INPUTS, D_MODEL)
	qkvg_weights = load_tensor("qkvg_weights.bin", QKVG_ROWS, SPARSE_FAN_IN)
	f_weights = load_tensor("f_weights.bin", F_PROJ_OUTPUTS, SPARSE_FAN_IN)
	w_attn = load_tensor("w_attn.bin", D_MODEL, ATTN_WIDTH)
	w_ffn = load_tensor("w_ffn.bin", D_MODEL, F_WIDTH)
	qk_norm_scales = load_tensor("qk_norm_scales.bin", 1, HEAD_DIM * N_HEADS)
	sinks_k = load_tensor("sinks_k.bin", N_HEADS, HEAD_DIM)
	sinks_v = load_tensor("sinks_v.bin", N_HEADS, HEAD_DIM)

	qkvg_weights = sparse_weights(qkvg_weights, D_MODEL, HEAD_DIM);
	q, k, v, g, sink_scores = qkvg_proj(inputs, qkvg_weights, qk_norm_scales, sinks_k)
	attn_out_pregate, attn_maxes = attn(q, k, v, sink_scores, sinks_v)

	aa = attn_out_pregate.clone()
	g[..., 0::2] = gelu_tanh_approx(g[..., 0::2] * SPARSE_SCALE) * ATTN_GEGLU_SCALE
	g[..., 1::2] = quantize_(g[..., 1::2]) * SPARSE_SCALE
	aa[..., 1::2] = gelu_tanh_approx(aa[..., 1::2]) * ATTN_GEGLU_SCALE

	qkvg = join_qkvg(q, k, v, g)
	attn_out = aa * g
	o_attn = o_proj_attn(attn_out, w_attn)

	f_weights = sparse_weights(f_weights, D_MODEL, HEAD_DIM)
	f_pregate_raw = f_proj_pregate(inputs, f_weights)
	f_pregate = f_pregate_raw * SPARSE_SCALE
	f = pairwise_geglu(f_pregate, F_GEGLU_SCALE)
	f_backvec = pairwise_geglu_backward_multipliers(f_pregate_raw, SPARSE_SCALE, F_GEGLU_SCALE)
	o_ffn = o_proj_ffn(f, w_ffn)
	grad_generator = torch.Generator(device=my_device)
	grad_generator.manual_seed(123)
	d_o_ffn = quantize_(new_randn(N_INPUTS, D_MODEL, generator=grad_generator))
	d_f, d_w_ffn = o_proj_ffn_backward(f, w_ffn, d_o_ffn)

	print("inputs shape:", inputs.shape)
	print("qkvg shape:", qkvg.shape)
	print("q shape:", q.shape)
	print("k shape:", k.shape)
	print("v shape:", v.shape)
	print("sinks_k shape:", sinks_k.shape)
	print("sinks_v shape:", sinks_v.shape)
	print("sink_scores shape:", sink_scores.shape)
	print("attn_maxes shape:", attn_maxes.shape)
	print("attn_out shape:", attn_out.shape)
	print("f_pregate shape:", f_pregate.shape)
	print("f shape:", f.shape)
	print("f_backvec shape:", f_backvec.shape)
	print("d_o_ffn shape:", d_o_ffn.shape)
	print("d_f shape:", d_f.shape)
	print("d_w_ffn shape:", d_w_ffn.shape)
	print("o_attn shape:", o_attn.shape)
	print("o_ffn shape:", o_ffn.shape)
	#print("attn_match shape:", attn_match.shape)

	store_f32_tensor(q, "q_f32.bin", expected_variance=1.0 / HEAD_DIM)
	store_f32_tensor(k, "k_f32.bin", expected_variance=1.0 / HEAD_DIM)
	store_f32_tensor(v, "v_f32.bin", expected_variance=SPARSE_FAN_IN / D_MODEL)
	store_f32_tensor(g, "g_f32.bin")
#	store_f32_tensor(g_bf16 * SPARSE_SCALE, "g_scaled_f32.bin", expected_variance=1.0)

	store_tensor(q, "q.bin", expected_variance=1.0 / HEAD_DIM)
	store_tensor(k, "k.bin", expected_variance=1.0 / HEAD_DIM)
	store_tensor(v, "v.bin", expected_variance=SPARSE_FAN_IN / D_MODEL)
	store_tensor(g, "g.bin")

	store_f32_tensor(sink_scores.transpose(0, 1), "sink_scores_f32.bin", expected_variance=1.0 / HEAD_DIM)
	store_f32_tensor(attn_maxes.transpose(0, 1), "attn_maxes_f32.bin")
	store_tensor(attn_out_pregate, "attn_out_pregate.bin", expected_variance=1.0)
	store_tensor(attn_out, "attn_out.bin", expected_variance=1.0 / ATTN_WIDTH)
	store_tensor(f_pregate, "f_pregate.bin", expected_variance=1.0)
	store_tensor(f, "f.bin", expected_variance=1.0 / F_WIDTH)
	store_tensor(f_backvec, "f_backvec.bin")
	store_tensor(d_o_ffn, "d_o_ffn.bin", expected_variance=1.0)
	store_tensor(d_f, "d_f.bin")
	store_tensor(d_w_ffn, "d_w_ffn.bin")
	store_tensor(o_attn, "o_attn.bin", expected_variance=1.0)
	store_tensor(o_ffn, "o_ffn.bin", expected_variance=1.0)
	store_tensor(qkvg, "qkvg.bin")

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--create-inputs", action="store_true")
	args = parser.parse_args()

	if args.create_inputs:
		create_inputs()

	run_block()

if __name__ == "__main__":
	main()
