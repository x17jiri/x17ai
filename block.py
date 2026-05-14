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

#---------------------------------------------------------------------------------------------------

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
	grad_generator = torch.Generator(device=my_device)
	grad_generator.manual_seed(123)

	# randn init
	inputs = new_randn(N_INPUTS, D_MODEL, generator=generator)
	qkvg_weights = new_randn(QKVG_ROWS, SPARSE_FAN_IN, generator=generator)
	f_weights = new_randn(F_PROJ_OUTPUTS, SPARSE_FAN_IN, generator=generator)
	w_attn = new_randn(D_MODEL, ATTN_WIDTH, generator=generator)
	w_ffn = new_randn(D_MODEL, F_WIDTH, generator=generator)
	sink_k = new_randn(N_HEADS, HEAD_DIM, generator=generator)
	sinks_v = new_randn(N_HEADS, HEAD_DIM, generator=generator)
	d_out = quantize_(new_randn(N_INPUTS, D_MODEL, generator=grad_generator))

	# constant value init
	qk_norm_scales = new_ones(1, HEAD_DIM * N_HEADS)

	inputs_l2 = l2_norm(inputs)
	sink_k = l2_norm(sink_k)

	store_tensor(inputs, "inputs.bin", expected_variance=1.0)
	store_tensor(inputs_l2, "inputs_l2.bin", expected_variance=1.0 / D_MODEL)
	store_tensor(qkvg_weights, "qkvg_weights.bin", expected_variance=1.0)
	store_tensor(f_weights, "ffn_f_weights.bin", expected_variance=1.0)
	store_tensor(w_attn, "w_attn.bin", expected_variance=1.0)
	store_tensor(w_ffn, "ffn_y_weights.bin", expected_variance=1.0)
	store_tensor(w_ffn, "ffn_y_weights_f8.bin", expected_variance=1.0)
	store_tensor(d_out, "d_out.bin", expected_variance=1.0)
	store_tensor(qk_norm_scales, "qk_norm_scales.bin", expected_variance=0.0)
	store_tensor(sink_k, "sinks_k.bin", expected_variance=1.0 / HEAD_DIM)
	store_tensor(sinks_v, "sinks_v.bin", expected_variance=1.0)

def sparse_weights(w: torch.Tensor, d_input, repeat_after) -> torch.Tensor:
	d_output = w.shape[0]
	fan_in = w.shape[1]
	step = d_input // repeat_after
	sparse = torch.zeros((d_output, d_input), dtype=w.dtype, device=w.device)
	cols = torch.arange(fan_in, dtype=torch.int64, device=w.device)
	for row in range(d_output):
		indices = (cols + row * step) % d_input
		sparse[row, indices] = w[row]
	return sparse

def gelu(x: torch.Tensor) -> torch.Tensor:
	ck = math.sqrt(2.0 / math.pi)
	ck3 = 0.044715 * ck
	y = ck3 * x * x * x + ck * x
	return 0.5 * x * torch.tanh(y) + 0.5 * x

#---------------------------------------------------------------------------------------------------

def ffn_f_fwd(inputs: torch.Tensor, f_weights: torch.Tensor) -> torch.Tensor:
	assert(f_weights.shape[0] == 2*F_WIDTH)
	inputs = quantize_(inputs)
	f_weights = quantize_(f_weights)
	t = torch.matmul(inputs, f_weights.transpose(0, 1))
	gate = t[..., 0::2]
	lin = t[..., 1::2]
	OUT_SCALE = GELU_VAR_FIX * math.sqrt(1.0 / F_WIDTH)

	gate = gate * SPARSE_SCALE
	lin = lin * SPARSE_SCALE
	warn_if_variance_is_unexpected("ffn_f_fwd: gate * SPARSE_SCALE", gate, 1.0)
	warn_if_variance_is_unexpected("ffn_f_fwd: lin * SPARSE_SCALE", lin, 1.0)

	return gelu(gate) * lin * OUT_SCALE

def ffn_y_fwd(f: torch.Tensor, w_ffn: torch.Tensor) -> torch.Tensor:
	f = quantize_(f)
	w_ffn = quantize_(w_ffn)
	return torch.matmul(f, w_ffn.transpose(0, 1))

#---------------------------------------------------------------------------------------------------

def qkvg_proj(
	inputs: torch.Tensor,
	qkvg_weights: torch.Tensor,
	qk_norm_scales: torch.Tensor,
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

	g[..., 0::2] = (
		gelu(g[..., 0::2] * SPARSE_SCALE) * GELU_VAR_FIX
		* SPARSE_SCALE * V_SCALE_FIX
		* math.sqrt(1.0 / ATTN_WIDTH)
	)

	return q, k, v, g

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

#---------------------------------------------------------------------------------------------------

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
	g: torch.Tensor,
	sink_k: torch.Tensor,
	sink_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	seq_len = q.shape[0]
	QK_DIM = q.shape[1]
	q = quantize_(q)
	k = quantize_(k)
	v = quantize_(v)
	g = quantize_(g)

	sink_k = quantize_(sink_k.unsqueeze(0))
	sink_v = quantize_(sink_v.unsqueeze(0)) / SPARSE_SCALE
	k = torch.cat((sink_k, k), dim=0)
	v = torch.cat((sink_v, v), dim=0)

	S = q @ k.transpose(0, 1)

	BASE_TEMPERATURE = math.sqrt(QK_DIM)
	temperature = BASE_TEMPERATURE * torch.log(ssmax_n(seq_len, WINDOW_SIZE))

	real_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
	if WINDOW_SIZE > 0:
		real_mask = real_mask | torch.tril(
			torch.ones(seq_len, seq_len, dtype=torch.bool),
			diagonal = -(WINDOW_SIZE + 1),
		)
	mask = torch.zeros(seq_len, seq_len + 1, dtype=torch.bool)
	mask[:, 1:] = real_mask
	S = S.masked_fill(mask, float("-inf"))

	S *= temperature
	max = S.amax(dim=1, keepdim=True)
	S -= max
	P = torch.exp(S)
	sum = P.sum(dim=1, keepdim=True)

	P = torch.cat((P[:, :1], quantize_(P[:, 1:])), dim=1)

	o = P @ v

	sum_recip = torch.reciprocal(sum)
	o *= sum_recip

	# zig-zag geglu
	even_out = g[..., 0::2] * o[..., 0::2]
	odd_out = (
		gelu(o[..., 1::2] * SPARSE_SCALE * V_SCALE_FIX)
		* GELU_VAR_FIX
		* SPARSE_SCALE
		* math.sqrt(1.0 / ATTN_WIDTH)
		* g[..., 1::2]
	)
	out = torch.stack((even_out, odd_out), dim=-1).reshape_as(o)

	LOG2_E = 1.0 / math.log(2.0)
	return out, max.squeeze(1) * LOG2_E

def attn(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	g: torch.Tensor,
	sinks_k: torch.Tensor,
	sinks_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	_, n_heads, _ = q.shape
	head_results = [
		attn_one_head(
			q[:, h, :],
			k[:, h, :],
			v[:, h, :],
			g[:, h, :],
			sinks_k[h, :],
			sinks_v[h, :],
		)
		for h in range(n_heads)
	]
	out = torch.stack([head_out for head_out, _ in head_results], dim=1)
	max_values = torch.stack([head_max for _, head_max in head_results], dim=1)
	return out, max_values

#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------

def run_ffn() -> None:
	x = load_tensor("inputs_l2.bin", N_INPUTS, D_MODEL)
	f_weights = load_tensor("ffn_f_weights.bin", F_PROJ_OUTPUTS, SPARSE_FAN_IN)
	y_weights = load_tensor("ffn_y_weights.bin", D_MODEL, F_WIDTH)
	d_y = load_tensor("d_out.bin", N_INPUTS, D_MODEL)

	x.requires_grad_(True)
	f_weights.requires_grad_(True)
	y_weights.requires_grad_(True)

	f_full_weights = sparse_weights(f_weights, D_MODEL, HEAD_DIM)
	t = torch.matmul(quantize_(x.detach()), quantize_(f_full_weights.detach()).transpose(0, 1))
	gate_scaled = (t[..., 0::2] * SPARSE_SCALE).detach().requires_grad_(True)
	lin_scaled = t[..., 1::2] * SPARSE_SCALE
	out_scale = GELU_VAR_FIX * math.sqrt(1.0 / F_WIDTH)
	local_f = gelu(gate_scaled) * lin_scaled * out_scale
	torch.autograd.backward(local_f, torch.ones_like(local_f))
	assert gate_scaled.grad is not None
	backvec = torch.empty_like(t)
	backvec[..., 0::2] = gate_scaled.grad.detach() * SPARSE_SCALE
	backvec[..., 1::2] = gelu(gate_scaled.detach()) * (SPARSE_SCALE * out_scale)

	f = ffn_f_fwd(x, f_full_weights)
	f.retain_grad()

	y = ffn_y_fwd(f, y_weights)

	torch.autograd.backward(y, d_y)
	assert f.grad is not None
	assert y_weights.grad is not None
	assert f_weights.grad is not None
	assert x.grad is not None

	store_tensor(f, "ffn_f.bin", expected_variance=1.0 / F_WIDTH)
	store_tensor(backvec, "ffn_f_backvec.bin")
	store_tensor(y, "ffn_y.bin", expected_variance=1.0)
	store_tensor(f.grad, "ffn_d_f.bin")
	store_tensor(y_weights.grad, "ffn_d_y_weights.bin")
	store_tensor(f_weights.grad, "ffn_d_f_weights.bin")
	store_tensor(x.grad, "ffn_d_x.bin")

#---------------------------------------------------------------------------------------------------

def run_block() -> None:

	run_ffn()
	return

	#-- Params

	inputs = load_tensor("inputs_l2.bin", N_INPUTS, D_MODEL)
	qkvg_weights = load_tensor("qkvg_weights.bin", QKVG_ROWS, SPARSE_FAN_IN)
	f_weights = load_tensor("ffn_f_weights.bin", F_PROJ_OUTPUTS, SPARSE_FAN_IN)
	w_attn = load_tensor("w_attn.bin", D_MODEL, ATTN_WIDTH)
	w_ffn = load_tensor("ffn_y_weights.bin", D_MODEL, F_WIDTH)
	d_out = load_tensor("d_out.bin", N_INPUTS, D_MODEL)
	qk_norm_scales = load_tensor("qk_norm_scales.bin", 1, HEAD_DIM * N_HEADS)
	sinks_k = load_tensor("sinks_k.bin", N_HEADS, HEAD_DIM)
	sinks_v = load_tensor("sinks_v.bin", N_HEADS, HEAD_DIM)

	#-- FFN

	ffn_inputs = inputs.detach().clone().requires_grad_(True)
	compact_f_weights = f_weights.detach().clone().requires_grad_(True)
	ffn_w_ffn = w_ffn.detach().clone().requires_grad_(True)

	dense_f_weights = sparse_weights(compact_f_weights, D_MODEL, HEAD_DIM)
	f = ffn_f_fwd(ffn_inputs, dense_f_weights)
	f.retain_grad()
	o_ffn = ffn_y_fwd(f, ffn_w_ffn)

	torch.autograd.backward(o_ffn, d_out)
	assert f.grad is not None
	assert ffn_inputs.grad is not None
	assert ffn_w_ffn.grad is not None
	assert compact_f_weights.grad is not None

	d_f = f.grad.detach()
	d_inputs_l2_ffn = ffn_inputs.grad.detach()
	d_w_ffn = ffn_w_ffn.grad.detach()
	d_f_weights = compact_f_weights.grad.detach()
	f = f.detach()
	o_ffn = o_ffn.detach()

	if True:
		store_tensor(f, "f.bin", expected_variance=1.0 / F_WIDTH)
		store_tensor(o_ffn, "o_ffn.bin", expected_variance=1.0)
		store_tensor(d_out, "d_o_ffn.bin", expected_variance=1.0)
		store_tensor(d_f, "d_f.bin")
		store_tensor(d_inputs_l2_ffn, "d_inputs_l2_ffn.bin")
		store_tensor(d_w_ffn, "d_w_ffn.bin")
		store_tensor(d_f_weights, "d_f_weights.bin")

	#-- Attn

	qkvg_weights = sparse_weights(qkvg_weights, D_MODEL, HEAD_DIM);
	q, k, v, g = qkvg_proj(inputs, qkvg_weights, qk_norm_scales)
	qkvg = join_qkvg(q, k, v, g)
	attn_out, attn_maxes = attn(q, k, v, g, sinks_k, sinks_v)
	o_attn = o_proj_attn(attn_out, w_attn)

	if True:
		store_f32_tensor(q, "q_f32.bin", expected_variance=1.0 / HEAD_DIM)
		store_f32_tensor(k, "k_f32.bin", expected_variance=1.0 / HEAD_DIM)
		store_f32_tensor(v, "v_f32.bin", expected_variance=SPARSE_FAN_IN / D_MODEL)
		store_f32_tensor(g, "g_f32.bin")

		store_tensor(q, "q.bin", expected_variance=1.0 / HEAD_DIM)
		store_tensor(k, "k.bin", expected_variance=1.0 / HEAD_DIM)
		store_tensor(v, "v.bin", expected_variance=SPARSE_FAN_IN / D_MODEL)
		store_tensor(g, "g.bin")

		store_tensor(qkvg, "qkvg.bin")

		store_f32_tensor(attn_maxes.transpose(0, 1), "attn_maxes_f32.bin")
		store_tensor(attn_out, "attn_out.bin", expected_variance=1.0 / ATTN_WIDTH)

		store_tensor(o_attn, "o_attn.bin", expected_variance=1.0)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--create-inputs", action="store_true")
	args = parser.parse_args()

	if args.create_inputs:
		create_inputs()

	run_block()

if __name__ == "__main__":
	main()
