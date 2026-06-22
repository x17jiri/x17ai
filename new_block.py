#!/usr/bin/env python3

import argparse
import math

import torch

from block_utils import *

#---------------------------------------------------------------------------------------------------

def quantize(a):
	return torch.clamp(torch.round(a * 8.0), -127.0, +127.0) / 8.0

def quantize_i8_codes(a: torch.Tensor) -> torch.Tensor:
	return torch.clamp(torch.round(a * 8.0), -127.0, +127.0).to(torch.int32)

def quantize_e4m3(a):
	return e4m3_ftz(a).to(F8_DTYPE).to(my_dtype)

def new_randn(*shape, generator):
	return torch.randn(shape, generator=generator)

def new_ones(*shape):
	return torch.full(shape, 1.0)

def create_inputs() -> None:
	generator = torch.Generator(device=my_device)
	generator.manual_seed(42)

	x = new_randn(N_INPUTS, MODEL_DIM, generator=generator)

	sinks_k, _ = rms_norm(new_randn(N_HEADS, HEAD_DIM, generator=generator))
	sinks_v = new_randn(N_HEADS, HEAD_DIM, generator=generator)

	attn_temperature = new_ones(N_HEADS, 1)

	TEST = True
	if TEST:
		attn_temperature = 0.5 + torch.rand((N_HEADS, 1), generator=generator)

	attn_q_weights = new_randn(N_HEADS*HEAD_DIM, MODEL_DIM, generator=generator)
	attn_kv_weights = new_randn(N_HEADS, 2, HEAD_DIM, MODEL_DIM, generator=generator)
	attn_y_weights = new_randn(2*MODEL_DIM, ATTN_WIDTH, generator=generator)

	attn_k_weights = attn_kv_weights[:, 0, :, :]
	attn_v_weights = attn_kv_weights[:, 1, :, :]
	attn_v_weights *= V_SCALE_FIX
	sinks_v *= V_SCALE_FIX

	ffn_f_weights = new_randn(2*F_WIDTH, MODEL_DIM, generator=generator)
	ffn_y_weights = new_randn(2*MODEL_DIM, F_WIDTH, generator=generator)

	store_tensor(x, "x_f32.bin", expected_variance=1.0)
	store_tensor(x, "x_i8.bin")
	store_tensor(x, "x_i8.safetensors")
	store_tensor(sinks_k, "sinks_k_f32.bin", expected_variance=1.0)
	store_tensor(sinks_k, "sinks_k_i8.bin")
	store_tensor(sinks_k, "sinks_k_i8.safetensors")
	store_tensor(sinks_v, "sinks_v_f32.bin", expected_variance=V_SCALE_FIX*V_SCALE_FIX)
	store_tensor(sinks_v, "sinks_v_i8.bin")
	store_tensor(sinks_v, "sinks_v_i8.safetensors")
	store_tensor(attn_temperature, "attn_temperature_f32.bin", expected_variance=0.0)
	store_tensor(attn_temperature, "attn_temperature_f32.safetensors")
	store_tensor(attn_q_weights, "attn_q_weights_f32.bin", expected_variance=1.0)
	store_tensor(attn_q_weights, "attn_q_weights_i8.bin")
	store_tensor(attn_q_weights, "attn_q_weights_i8.safetensors")
	store_tensor(attn_k_weights, "attn_k_weights_f32.bin", expected_variance=1.0)
	store_tensor(attn_v_weights, "attn_v_weights_f32.bin", expected_variance=V_SCALE_FIX*V_SCALE_FIX)
	store_tensor(attn_kv_weights, "attn_kv_weights_f32.bin", expected_variance=1.0)
	store_tensor(attn_kv_weights, "attn_kv_weights_i8.bin")
	store_tensor(attn_kv_weights, "attn_kv_weights_i8.safetensors")
	store_tensor(attn_y_weights, "attn_y_weights_f32.bin", expected_variance=1.0)
	store_tensor(attn_y_weights, "attn_y_weights_i8.bin")
	store_tensor(attn_y_weights, "attn_y_weights_i8.safetensors")
	store_tensor(ffn_f_weights, "ffn_f_weights_f32.bin", expected_variance=1.0)
	store_tensor(ffn_f_weights, "ffn_f_weights_i8.bin")
	store_tensor(ffn_f_weights, "ffn_f_weights_i8.safetensors")
	store_tensor(ffn_y_weights, "ffn_y_weights_f32.bin", expected_variance=1.0)
	store_tensor(ffn_y_weights, "ffn_y_weights_i8.bin")

def rms_norm(tensor: torch.Tensor, eps: float = L2_NORM_EPS) -> tuple[torch.Tensor, torch.Tensor]:
	rrms = torch.rsqrt(torch.mean(tensor * tensor, dim=-1, keepdim=True) + eps)
	return tensor * rrms, rrms

def gelu(x: torch.Tensor) -> torch.Tensor:
	ck = math.sqrt(2.0 / math.pi)
	ck3 = 0.044715 * ck
	y = ck3 * x * x * x + ck * x
	return 0.5 * x * torch.tanh(y) + 0.5 * x

def geglu(x: torch.Tensor) -> torch.Tensor:
	gate = x[..., 0::2]
	lin = x[..., 1::2]
	return gelu(gate) * lin

def exp4(x):
	return torch.exp2(2.0 * x)

def log4(x):
	return torch.log2(x) / 2.0

def sigmoid_base4(x: torch.Tensor) -> torch.Tensor:
	return torch.reciprocal(1.0 + exp4(-x))

def softplus_base4(x: torch.Tensor) -> torch.Tensor:
	return log4(1.0 + exp4(x))

def gated_residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	assert y.shape[-1] == 2 * x.shape[-1]
	gate = y[..., 0::2]
	old_weight = sigmoid_base4(-gate)
	new_weight = softplus_base4(gate)
	output = y[..., 1::2]
	return old_weight * x + new_weight * output

#---------------------------------------------------------------------------------------------------

def run_ffn() -> None:
	x = load_tensor("x_i8.bin", N_INPUTS, MODEL_DIM)
	f_weights = load_tensor("ffn_f_weights_i8.bin", 2*F_WIDTH, MODEL_DIM)
	y_weights = load_tensor("ffn_y_weights_i8.bin", 2*MODEL_DIM, F_WIDTH)

	x.requires_grad_(True)
	f_weights.requires_grad_(True)
	y_weights.requires_grad_(True)

	f_pregate = (
		torch.matmul(x, f_weights.transpose(0, 1))
		* torch.rsqrt(torch.tensor(float(MODEL_DIM)))
	)
	f = geglu(f_pregate)

	f_f8 = quantize_e4m3(f)
	y_pregate = (
		torch.matmul(f_f8, y_weights.transpose(0, 1))
		* torch.rsqrt(torch.tensor(float(F_WIDTH)))
		* GELU_VAR_FIX
	)
	y = gated_residual(x, y_pregate)

	store_tensor(f_pregate, "ffn_f_pregate_bf16.bin", expected_variance=1.0)
	store_tensor(f_pregate, "ffn_f_pregate_i8.bin")
	store_tensor(y_pregate, "ffn_y_pregate_bf16.bin", expected_variance=1.0)
	store_tensor(y_pregate, "ffn_y_pregate_i8.bin")
	store_tensor(f, "ffn_f_bf16.bin", expected_variance=1.0 / GELU_VAR_FIX_2)
	store_tensor(f_f8, "ffn_f_f8.bin")
	store_tensor(f_f8, "ffn_f_f8.safetensors")
	store_tensor(y, "ffn_y_bf16.bin")
	store_tensor(y, "ffn_y_i8.bin")

#---------------------------------------------------------------------------------------------------

def ssmax_n(seq_len, window_size=0):
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
	visible_real_tokens = torch.arange(seq_len, dtype=my_dtype)
	if window_size > 0:
		n = torch.clamp(visible_real_tokens, max=float(window_size)) + E_APPROX_PLUS_1
	else:
		n = visible_real_tokens + E_APPROX_PLUS_1
	return n.unsqueeze(1)

def attn_one_head(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sink_k: torch.Tensor,
	sink_v: torch.Tensor,
	temperature_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	seq_len = q.shape[0]
	QK_DIM = q.shape[1]

	k = torch.cat((sink_k.unsqueeze(0), k), dim=0)
	v = torch.cat((sink_v.unsqueeze(0), v), dim=0)
	q_i32 = quantize_i8_codes(q)
	k_i32 = quantize_i8_codes(k)
	bare_scores_i32 = torch.matmul(q_i32, k_i32.transpose(0, 1))

	S = q @ k.transpose(0, 1)

	BASE_TEMPERATURE = math.sqrt(1.0 / QK_DIM)
	temperature = BASE_TEMPERATURE * torch.log(ssmax_n(seq_len, WINDOW_SIZE)) * temperature_scale

	real_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
	if WINDOW_SIZE > 0:
		real_mask = real_mask | torch.tril(
			torch.ones(seq_len, seq_len, dtype=torch.bool),
			diagonal = -(WINDOW_SIZE + 1),
		)
	mask = torch.zeros(seq_len, seq_len + 1, dtype=torch.bool)
	mask[:, 1:] = real_mask
	bare_scores_i32 = bare_scores_i32.masked_fill(mask, torch.iinfo(torch.int32).min)
	bare_max = bare_scores_i32.amax(dim=1)
	S = S.masked_fill(mask, float("-inf"))

	S *= temperature
	max = S.amax(dim=1, keepdim=True)
	S -= max
	P = torch.exp(S)
	sum = P.sum(dim=1, keepdim=True)

	P = torch.cat((P[:, :1], quantize(P[:, 1:])), dim=1)

	o = P @ v

	sum_recip = torch.reciprocal(sum)
	o *= sum_recip

	LOG2_E = 1.0 / math.log(2.0)
	return o, max.squeeze(1) * LOG2_E, bare_max

def attn(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	sinks_k: torch.Tensor,
	sinks_v: torch.Tensor,
	attn_temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	_, n_heads, _ = q.shape
	head_results = [
		attn_one_head(
			q[:, h, :],
			k[:, h, :],
			v[:, h, :],
			sinks_k[h, :],
			sinks_v[h, :],
			attn_temperature[h, 0],
		)
		for h in range(n_heads)
	]
	out = torch.stack([head_out for head_out, _, _ in head_results], dim=1)
	max_values = torch.stack([head_max for _, head_max, _ in head_results], dim=1)
	bare_max_values = torch.stack([head_max_i32 for _, _, head_max_i32 in head_results], dim=1)
	return out, max_values, bare_max_values

def run_attn() -> None:
	x = load_tensor("x_i8.bin", N_INPUTS, MODEL_DIM)
	sinks_k = load_tensor("sinks_k_i8.bin", N_HEADS, HEAD_DIM)
	sinks_v = load_tensor("sinks_v_i8.bin", N_HEADS, HEAD_DIM)
	attn_temperature = load_tensor("attn_temperature_f32.bin", N_HEADS, 1)
	q_weights = load_tensor("attn_q_weights_i8.bin", N_HEADS*HEAD_DIM, MODEL_DIM)
	kv_weights = load_tensor("attn_kv_weights_i8.bin", 2*N_HEADS*HEAD_DIM, MODEL_DIM)
	y_weights = load_tensor("attn_y_weights_i8.bin", 2*MODEL_DIM, ATTN_WIDTH)

	scale = torch.rsqrt(torch.tensor(float(MODEL_DIM)))
	q = torch.matmul(x, q_weights.transpose(0, 1)) * scale
	kv = torch.matmul(x, kv_weights.transpose(0, 1))

	q = q.reshape(N_INPUTS, N_HEADS, HEAD_DIM)

	kv = kv.reshape(N_INPUTS, N_HEADS, 2*HEAD_DIM)
	k, k_rrms = rms_norm(kv[..., :HEAD_DIM])
	k_rrms /= 8.0 # The CUDA output is actually scaled by `FIXED_I8_SCALE`. Need to keep that in mind
	v = kv[..., HEAD_DIM:] * scale
	kv = torch.cat((k, v), dim=2)

	q_i8 = quantize(q)
	k_i8 = quantize(k)
	v_i8 = quantize(v)
	kv_i8 = torch.cat((k_i8, v_i8), dim=2)

	RUN_ATTN = True
	if RUN_ATTN:
		attn_out, attn_maxes, attn_maxes_i32 = attn(q_i8, k_i8, v_i8, sinks_k, sinks_v, attn_temperature)
		attn_out_flat = attn_out.reshape(N_INPUTS, ATTN_WIDTH)
		attn_out_i8 = quantize(attn_out_flat)
		attn_y_pregate = (
			torch.matmul(attn_out_i8, y_weights.transpose(0, 1))
			* torch.rsqrt(torch.tensor(float(ATTN_WIDTH)))
		)
		attn_y = gated_residual(x, attn_y_pregate)

	store_tensor(q, "q_f32.bin", expected_variance=1.0)
	store_tensor(q_i8, "q_i8.bin")
	store_tensor(q_i8, "q_i8.safetensors")
	store_tensor(k, "k_f32.bin", expected_variance=1.0)
	store_tensor(k_i8, "k_i8.bin")
	store_tensor(k_rrms.squeeze(-1), "k_rrms_f32.bin")
	store_tensor(k_rrms.squeeze(-1), "k_rrms_f32.safetensors")
	store_tensor(v, "v_f32.bin", expected_variance=1.0)
	store_tensor(v_i8, "v_i8.bin")
	store_tensor(kv, "kv_f32.bin", expected_variance=1.0)
	store_tensor(kv_i8, "kv_i8.bin")
	store_tensor(kv_i8, "kv_i8.safetensors")
	if RUN_ATTN:
		store_tensor(attn_maxes.transpose(0, 1), "attn_maxes_f32.bin")
		store_i32_tensor(attn_maxes_i32.transpose(0, 1), "attn_maxes_i32.bin")
		store_tensor(attn_out, "attn_out_f32.bin", expected_variance=1.0)
		store_tensor(attn_out_i8, "attn_out_i8.bin")
		store_tensor(attn_y_pregate, "attn_y_pregate_f32.bin", expected_variance=1.0)
		store_tensor(attn_y_pregate, "attn_y_pregate_i8.bin")
		store_tensor(attn_y, "attn_y_f32.bin")
		store_tensor(attn_y, "attn_y_i8.bin")
		store_tensor(attn_y, "attn_y_i8.safetensors")

#---------------------------------------------------------------------------------------------------

def run_block() -> None:
	run_ffn()
	run_attn()

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--create-inputs", action="store_true")
	args = parser.parse_args()

	if args.create_inputs:
		create_inputs()

	run_block()

if __name__ == "__main__":
	main()
