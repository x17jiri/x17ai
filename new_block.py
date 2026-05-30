#!/usr/bin/env python3

import argparse
import math

import torch

from block_utils import *

#---------------------------------------------------------------------------------------------------

def quantize(a):
	return torch.clamp(torch.round(a * 8.0), -127.0, +127.0) / 8.0

def new_randn(*shape, generator):
	return torch.randn(shape, generator=generator)

def new_ones(*shape):
	return torch.full(shape, 1.0)

def create_inputs() -> None:
	generator = torch.Generator(device=my_device)
	generator.manual_seed(42)

	x = new_randn(N_INPUTS, MODEL_DIM, generator=generator)

	if 0:
		qk_norm_scales = new_ones(1, HEAD_DIM * N_HEADS)
	else:
		qk_norm_scales = new_randn(1, HEAD_DIM * N_HEADS, generator=generator)

	attn_q_weights = new_randn(N_HEADS*HEAD_DIM, MODEL_DIM, generator=generator)
	attn_kv_weights = new_randn(2*N_HEADS*HEAD_DIM, MODEL_DIM, generator=generator)
	ffn_f_weights = new_randn(2*F_WIDTH, MODEL_DIM, generator=generator)
	ffn_y_weights = new_randn(2*MODEL_DIM, Y_SPARSE_FAN_IN, generator=generator)

	store_tensor(x, "x.bin", expected_variance=1.0)
	store_tensor(x, "x_i8.bin")
	store_tensor(qk_norm_scales, "qk_norm_scales.bin", expected_variance=0.0)
	store_tensor(qk_norm_scales, "qk_norm_scales_i8.bin")
	store_tensor(attn_q_weights, "attn_q_weights.bin", expected_variance=1.0)
	store_tensor(attn_q_weights, "attn_q_weights_i8.bin")
	store_tensor(attn_kv_weights, "attn_kv_weights.bin", expected_variance=1.0)
	store_tensor(attn_kv_weights, "attn_kv_weights_i8.bin")
	store_tensor(ffn_f_weights, "ffn_f_weights.bin", expected_variance=1.0)
	store_tensor(ffn_f_weights, "ffn_f_weights_i8.bin")
	store_tensor(ffn_y_weights, "ffn_y_weights.bin", expected_variance=1.0)
	store_tensor(ffn_y_weights, "ffn_y_weights_i8.bin")

def expand_sparse_weights(w: torch.Tensor, d_inp: int, block_size: int, step: int) -> torch.Tensor:
	d_out = w.shape[0]
	fan_in = w.shape[1]

	assert fan_in <= d_inp
	assert block_size >= 128

	expanded = torch.zeros((d_out, d_inp), dtype=w.dtype, device=w.device)
	cols = torch.arange(fan_in, dtype=torch.int64, device=w.device)
	for row in range(d_out):
		block_idx = row // block_size
		indices = (cols + block_idx * step) % d_inp
		expanded[row, indices] = w[row]
	return expanded

#---------------------------------------------------------------------------------------------------

def l2_norm(tensor: torch.Tensor, eps: float = L2_NORM_EPS) -> torch.Tensor:
	norm = torch.linalg.vector_norm(tensor, ord=2, dim=-1, keepdim=True)
	return tensor / (norm + eps)


def rms_norm(tensor: torch.Tensor, eps: float = L2_NORM_EPS) -> torch.Tensor:
	mean_sq = torch.mean(tensor * tensor, dim=-1, keepdim=True)
	return tensor / torch.sqrt(mean_sq + eps)

def gelu(x: torch.Tensor) -> torch.Tensor:
	ck = math.sqrt(2.0 / math.pi)
	ck3 = 0.044715 * ck
	y = ck3 * x * x * x + ck * x
	return 0.5 * x * torch.tanh(y) + 0.5 * x

def geglu(x: torch.Tensor) -> torch.Tensor:
	gate = x[..., 0::2]
	lin = x[..., 1::2]
	return gelu(gate) * lin

def gated_residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	assert y.shape[-1] == 2 * x.shape[-1]
	gate = torch.sigmoid(y[..., 0::2])
	output = y[..., 1::2]
	return (1.0 - gate) * x + gate * output

#---------------------------------------------------------------------------------------------------

def run_ffn() -> None:
	x = load_tensor("x_i8.bin", N_INPUTS, MODEL_DIM)
	f_weights = load_tensor("ffn_f_weights_i8.bin", 2*F_WIDTH, MODEL_DIM)
	y_weights = load_tensor("ffn_y_weights_i8.bin", 2*MODEL_DIM, Y_SPARSE_FAN_IN)

	x.requires_grad_(True)
	f_weights.requires_grad_(True)
	y_weights.requires_grad_(True)

	y_weights = expand_sparse_weights(y_weights, F_WIDTH, Y_SPARSE_BLOCK, Y_SPARSE_STEP)

	f_pregate = (
		torch.matmul(x, f_weights.transpose(0, 1))
		* torch.rsqrt(torch.tensor(float(MODEL_DIM)))
	)
	f = geglu(f_pregate)

	f = quantize(f)
	y_pregate = (
		torch.matmul(f, y_weights.transpose(0, 1))
		* torch.rsqrt(torch.tensor(float(Y_SPARSE_FAN_IN)))
		* GELU_VAR_FIX
	)
	y = gated_residual(x, y_pregate)

	store_tensor(f_pregate, "ffn_f_pregate.bin", expected_variance=1.0)
	store_tensor(f_pregate, "ffn_f_pregate_i8.bin")
	store_tensor(y_pregate, "ffn_y_pregate.bin", expected_variance=1.0)
	store_tensor(y_pregate, "ffn_y_pregate_i8.bin")
	store_tensor(f, "ffn_f.bin", expected_variance=1.0 / GELU_VAR_FIX_2)
	store_tensor(f, "ffn_f_i8.bin")
	store_tensor(y, "ffn_y.bin")
	store_tensor(y, "ffn_y_i8.bin")

#---------------------------------------------------------------------------------------------------

def run_attn() -> None:
	x = load_tensor("x_i8.bin", N_INPUTS, MODEL_DIM)
	qk_norm_scales = load_tensor("qk_norm_scales.bin", 1, HEAD_DIM * N_HEADS)
	q_weights = load_tensor("attn_q_weights_i8.bin", N_HEADS*HEAD_DIM, MODEL_DIM)
	kv_weights = load_tensor("attn_kv_weights_i8.bin", 2*N_HEADS*HEAD_DIM, MODEL_DIM)

	scale = torch.rsqrt(torch.tensor(float(MODEL_DIM)))
	q = torch.matmul(x, q_weights.transpose(0, 1)) * scale
	kv = torch.matmul(x, kv_weights.transpose(0, 1)) * scale

	q = q.reshape(N_INPUTS, N_HEADS, HEAD_DIM)
	kv = kv.reshape(N_INPUTS, N_HEADS, 2*HEAD_DIM)
	k = kv[..., :HEAD_DIM]
	v = kv[..., HEAD_DIM:]

	q = rms_norm(q)
	q = q * qk_norm_scales.reshape(1, N_HEADS, HEAD_DIM)

	k = rms_norm(k)
	kv = torch.cat((k, v), dim=2)

	store_tensor(q, "q.bin", expected_variance=1.0)
	store_tensor(q, "q_i8.bin")
	store_tensor(q, "k.bin", expected_variance=1.0)
	store_tensor(q, "k_i8.bin")
	store_tensor(q, "v.bin", expected_variance=1.0)
	store_tensor(q, "v_i8.bin")
	store_tensor(kv, "kv.bin", expected_variance=1.0)
	store_tensor(kv, "kv_i8.bin")

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
