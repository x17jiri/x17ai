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

	x = new_randn(N_INPUTS, D_MODEL, generator=generator)
	attn_q_weights = new_randn(N_HEADS*HEAD_DIM, SPARSE_FAN_IN, generator=generator)
	attn_kv_weights = new_randn(2*N_HEADS*HEAD_DIM, SPARSE_FAN_IN, generator=generator)
	attn_g_weights = new_randn(N_HEADS*HEAD_DIM, SPARSE_FAN_IN, generator=generator)
	ffn_f_weights = new_randn(2*F_WIDTH, SPARSE_FAN_IN, generator=generator)
	ffn_y_weights = new_randn(D_MODEL, F_WIDTH, generator=generator)

	store_tensor(x, "x.bin", expected_variance=1.0)
	store_tensor(x, "x_i8.bin")
	store_tensor(attn_q_weights, "attn_q_weights.bin", expected_variance=1.0)
	store_tensor(attn_q_weights, "attn_q_weights_i8.bin")
	store_tensor(attn_kv_weights, "attn_kv_weights.bin", expected_variance=1.0)
	store_tensor(attn_kv_weights, "attn_kv_weights_i8.bin")
	store_tensor(attn_g_weights, "attn_g_weights.bin", expected_variance=1.0)
	store_tensor(attn_g_weights, "attn_g_weights_i8.bin")
	store_tensor(ffn_f_weights, "ffn_f_weights.bin", expected_variance=1.0)
	store_tensor(ffn_f_weights, "ffn_f_weights_i8.bin")
	store_tensor(ffn_y_weights, "ffn_y_weights.bin", expected_variance=1.0)
	store_tensor(ffn_y_weights, "ffn_y_weights_i8.bin")

def expand_weights(w: torch.Tensor, D_INP) -> torch.Tensor:
	D_OUT = w.shape[0]
	FAN_IN = w.shape[1]

	assert FAN_IN < D_INP
	STEP = FAN_IN // 2
	STEPS = D_INP // STEP
	CNT_PER_STEP = D_OUT // STEPS

	expanded = torch.zeros((D_OUT, D_INP), dtype=w.dtype, device=w.device)
	cols = torch.arange(FAN_IN, dtype=torch.int64, device=w.device)
	for row in range(D_OUT):
		indices = (cols + (row // CNT_PER_STEP) * STEP) % D_INP
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

#---------------------------------------------------------------------------------------------------

def run_ffn() -> None:
	x = load_tensor("x_i8.bin", N_INPUTS, D_MODEL)
	f_weights = load_tensor("ffn_f_weights_i8.bin", 2*F_WIDTH, SPARSE_FAN_IN)
	y_weights = load_tensor("ffn_y_weights_i8.bin", D_MODEL, F_WIDTH)

	x.requires_grad_(True)
	f_weights.requires_grad_(True)
	y_weights.requires_grad_(True)

	f_weights = expand_weights(f_weights, D_MODEL)

	f_pregate = torch.matmul(x, f_weights.transpose(0, 1)) * torch.rsqrt(torch.tensor(SPARSE_FAN_IN))
	f_gate = f_pregate[..., 0::2]
	f_lin = f_pregate[..., 1::2]
	f = gelu(f_gate) * f_lin

	f = quantize(f)
	y = torch.matmul(f, y_weights.transpose(0, 1)) * torch.rsqrt(torch.tensor(F_WIDTH)) * GELU_VAR_FIX

	store_tensor(f_pregate, "ffn_f_pregate.bin", expected_variance=1.0)
	store_tensor(f_pregate, "ffn_f_pregate_i8.bin")
	store_tensor(f, "ffn_f.bin", expected_variance=1.0 / GELU_VAR_FIX_2)
	store_tensor(f, "ffn_f_i8.bin")
	store_tensor(y, "ffn_y.bin", expected_variance=1.0)
	store_tensor(y, "ffn_y_i8.bin")

#---------------------------------------------------------------------------------------------------

def run_attn() -> None:
	x = load_tensor("x_i8.bin", N_INPUTS, D_MODEL)
	q_weights = load_tensor("attn_q_weights_i8.bin", N_HEADS*HEAD_DIM, SPARSE_FAN_IN)
	kv_weights = load_tensor("attn_kv_weights_i8.bin", 2*N_HEADS*HEAD_DIM, SPARSE_FAN_IN)
	g_weights = load_tensor("attn_g_weights_i8.bin", N_HEADS*HEAD_DIM, SPARSE_FAN_IN)

	q_weights = expand_weights(q_weights, D_MODEL)
	kv_weights = expand_weights(kv_weights, D_MODEL)
	g_weights = expand_weights(g_weights, D_MODEL)
	scale = torch.rsqrt(torch.tensor(float(SPARSE_FAN_IN)))

	q = torch.matmul(x, q_weights.transpose(0, 1)) * scale
	kv = torch.matmul(x, kv_weights.transpose(0, 1)) * scale
	g = torch.matmul(x, g_weights.transpose(0, 1)) * scale

	q = q.reshape(N_INPUTS, N_HEADS, HEAD_DIM)
	kv = kv.reshape(N_INPUTS, N_HEADS, 2*HEAD_DIM)
	k = kv[..., :HEAD_DIM]
	v = kv[..., HEAD_DIM:]
	g = g.reshape(N_INPUTS, N_HEADS, VG_DIM)

	q = rms_norm(q)
	k = rms_norm(k)

	store_tensor(q, "q.bin", expected_variance=1.0)
	store_tensor(q, "q_i8.bin")
	store_tensor(kv, "kv.bin", expected_variance=1.0)
	store_tensor(kv, "kv_i8.bin")
	store_tensor(g, "g.bin", expected_variance=1.0)
	store_tensor(g, "g_i8.bin")

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
