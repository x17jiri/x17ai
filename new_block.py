#!/usr/bin/env python3

import argparse
import math

import torch

from block_utils import *

#---------------------------------------------------------------------------------------------------

def new_randn(*shape, generator):
	return torch.randn(shape, generator=generator)

def new_ones(*shape):
	return torch.full(shape, 1.0)

def create_inputs() -> None:
	generator = torch.Generator(device=my_device)
	generator.manual_seed(42)

	x = new_randn(N_INPUTS, D_MODEL, generator=generator)
	ffn_f_weights = new_randn(2*F_WIDTH, SPARSE_FAN_IN, generator=generator)
	ffn_y_weights = new_randn(D_MODEL, F_WIDTH, generator=generator)

	store_tensor(x, "x.bin", expected_variance=1.0)
	store_tensor(x, "x_i8.bin")
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

def ffn_y_fwd(f: torch.Tensor, ffn_y_weights: torch.Tensor, fan_in: int | None = None) -> torch.Tensor:
	if fan_in is None:
		fan_in = ffn_y_weights.shape[1]
	SCALE = torch.rsqrt(torch.tensor(float(fan_in)))
	return torch.matmul(f, ffn_y_weights.transpose(0, 1)) * SCALE

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

	y = torch.matmul(f, y_weights.transpose(0, 1)) * torch.rsqrt(torch.tensor(F_WIDTH)) * GELU_VAR_FIX

	store_tensor(f_pregate, "ffn_f_pregate.bin", expected_variance=1.0)
	store_tensor(f_pregate, "ffn_f_pregate_i8.bin")
	store_tensor(f, "ffn_f.bin", expected_variance=1.0 / GELU_VAR_FIX_2)
	store_tensor(f, "ffn_f_i8.bin")
	store_tensor(y, "ffn_y.bin", expected_variance=1.0)
	store_tensor(y, "ffn_y_i8.bin")

#---------------------------------------------------------------------------------------------------

def run_block() -> None:
	run_ffn()

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--create-inputs", action="store_true")
	args = parser.parse_args()

	if args.create_inputs:
		create_inputs()

	run_block()

if __name__ == "__main__":
	main()
