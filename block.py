#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "block.json"
TENSOR_DIR = ROOT / "tmp" / "block"


def get_device() -> torch.device:
	return torch.device("cpu")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for block.py")
	return torch.device("cuda")


def load_config() -> dict:
	with CONFIG_PATH.open("r", encoding="ascii") as config_file:
		config_text = config_file.read()
	config_text = re.sub(r",(\s*[}\]])", r"\1", config_text)
	return json.loads(config_text)


def tensor_path(name: str) -> Path:
	return TENSOR_DIR / name


def load_tensor(tensor: str, rows: int, cols: int) -> torch.Tensor:
	path = tensor_path(tensor)
	raw = path.read_bytes()
	data = torch.frombuffer(bytearray(raw), dtype=torch.int16)
	return data.view(torch.bfloat16).reshape(rows, cols).to(get_device())


def store_tensor(tensor: torch.Tensor, file_name: str) -> None:
	path = tensor_path(file_name)
	path.parent.mkdir(parents=True, exist_ok=True)
	data = tensor.contiguous().cpu()
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
	head_size = int(config["head_size"])
	qkv_fan_in = int(config["qkv_fan_in"])
	qkv_weights_rows = 3 * n_heads * head_size
	device = get_device()

	generator = torch.Generator(device=device)
	generator.manual_seed(42)
	inputs = torch.randn((n_inputs, d_model), dtype=torch.float32, device=device, generator=generator).to(torch.bfloat16)
	qkv_weights = torch.randn((qkv_weights_rows, qkv_fan_in), dtype=torch.float32, device=device, generator=generator).to(torch.bfloat16)
	head_params = torch.zeros((n_heads, 4), dtype=torch.float32, device=device)
	head_params[:, 0] = 1.0 # gate
	head_params[:, 1] = 1.0 # temperature
	head_params[:, 2] = 0.0 # sink score
	head_params[:, 3] = 0.0 # unused
	store_tensor(inputs, "inputs.bin")
	store_tensor(qkv_weights, "qkv_weights.bin")
	store_f32_tensor(head_params, "head_params.bin")


def expand_qkv_weights(qkv_weights: torch.Tensor, d_model: int, n_heads: int) -> torch.Tensor:
	if d_model % n_heads != 0:
		raise ValueError(f"Expected d_model={d_model} to be divisible by n_heads={n_heads}")
	step = d_model // n_heads
	expanded = torch.zeros(
		(qkv_weights.shape[0], d_model),
		dtype=qkv_weights.dtype,
		device=qkv_weights.device,
	)
	cols = torch.arange(qkv_weights.shape[1], dtype=torch.int64, device=qkv_weights.device)
	for row in range(qkv_weights.shape[0]):
		indices = (cols + row * step) % d_model
		expanded[row, indices] = qkv_weights[row]
	return expanded


def qkv_proj(inputs: torch.Tensor, qkv_weights: torch.Tensor, config: dict) -> torch.Tensor:
	d_model = int(config["d_model"])
	n_heads = int(config["n_heads"])
	expanded_qkv_weights = expand_qkv_weights(qkv_weights, d_model, n_heads)
	print(f"inputs shape: {inputs.shape[0]} x {inputs.shape[1]}")
	print(f"Expanded qkv_weights shape: {expanded_qkv_weights.shape[0]} x {expanded_qkv_weights.shape[1]}")
	return torch.matmul(inputs, expanded_qkv_weights.transpose(0, 1)).to(torch.bfloat16)


def l2_norm_last_dim(tensor: torch.Tensor) -> torch.Tensor:
	tensor_f32 = tensor.to(torch.float32)
	norm = torch.linalg.vector_norm(tensor_f32, ord=2, dim=-1, keepdim=True)
	return (tensor_f32 / norm.clamp_min(1e-30)).to(torch.bfloat16)


def split_qkv(qkv: torch.Tensor, config: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	n_inputs = int(config["n_inputs"])
	n_heads = int(config["n_heads"])
	head_size = int(config["head_size"])
	qkv_cols = n_heads * head_size

	q = qkv[:, :qkv_cols].reshape(n_inputs, n_heads, head_size)
	k = qkv[:, qkv_cols:2 * qkv_cols].reshape(n_inputs, n_heads, head_size)
	v = qkv[:, 2 * qkv_cols:3 * qkv_cols].reshape(n_inputs, n_heads, head_size)

	return l2_norm_last_dim(q), l2_norm_last_dim(k), l2_norm_last_dim(v)


def run_block(config: dict) -> None:
	n_inputs = int(config["n_inputs"])
	d_model = int(config["d_model"])
	n_heads = int(config["n_heads"])
	head_size = int(config["head_size"])
	qkv_fan_in = int(config["qkv_fan_in"])
	qkv_weights_rows = 3 * n_heads * head_size

	inputs = load_tensor("inputs.bin", n_inputs, d_model)
	qkv_weights = load_tensor("qkv_weights.bin", qkv_weights_rows, qkv_fan_in)
	qkv = qkv_proj(inputs, qkv_weights, config)
	q, k, v = split_qkv(qkv, config)

	print("inputs shape:", inputs.shape)
	print("qkv_weights shape:", qkv_weights.shape)
	print("qkv shape:", qkv.shape)
	print("q shape:", q.shape)
	print("k shape:", k.shape)
	print("v shape:", v.shape)

	store_tensor(qkv, "qkv.bin")
	store_tensor(q, "q.bin")
	store_tensor(k, "k.bin")
	store_tensor(v, "v.bin")


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
