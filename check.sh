#!/usr/bin/env bash

out="tmp/check.txt"
mkdir -p tmp
echo "" > "$out"

python verify_i8_tensor.py tmp/block_torch/q_i8.safetensors tmp/block_rust/q_i8.safetensors >> "$out"
python verify_i8_tensor.py tmp/block_torch/kv_i8.safetensors tmp/block_rust/kv_i8.safetensors >> "$out"
python verify_tensor.py tmp/block_torch/k_rrms_f32.safetensors tmp/block_rust/k_rrms_f32.safetensors >> "$out"
python verify_i8_tensor.py tmp/block_torch/attn_out_i8.safetensors tmp/block_rust/attn_out_i8.safetensors >> "$out"
python verify_tensor.py tmp/block_torch/attn_l_f32.safetensors tmp/block_rust/attn_l_f32.safetensors >> "$out"
python verify_i8_tensor.py tmp/block_torch/attn_y_i8.safetensors tmp/block_rust/attn_y_i8.safetensors >> "$out"
python verify_tensor.py tmp/block_torch/ffn_f_f8.safetensors tmp/block_rust/ffn_f_f8.safetensors >> "$out"
python verify_i8_tensor.py tmp/block_torch/ffn_y_i8.safetensors tmp/block_rust/ffn_y_i8.safetensors >> "$out"

if ! diff -u expected.txt "$out"; then
	echo "ERROR: tensor check output differs from expected.txt"
	echo "Full output is in $out"
	exit 1
fi

echo "OK: tensor check output matches expected.txt"
