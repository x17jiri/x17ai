# Attn i8

## Attn - QKV Fwd

./nvcc.sh attn_q_fwd_i8.cu && tmp/attn_q_fwd_i8
python verify_i8_tensor.py tmp/block_torch/q_i8.bin tmp/block_cuda/q_i8.bin --shape 16384 64 1 32

python tensor_stats.py tmp/block_torch/q_i8.bin

./nvcc.sh attn_kv_fwd_i8.cu && tmp/attn_kv_fwd_i8
python verify_i8_tensor.py tmp/block_torch/kv_i8.bin tmp/block_cuda/kv_i8.bin --shape 16384 64 2 32

python tensor_stats.py tmp/block_torch/kv_i8.bin
python tensor_stats.py tmp/block_torch/k_i8.bin
python tensor_stats.py tmp/block_torch/v_i8.bin

## Attn - A Fwd

./nvcc.sh attn_fwd_i8.cu && tmp/attn_fwd_i8
python verify_i8_tensor.py tmp/block_torch/attn_out_i8.bin tmp/block_cuda/attn_out_i8.bin --shape 16384 64 1 32

python tensor_stats.py tmp/block_torch/attn_out_i8.bin tmp/block_torch/attn_out.bin.var

## Attn - Y Fwd

./nvcc.sh attn_y_fwd_i8.cu && tmp/attn_y_fwd_i8
python verify_i8_tensor.py tmp/block_torch/attn_y_i8.bin tmp/block_cuda/attn_y_i8.bin

python tensor_stats.py tmp/block_torch/ffn_y.bin

./nvcc.sh attn_y_fwd_fp8.cu && tmp/attn_y_fwd_fp8
python verify_tensor.py tmp/block_torch/o_attn.bin tmp/block_cuda/o_attn.bin
python tensor_stats.py tmp/block_torch/o_attn.bin tmp/block_torch/o_attn.bin.var

# FFN i8

## FFN - F Fwd

./nvcc.sh ffn_f_fwd_f8.cu && tmp/ffn_f_fwd_f8
python verify_tensor.py tmp/block_torch/ffn_f_f8.bin tmp/block_cuda/ffn_f_f8.bin
python tensor_stats.py tmp/block_torch/ffn_f.bin tmp/block_torch/ffn_f.bin.var

## FFN - Y Fwd

./nvcc.sh ffn_y_fwd_i8.cu && tmp/ffn_y_fwd_i8
python verify_i8_tensor.py tmp/block_torch/ffn_y_i8.bin tmp/block_cuda/ffn_y_i8.bin

python tensor_stats.py tmp/block_torch/ffn_y.bin








# Attn

## Attn - QKVG Fwd

./nvcc.sh qkvg_fwd.cu && tmp/qkvg_fwd
python verify_tensor.py tmp/block_torch/qkvg.bin tmp/block_cuda/qkvg.bin --shape 16384 4 1024
python verify_tensor.py tmp/block_torch/q.bin tmp/block_cuda/q.bin --shape 16384 1024
python verify_tensor.py tmp/block_torch/k.bin tmp/block_cuda/k.bin --shape 16384 1024
python verify_tensor.py tmp/block_torch/v.bin tmp/block_cuda/v.bin --shape 16384 1024
python verify_tensor.py tmp/block_torch/g.bin tmp/block_cuda/g.bin --shape 16384 1024

python tensor_stats.py tmp/block_torch/q.bin tmp/block_torch/q.bin.var
python tensor_stats.py tmp/block_torch/k.bin tmp/block_torch/k.bin.var
python tensor_stats.py tmp/block_torch/v.bin tmp/block_torch/v.bin.var
python tensor_stats.py tmp/block_torch/g.bin

./nvcc.sh attn_kv_fwd_i8.cu && tmp/attn_kv_fwd_i8
python verify_i8_tensor.py tmp/block_torch/kv_i8.bin tmp/block_cuda/kv_i8.bin --shape 16384 64 2 32
python tensor_stats.py tmp/block_torch/kv_i8.bin
python tensor_stats.py tmp/block_torch/k_i8.bin
python tensor_stats.py tmp/block_torch/v_i8.bin

./nvcc.sh attn_q_fwd_i8.cu && tmp/attn_q_fwd_i8
python verify_i8_tensor.py tmp/block_torch/q_i8.bin tmp/block_cuda/q_i8.bin --shape 16384 64 1 32
python tensor_stats.py tmp/block_torch/q_i8.bin

## Attn - A Fwd

./nvcc.sh attn_fwd_i8.cu && tmp/attn_fwd_i8
python verify_i8_tensor.py tmp/block_torch/attn_out_i8.bin tmp/block_cuda/attn_out_i8.bin --shape 16384 64 1 32

python tensor_stats.py tmp/block_torch/attn_out_i8.bin tmp/block_torch/attn_out.bin.var

./nvcc.sh attn_fwd.cu && tmp/attn_fwd
python verify_tensor.py tmp/block_torch/attn_out.bin tmp/block_cuda/attn_out.bin --shape 16384 32 32

python tensor_stats.py tmp/block_torch/attn_out.bin tmp/block_torch/attn_out.bin.var
python tensor_stats.py tmp/block_torch/attn_out_pregate.bin tmp/block_torch/attn_out_pregate.bin.var
python tensor_stats.py tmp/block_torch/attn_out.bin tmp/block_torch/attn_out.bin.var --overlay tmp/block_torch/f.bin --overlay-var tmp/block_torch/f.bin.var

### Attn Forward - use maxes from torch to eliminate online softmax errors
./nvcc.sh attn_fwd.cu && tmp/attn_fwd --use-torch-maxes

## Attn - Y Fwd

./nvcc.sh attn_y.cu && tmp/attn_y
python verify_tensor.py tmp/block_torch/o_attn.bin tmp/block_cuda/o_attn.bin
python tensor_stats.py tmp/block_torch/o_attn.bin tmp/block_torch/o_attn.bin.var

# FFN

## FFN F Forward

./nvcc.sh ffn_f_fwd.cu && tmp/ffn_f_fwd
python verify_tensor.py tmp/block_torch/ffn_f.bin tmp/block_cuda/ffn_f.bin
python tensor_stats.py tmp/block_torch/ffn_f.bin tmp/block_torch/ffn_f.bin.var

./nvcc.sh ffn_f_fwd_i8.cu && tmp/ffn_f_fwd_i8
python verify_i8_tensor.py tmp/block_torch/ffn_f_i8.bin tmp/block_cuda/ffn_f_i8.bin
python tensor_stats.py tmp/block_torch/ffn_f_i8.bin

## FFN Y Forward

./nvcc.sh ffn_y_fwd.cu && tmp/ffn_y_fwd
python verify_tensor.py tmp/block_torch/ffn_y.bin tmp/block_cuda/ffn_y.bin
python tensor_stats.py tmp/block_torch/ffn_y.bin tmp/block_torch/ffn_y.bin.var

## FFN O Backward

./nvcc.sh ffn_d_f.cu && tmp/ffn_d_f
python verify_tensor.py tmp/block_torch/ffn_d_f.bin tmp/block_cuda/ffn_d_f.bin

./nvcc.sh ffn_d_y_weights.cu && tmp/ffn_d_y_weights
python verify_tensor.py tmp/block_torch/ffn_d_y_weights.bin tmp/block_cuda/ffn_d_y_weights.bin --shape 1024 2048

./nvcc.sh ffn_d_f_weights.cu && tmp/ffn_d_f_weights
python verify_tensor.py tmp/block_torch/ffn_d_f_weights.bin tmp/block_cuda/ffn_d_f_weights.bin --shape 4096 512

## FFN Input Backward

- `ffn_d_x.cu` consumes `ffn_d_f.bin` and `ffn_f_backvec.bin`.
- It fuses `ffn_d_f` with the cached GeGLU backvec on the GPU during the GEMM preload path; there is no host-side `ffn_d_t` materialization.
- `block.py` now emits `ffn_f_backvec.bin` alongside the other FFN reference tensors, so the default run uses `tmp/block_torch` directly.

./nvcc.sh ffn_d_x.cu && tmp/ffn_d_x
python verify_tensor.py tmp/block_torch/ffn_d_x.bin tmp/block_cuda/ffn_d_x.bin --shape 16384 1024

- For a chained CUDA-only intermediate path, generate CUDA `ffn_f_backvec.bin` and `ffn_d_f.bin` first:

./nvcc.sh ffn_f_fwd.cu && tmp/ffn_f_fwd
./nvcc.sh ffn_d_f.cu && tmp/ffn_d_f
tmp/ffn_d_x --cuda-inputs

# Chained run using CUDA outputs as inputs for later kernels.

- Model parameters and block-entry tensors still load from tmp/block_torch.

./nvcc.sh qkvg_fwd.cu && tmp/qkvg_fwd
./nvcc.sh attn_fwd.cu && tmp/attn_fwd --cuda-inputs
./nvcc.sh attn_y.cu && tmp/attn_y --cuda-inputs
python verify_tensor.py tmp/block_torch/o_attn.bin tmp/block_cuda/o_attn.bin

./nvcc.sh ffn_f_fwd.cu && tmp/ffn_f_fwd
./nvcc.sh ffn_y_fwd.cu && tmp/ffn_y_fwd --cuda-inputs
python verify_tensor.py tmp/block_torch/ffn_y.bin tmp/block_cuda/ffn_y.bin

- It is still possible to use maxes from torch to get closer match

./nvcc.sh attn_fwd.cu && tmp/attn_fwd --cuda-inputs --use-torch-maxes

# Check CPU Temperature

paste <(cat /sys/class/thermal/thermal_zone*/type) <(cat /sys/class/thermal/thermal_zone*/temp) | column -s $'\t' -t | sed 's/\(.\)..$/.\1°C/'
