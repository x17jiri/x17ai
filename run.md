# QKVG Fwd

./nvcc.sh qkvg_fwd.cu && tmp/qkvg_fwd
python verify_tensor.py tmp/block_torch/sink_scores_f32.bin tmp/block_cuda/sink_scores_f32.bin
python verify_tensor.py tmp/block_torch/qkvg.bin tmp/block_cuda/qkvg.bin --shape 16384 4 1024
python verify_tensor.py tmp/block_torch/q.bin tmp/block_cuda/q.bin --shape 16384 1024
python verify_tensor.py tmp/block_torch/k.bin tmp/block_cuda/k.bin --shape 16384 1024
python verify_tensor.py tmp/block_torch/v.bin tmp/block_cuda/v.bin --shape 16384 1024
python verify_tensor.py tmp/block_torch/g.bin tmp/block_cuda/g.bin --shape 16384 1024

python tensor_stats.py tmp/block_torch/sink_scores_f32.bin tmp/block_torch/sink_scores_f32.bin.var
python tensor_stats.py tmp/block_torch/q.bin tmp/block_torch/q.bin.var
python tensor_stats.py tmp/block_torch/k.bin tmp/block_torch/k.bin.var
python tensor_stats.py tmp/block_torch/v.bin tmp/block_torch/v.bin.var
python tensor_stats.py tmp/block_torch/g.bin tmp/block_torch/g.bin.var

# Attn Forward

./nvcc.sh attn_fwd.cu && tmp/attn_fwd
python verify_tensor.py tmp/block_torch/attn_out.bin tmp/block_cuda/attn_out.bin --shape 16384 32 32

python tensor_stats.py tmp/block_torch/attn_out.bin tmp/block_torch/attn_out.bin.var
python tensor_stats.py tmp/block_torch/attn_out_pregate.bin tmp/block_torch/attn_out_pregate.bin.var
python tensor_stats.py tmp/block_torch/attn_out.bin tmp/block_torch/attn_out.bin.var --overlay tmp/block_torch/f.bin --overlay-var tmp/block_torch/f.bin.var

## Attn Forward - use maxes from torch to eliminate online softmax errors
./nvcc.sh attn_fwd.cu && tmp/attn_fwd --use-torch-maxes

# Attn O Proj Forward

./nvcc.sh o_attn_fwd.cu && tmp/o_attn_fwd
python verify_tensor.py tmp/block_torch/o_attn.bin tmp/block_cuda/o_attn.bin
python tensor_stats.py tmp/block_torch/o_attn.bin tmp/block_torch/o_attn.bin.var

# FFN Forward

./nvcc.sh ffn_fwd.cu && tmp/ffn_fwd
python verify_tensor.py tmp/block_torch/f.bin tmp/block_cuda/f.bin
python tensor_stats.py tmp/block_torch/f.bin tmp/block_torch/f.bin.var

# FFN O Proj Forward

./nvcc.sh o_ffn_fwd.cu && tmp/o_ffn_fwd
python verify_tensor.py tmp/block_torch/o_ffn.bin tmp/block_cuda/o_ffn.bin
python tensor_stats.py tmp/block_torch/o_ffn.bin tmp/block_torch/o_ffn.bin.var

# FFN O Proj Backward

./nvcc.sh o_ffn_dx.cu && tmp/o_ffn_dx
python verify_tensor.py tmp/block_torch/d_f.bin tmp/block_cuda/d_f.bin

# Chained run using CUDA outputs as inputs for later kernels.

- Model parameters and block-entry tensors still load from tmp/block_torch.

./nvcc.sh qkvg_fwd.cu && tmp/qkvg_fwd
./nvcc.sh attn_fwd.cu && tmp/attn_fwd --cuda-inputs
./nvcc.sh o_attn_fwd.cu && tmp/o_attn_fwd --cuda-inputs
python verify_tensor.py tmp/block_torch/o_attn.bin tmp/block_cuda/o_attn.bin

./nvcc.sh ffn_fwd.cu && tmp/ffn_fwd
./nvcc.sh o_ffn_fwd.cu && tmp/o_ffn_fwd --cuda-inputs
python verify_tensor.py tmp/block_torch/o_ffn.bin tmp/block_cuda/o_ffn.bin

- It is still possible to use maxes from torch to get closer match

./nvcc.sh attn_fwd.cu && tmp/attn_fwd --cuda-inputs --use-torch-maxes

# Check CPU Temperature

paste <(cat /sys/class/thermal/thermal_zone*/type) <(cat /sys/class/thermal/thermal_zone*/temp) | column -s $'\t' -t | sed 's/\(.\)..$/.\1°C/'
