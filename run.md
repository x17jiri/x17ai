./nvcc.sh qkv_gemm.cu && tmp/qkv_gemm
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

./nvcc.sh attn_forward.cu && tmp/attn_forward
python verify_tensor.py tmp/block_torch/attn_out.bin tmp/block_cuda/attn_out.bin

python tensor_stats.py tmp/block_torch/attn_out.bin tmp/block_torch/attn_out.bin.var
python tensor_stats.py tmp/block_torch/attn_out_pregate.bin tmp/block_torch/attn_out_pregate.bin.var
python tensor_stats.py tmp/block_torch/attn_out.bin tmp/block_torch/attn_out.bin.var --overlay tmp/block_torch/f.bin --overlay-var tmp/block_torch/f.bin.var

./nvcc.sh f_gemm.cu && tmp/f_gemm
python verify_tensor.py tmp/block_torch/f.bin tmp/block_cuda/f.bin
python tensor_stats.py tmp/block_torch/f.bin tmp/block_torch/f.bin.var

./nvcc.sh o_gemm.cu && tmp/o_gemm
python verify_tensor.py tmp/block_torch/o.bin tmp/block_cuda/o.bin
python tensor_stats.py tmp/block_torch/o.bin tmp/block_torch/o.bin.var

paste <(cat /sys/class/thermal/thermal_zone*/type) <(cat /sys/class/thermal/thermal_zone*/temp) | column -s $'\t' -t | sed 's/\(.\)..$/.\1°C/'
