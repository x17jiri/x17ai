# Attn forward precision

By enabling this line in attn_forward.cu:

```cpp
	if (0 && std::filesystem::exists("tmp/block_torch/attn_maxes_f32.bin")) {
```

The forward attention will use "max" values pre-calculated from pytorch.
This way it will do no rescaling which will eliminate error introduced
by the online softmax.

Current error with maxes:

	--- BF16 Tensor Compare ---
	A: tmp/block_torch/attn_out.bin
	B: tmp/block_cuda/attn_out.bin
	Shape: [16384, 32, 32]
	Max abs diff:     1.953125e-03
	Mean abs diff:    5.937132e-08
	Max pct diff:     9247.7180%, 3358.0647%, 622.7273%, 163.9175%, 23.5294%, 7.4074%
	(MIN_MAG after scaling each tensor to variance 1:
				1e-6,     1e-5,     1e-4,     1e-3,     1e-2,     1e-1)
	Exact bf16 match: 16714978/16777216 (99.63%)

Without:

	--- BF16 Tensor Compare ---
	A: tmp/block_torch/attn_out.bin
	B: tmp/block_cuda/attn_out.bin
	Shape: [16384, 32, 32]
	Max abs diff:     3.906250e-03
	Mean abs diff:    1.174604e-05
	Max pct diff:     165743.4814%, 25689.6301%, 5891.4894%, 1011.1111%, 149.3724%, 13.6126%
	(MIN_MAG after scaling each tensor to variance 1:
				1e-6,     1e-5,     1e-4,     1e-3,     1e-2,     1e-1)
	Exact bf16 match: 12181541/16777216 (72.61%)
