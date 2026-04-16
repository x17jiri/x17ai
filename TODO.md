-
	In qkv kernel, try to merge these function:
			apply_rope(acc_t, block_m, block_n, warp_m, warp_n);
			apply_g_weight_and_gelu(acc_t, g_weights, block_m, warp_m);
	They both have a loop condition that could be merged.

-
	Exclusive Self Attention: https://arxiv.org/abs/2603.09078
	Gated Attention: https://arxiv.org/abs/2505.06708
	Attn Sinks: https://arxiv.org/abs/2309.17453

-
	Assert seq_len is multiple of 64.
	Will need to use the incremental kernel for the rest of the sequence if not.
