1.
	In qkv kernel, try to merge these function:
			apply_rope(acc_t, block_m, block_n, warp_m, warp_n);
			apply_g_weight_and_gelu(acc_t, g_weights, block_m, warp_m);
	They both have a loop condition that could be merged.
