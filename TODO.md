-
	Exclusive Self Attention: https://arxiv.org/abs/2603.09078
	Gated Attention: https://arxiv.org/abs/2505.06708
	Attn Sinks: https://arxiv.org/abs/2309.17453

-
	add sinkV

-
	Assert seq_len is multiple of 64.
	Will need to use the incremental kernel for the rest of the sequence if not.

- Optimizers: Lion - https://arxiv.org/pdf/2302.06675
