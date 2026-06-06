# Rationale

- I decided to split qk_norm_scales into separate q_norm_scales and k_norm_scales.
If we scale both sides, it is less likely that one of them will overflow with i8.

- For residual connection I use base-4 sigmoid and softplus.
In base 4, they both equal 0.5 when gate == 0.
