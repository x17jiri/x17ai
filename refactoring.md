1. `cp_async_gmem_to_smem` migration note:
   If a call of this function is failing to compile, it is likely it hasn't been updated for the new header.
   The old header was:
   `usize tid, GMatrix<T, GM, GN> src, usize dst_row, usize dst_col`

   Newly we also have `src_row`, `src_col`.

   Additionally, the old version didn't have `WIDTH` and `HEIGHT` template parameters. The area to copy was inferred from `GN` and `GM`.

   So:
   `cp_async_gmem_to_smem<THREADS>(tid, src, dst, dst_row, dst_col)`
   becomes:
   `cp_async_gmem_to_smem<THREADS, SRC_COLS, SRC_ROWS>(tid, src, dst, 0, 9, dst_row, dst_col)`

2. `head_params.bin` layout from `block.py` is now just `[temperature]` per head.
   The previous 4-value layout `[gate, temperature, sink score, unused]` is no longer written by the Python generator.
   Any legacy consumer that still assumes 4 floats per head needs to be updated before using this file.

3. The QKV projection path has been extended to QKVG.
   The Python reference and CUDA harness now use `4 * n_heads * head_dim` projection rows and emit `qkvg.bin`.
   The `G` branch is postprocessed as `l2_norm`, then elementwise multiplied by `g_weights`, then passed through the tanh-based GELU approximation used in `utils.cuh`.
   The measured throughput drop from about `35.3` TFLOPS to about `34.5` TFLOPS persists even when `g_weights` are replaced with constants, so the cost appears to come from the added `G` math rather than from preloading or fetching `g_weights`.
