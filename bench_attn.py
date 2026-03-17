#!/usr/bin/env python3
"""
Benchmark PyTorch's scaled_dot_product_attention (FlashAttention / mem-efficient)
on the same problem size as the custom kernel in ldmatrix_example.cu.
"""

import torch
import torch.nn.functional as F
import math
import argparse

QK_DIM = 64
V_DIM = 64


def time_kernel(fn, warmup, iters):
    """Returns sorted list of times in ms."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.no_grad():
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in zip(starts, ends))


def bench_sdpa(q_len, kv_len, device="cuda", warmup=0, iters=1, dtype=torch.bfloat16, v_dim_override=None):
    v_dim = v_dim_override or V_DIM
    torch.manual_seed(42)

    Q = torch.randn(1, 1, q_len, QK_DIM, device=device, dtype=dtype)
    K = torch.randn(1, 1, kv_len, QK_DIM, device=device, dtype=dtype)
    V = torch.randn(1, 1, kv_len, v_dim, device=device, dtype=dtype)

    print(f"Problem: Q[{q_len}, {QK_DIM}]  K[{kv_len}, {QK_DIM}]  V[{kv_len}, {v_dim}]")
    print(f"dtype={dtype}, device={device}")

    # Detect available backends
    backends = []
    for name, backend in [
        ("flash",   torch.nn.attention.SDPBackend.FLASH_ATTENTION),
#        ("mem_eff", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
#        ("math",    torch.nn.attention.SDPBackend.MATH),
    ]:
        try:
            with torch.no_grad(), torch.nn.attention.sdpa_kernel(backend):
                _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            backends.append(name)
        except RuntimeError:
            pass
    print(f"Available backends: {backends}")

    # FLOPs: Q@K^T + softmax + attn@V
    flops_full = 2 * q_len * kv_len * QK_DIM + 2 * q_len * kv_len * v_dim + 5 * q_len * kv_len
    flops_causal = flops_full / 2
    print(f"FLOPs (full): {flops_full:.3e}  (causal): {flops_causal:.3e}")

    trim = iters // 10

    def report(label, times_ms, flops):
        trimmed = times_ms[trim:-trim] if trim > 0 else times_ms
        med = trimmed[len(trimmed) // 2]
        mn = times_ms[0]
        tflops = flops / (med * 1e-3) / 1e12
        print(f"  {label:20s}  median: {med:.4f} ms  min: {mn:.4f} ms  TFLOPS: {tflops:.2f}")
        return med

    # Per-backend: causal vs non-causal
    def make_ctx(backend):
        return torch.nn.attention.sdpa_kernel(backend)

    backend_map = {
        "flash":   torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        "mem_eff": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    }
    for bname, backend in backend_map.items():
        if bname not in backends:
            continue
        print(f"\n--- [{bname}] ---")
        for is_causal in [True]:
            label = "causal" if is_causal else "non-causal"
            flops = flops_causal if is_causal else flops_full
            def run(ic=is_causal, b=backend):
                with make_ctx(b):
                    return F.scaled_dot_product_attention(Q, K, V, is_causal=ic)
            times = time_kernel(run, warmup, iters)
            report(label, times, flops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch SDPA attention")
    parser.add_argument("--q-len", type=int, default=2*32768)
    parser.add_argument("--kv-len", type=int, default=2*32768)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        exit(1)

    #bench_sdpa(args.q_len, args.kv_len, warmup=args.warmup, iters=args.iters)

    # Also try with V padded to QK_DIM so flash attention backend is eligible
    print("\n" + "=" * 70)
    print("Re-running with V_DIM padded to QK_DIM (",QK_DIM,") to enable flash attention")
    print("=" * 70)
    bench_sdpa(args.q_len, args.kv_len, warmup=args.warmup, iters=args.iters)
