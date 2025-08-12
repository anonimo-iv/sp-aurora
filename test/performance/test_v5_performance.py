#!/usr/bin/env python3
"""
Test V5 kernel performance specifically for non-causal attention
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_aurora.intel_flash_attn_sycl import intel_flash_attn_forward_sycl
from sp_aurora.intel_flash_attn import intel_flash_attn_forward

def benchmark(func, q, k, v, causal=False, warmup=5, iterations=20):
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        _ = func(q, k, v, causal=causal)
        if q.device.type == 'xpu':
            torch.xpu.synchronize()
    
    # Time
    torch.xpu.synchronize()
    start = time.time()
    for _ in range(iterations):
        output, lse = func(q, k, v, causal=causal)
    torch.xpu.synchronize()
    elapsed = (time.time() - start) / iterations * 1000  # ms
    
    return elapsed, output, lse

def main():
    device = 'xpu'
    batch_size = 1
    num_heads = 32
    head_dim = 128
    
    print("Testing V4 vs V5 kernel performance for non-causal attention")
    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}")
    print()
    
    for seq_len in [128, 256, 512]:
        print(f"\nSequence length: {seq_len}")
        
        # Create tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Normalize
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        
        # Test PyTorch baseline
        def pytorch_func(q, k, v, causal=False):
            return intel_flash_attn_forward(q, k, v, causal=causal)
        
        # Test V4
        def v4_func(q, k, v, causal=False):
            return intel_flash_attn_forward_sycl(q, k, v, causal=causal, kernel_type="optimized_v4")
        
        # Test V5
        def v5_func(q, k, v, causal=False):
            return intel_flash_attn_forward_sycl(q, k, v, causal=causal, kernel_type="optimized_v5")
        
        # Benchmark
        pytorch_time, pytorch_out, _ = benchmark(pytorch_func, q, k, v, causal=False)
        v4_time, v4_out, _ = benchmark(v4_func, q, k, v, causal=False)
        v5_time, v5_out, _ = benchmark(v5_func, q, k, v, causal=False)
        
        # Verify correctness
        pytorch_out_cpu = pytorch_out.cpu()
        v4_diff = torch.abs(v4_out.cpu() - pytorch_out_cpu).max().item()
        v5_diff = torch.abs(v5_out.cpu() - pytorch_out_cpu).max().item()
        
        print(f"  PyTorch: {pytorch_time:.2f} ms")
        print(f"  V4:      {v4_time:.2f} ms (speedup: {pytorch_time/v4_time:.2f}x, max_diff: {v4_diff:.6f})")
        print(f"  V5:      {v5_time:.2f} ms (speedup: {pytorch_time/v5_time:.2f}x, max_diff: {v5_diff:.6f})")
        
        # Memory bandwidth calculation
        memory_bytes = batch_size * num_heads * seq_len * head_dim * 4 * 4  # 4 tensors, 4 bytes
        memory_gb = memory_bytes / (1024**3)
        
        print(f"  V4 BW:   {memory_gb / (v4_time / 1000):.2f} GB/s")
        print(f"  V5 BW:   {memory_gb / (v5_time / 1000):.2f} GB/s")

if __name__ == "__main__":
    main()