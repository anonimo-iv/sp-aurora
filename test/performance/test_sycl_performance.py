#!/usr/bin/env python3
"""
Quick performance test for optimized SYCL Flash Attention
"""
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_aurora.intel_flash_attn_sycl import (
    is_sycl_available,
    intel_flash_attn_forward_sycl,
)
from sp_aurora.intel_flash_attn import intel_flash_attn_forward

def test_performance():
    if not is_sycl_available():
        print("SYCL not available!")
        return
        
    device = 'xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test configurations
    configs = [
        {"batch": 1, "heads": 16, "seq_len": 128, "head_dim": 64},
        {"batch": 1, "heads": 32, "seq_len": 256, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 512, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 1024, "head_dim": 128},
    ]
    
    print("\n{:<10} {:<15} {:<15} {:<10}".format("SeqLen", "Python (ms)", "SYCL (ms)", "Speedup"))
    print("-" * 55)
    
    for config in configs:
        # Create test tensors
        q = torch.randn(config["batch"], config["heads"], config["seq_len"], 
                       config["head_dim"], device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Warmup
        for _ in range(3):
            _ = intel_flash_attn_forward(q, k, v, causal=True)
            _ = intel_flash_attn_forward_sycl(q, k, v, causal=True)
        
        # Benchmark Python
        torch.xpu.synchronize() if device == 'xpu' else None
        start = time.time()
        for _ in range(5):
            _ = intel_flash_attn_forward(q, k, v, causal=True)
        torch.xpu.synchronize() if device == 'xpu' else None
        python_time = (time.time() - start) / 5 * 1000
        
        # Benchmark SYCL
        torch.xpu.synchronize() if device == 'xpu' else None
        start = time.time()
        for _ in range(5):
            _ = intel_flash_attn_forward_sycl(q, k, v, causal=True)
        torch.xpu.synchronize() if device == 'xpu' else None
        sycl_time = (time.time() - start) / 5 * 1000
        
        speedup = python_time / sycl_time
        status = "✓" if speedup > 1 else "✗"
        
        print("{:<10} {:<15.2f} {:<15.2f} {:<10.2f}x {}".format(
            config["seq_len"], python_time, sycl_time, speedup, status
        ))

if __name__ == "__main__":
    test_performance()