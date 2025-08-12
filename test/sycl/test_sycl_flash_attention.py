#!/usr/bin/env python3
"""
Test SYCL Flash Attention implementation
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_aurora.intel_flash_attn_sycl import (
    is_sycl_available,
    get_sycl_device_info,
    intel_flash_attn_forward_sycl,
    auto_select_flash_attn_forward
)
from sp_aurora.intel_flash_attn import intel_flash_attn_forward


def test_sycl_availability():
    """Test if SYCL module is available"""
    print("\n" + "="*60)
    print("TEST: SYCL Availability")
    print("="*60)
    
    if is_sycl_available():
        print("✓ SYCL Flash Attention is available")
        
        # Get device info
        info = get_sycl_device_info()
        print("\nIntel GPU Device Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("✗ SYCL Flash Attention is NOT available")
        print("  Please build with: BUILD_SYCL=1 pip install -e .")
    
    return is_sycl_available()


def test_correctness():
    """Test correctness of SYCL implementation against reference"""
    print("\n" + "="*60)
    print("TEST: Correctness Validation")
    print("="*60)
    
    # Test configurations
    configs = [
        {"batch": 2, "heads": 8, "seq_len": 128, "head_dim": 64, "causal": True},
        {"batch": 1, "heads": 12, "seq_len": 256, "head_dim": 64, "causal": False},
        {"batch": 4, "heads": 16, "seq_len": 512, "head_dim": 128, "causal": True},
    ]
    
    device = 'xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config}")
        
        # Create test tensors
        torch.manual_seed(42)
        q = torch.randn(config["batch"], config["heads"], config["seq_len"], 
                       config["head_dim"], device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Normalize to prevent overflow
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        
        # Run reference implementation
        ref_out, ref_lse = intel_flash_attn_forward(
            q, k, v, causal=config["causal"]
        )
        
        # Run SYCL implementation
        sycl_out, sycl_lse = intel_flash_attn_forward_sycl(
            q, k, v, causal=config["causal"]
        )
        
        # Compare outputs
        out_diff = torch.abs(ref_out - sycl_out).max().item()
        lse_diff = torch.abs(ref_lse - sycl_lse).max().item()
        
        print(f"  Output max diff: {out_diff:.6e}")
        print(f"  LSE max diff: {lse_diff:.6e}")
        
        # Check if within tolerance
        tolerance = 1e-3
        if out_diff < tolerance and lse_diff < tolerance:
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED - Difference exceeds tolerance")


def benchmark_performance():
    """Benchmark SYCL vs Python implementation"""
    print("\n" + "="*60)
    print("TEST: Performance Benchmark")
    print("="*60)
    
    # Benchmark configurations - extended with larger sequences
    configs = [
        {"batch": 1, "heads": 16, "seq_len": 32, "head_dim": 64},
        {"batch": 1, "heads": 32, "seq_len": 64, "head_dim": 64},
        {"batch": 1, "heads": 32, "seq_len": 128, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 256, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 512, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 1024, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128},
        {"batch": 1, "heads": 16, "seq_len": 4096, "head_dim": 128},
    ]
    
    device = 'xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print("Causal=True, measuring average over 10 iterations")
    print("\n{:<10} {:<10} {:<15} {:<15} {:<10}".format("SeqLen", "Heads", "Python (ms)", "SYCL (ms)", "Speedup"))
    print("-" * 70)
    
    for config in configs:
        # Create test tensors
        q = torch.randn(config["batch"], config["heads"], config["seq_len"], 
                       config["head_dim"], device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Warmup - with error handling for large sequences
        try:
            for _ in range(3):
                _ = intel_flash_attn_forward(q, k, v, causal=True)
                _ = intel_flash_attn_forward_sycl(q, k, v, causal=True)
        except Exception as e:
            print("{:<10} {:<10} {:<15} {:<15} {:<10}".format(
                config["seq_len"], config["heads"], "N/A", "Error", "N/A"
            ))
            print(f"           Error: {str(e)[:50]}...")
            continue
        
        # Benchmark Python implementation
        try:
            torch.xpu.synchronize() if device == 'xpu' else (torch.cuda.synchronize() if device == 'cuda' else None)
            start = time.time()
            for _ in range(10):
                _ = intel_flash_attn_forward(q, k, v, causal=True)
            torch.xpu.synchronize() if device == 'xpu' else (torch.cuda.synchronize() if device == 'cuda' else None)
            python_time = (time.time() - start) / 10 * 1000  # ms
        except Exception as e:
            python_time = float('inf')
        
        # Benchmark SYCL implementation  
        try:
            torch.xpu.synchronize() if device == 'xpu' else (torch.cuda.synchronize() if device == 'cuda' else None)
            start = time.time()
            for _ in range(10):
                _ = intel_flash_attn_forward_sycl(q, k, v, causal=True)
            torch.xpu.synchronize() if device == 'xpu' else (torch.cuda.synchronize() if device == 'cuda' else None)
            sycl_time = (time.time() - start) / 10 * 1000  # ms
        except Exception as e:
            sycl_time = float('inf')
            print("{:<10} {:<10} {:<15.2f} {:<15} {:<10}".format(
                config["seq_len"], config["heads"], python_time, "Error", "N/A"
            ))
            continue
        
        speedup = python_time / sycl_time
        
        print("{:<10} {:<10} {:<15.2f} {:<15.2f} {:<10.2f}x".format(
            config["seq_len"], config["heads"], python_time, sycl_time, speedup
        ))


def test_auto_selection():
    """Test automatic implementation selection"""
    print("\n" + "="*60)
    print("TEST: Auto Implementation Selection")
    print("="*60)
    
    device = 'xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create small test tensor
    q = torch.randn(1, 8, 128, 64, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    # Test auto selection
    out, lse = auto_select_flash_attn_forward(q, k, v, causal=True)
    
    print(f"Auto-selected implementation returned:")
    print(f"  Output shape: {out.shape}")
    print(f"  LSE shape: {lse.shape}")
    print("✓ Auto selection working correctly")


if __name__ == "__main__":
    print("SYCL Flash Attention Test Suite")
    print("================================\n")
    
    # Check availability first
    sycl_available = test_sycl_availability()
    
    if sycl_available:
        # Run tests
        test_correctness()
        benchmark_performance()
        test_auto_selection()
    else:
        print("\nSkipping tests as SYCL is not available.")
        print("To build SYCL support:")
        print("  1. Ensure Intel oneAPI is installed")
        print("  2. Run: cd sp_aurora/sycl && ./build.sh")
        print("  3. Install: BUILD_SYCL=1 pip install -e .")
    
    print("\nTest suite completed!")