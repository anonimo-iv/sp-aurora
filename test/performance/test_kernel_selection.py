#!/usr/bin/env python3
"""
Benchmark different SYCL Flash Attention kernel implementations
Allows manual selection of kernels to identify performance bottlenecks
"""

import os
import sys
import torch
import numpy as np
import time
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_aurora.intel_flash_attn_sycl import (
    is_sycl_available,
    get_sycl_device_info,
    intel_flash_attn_forward_sycl,
)

# Import the SYCL module directly to access individual kernels
try:
    import sp_aurora.sycl_flash_attn as sycl_fa
    HAS_SYCL = True
except ImportError:
    HAS_SYCL = False
    print("SYCL module not available!")
    sys.exit(1)

# Import baseline implementations
from sp_aurora.intel_flash_attn import intel_flash_attn_forward


def benchmark_kernel(func, q, k, v, causal=True, warmup=3, iterations=10, kernel_name="Unknown"):
    """Benchmark a single kernel implementation"""
    device = q.device
    
    # Warmup
    try:
        for _ in range(warmup):
            _ = func(q, k, v, causal=causal)
            if device.type == 'xpu':
                torch.xpu.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
    except Exception as e:
        print(f"  {kernel_name} warmup failed: {str(e)}")
        return None, None
    
    # Benchmark
    try:
        if device.type == 'xpu':
            torch.xpu.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
            
        start = time.time()
        for _ in range(iterations):
            output, lse = func(q, k, v, causal=causal)
            
        if device.type == 'xpu':
            torch.xpu.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
            
        elapsed = (time.time() - start) / iterations * 1000  # ms
        
        # Calculate memory bandwidth
        # Each attention operation reads Q, K, V and writes O
        # Memory accessed = (batch * heads * seq_len * head_dim * 4 * sizeof(float)) bytes
        batch, heads, seq_len, head_dim = q.shape
        memory_bytes = batch * heads * seq_len * head_dim * 4 * 4  # 4 tensors, 4 bytes per float
        memory_gb = memory_bytes / (1024**3)
        bandwidth_gbps = memory_gb / (elapsed / 1000)  # GB/s
        
        return elapsed, bandwidth_gbps
        
    except Exception as e:
        print(f"  {kernel_name} benchmark failed: {str(e)}")
        return None, None


def get_theoretical_flops(batch, heads, seq_len, head_dim, causal=True):
    """Calculate theoretical FLOPs for attention"""
    # QK^T: 2 * batch * heads * seq_len * seq_len * head_dim
    qk_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    
    # Softmax: ~5 ops per element (exp, sum, div)
    if causal:
        softmax_flops = 5 * batch * heads * seq_len * (seq_len + 1) // 2
    else:
        softmax_flops = 5 * batch * heads * seq_len * seq_len
    
    # Attention * V: 2 * batch * heads * seq_len * seq_len * head_dim
    av_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    
    return (qk_flops + softmax_flops + av_flops) / 1e9  # GFLOPs


def create_sycl_kernel_wrapper(kernel_type):
    """Create a wrapper for different SYCL kernel implementations"""
    def wrapper(q, k, v, causal=True):
        return intel_flash_attn_forward_sycl(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=None,  # Will be calculated automatically
            causal=causal,
            kernel_type=kernel_type
        )
    
    return wrapper


def main():
    parser = argparse.ArgumentParser(description='Benchmark SYCL Flash Attention kernels')
    parser.add_argument('--seq-lengths', type=int, nargs='+', 
                       default=[32, 64, 128, 256, 512, 1024, 2048],
                       help='Sequence lengths to test')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--num-heads', type=int, default=32,
                       help='Number of attention heads')
    parser.add_argument('--head-dim', type=int, default=128,
                       help='Head dimension')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    if not is_sycl_available():
        print("SYCL is not available!")
        return
    
    # Get device info
    device_info = get_sycl_device_info()
    print("Intel GPU Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print()
    
    device = 'xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Testing configurations:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Head dimension: {args.head_dim}")
    print(f"  Sequence lengths: {args.seq_lengths}")
    print()
    
    # Kernel implementations to test
    kernels = {
        "PyTorch Baseline": intel_flash_attn_forward,
        "SYCL Auto-Select": create_sycl_kernel_wrapper("auto"),
        "SYCL XMX": create_sycl_kernel_wrapper("xmx"),
        "SYCL Optimized V3": create_sycl_kernel_wrapper("optimized_v3"),
        "SYCL Optimized V4": create_sycl_kernel_wrapper("optimized_v4"),
        "SYCL Optimized V5": create_sycl_kernel_wrapper("optimized_v5"),
        "SYCL Optimized V7": create_sycl_kernel_wrapper("optimized_v7"),
        "SYCL Optimized V8": create_sycl_kernel_wrapper("optimized_v8"),
    }
    
    # Results storage
    results = {kernel: {'seq_lens': [], 'times': [], 'bandwidth': [], 'gflops': []} 
               for kernel in kernels.keys()}
    
    # Benchmark each configuration
    print("Benchmarking...")
    print(f"{'SeqLen':<10} {'Kernel':<25} {'Time (ms)':<15} {'BW (GB/s)':<15} {'GFLOPS':<15} {'% Peak':<10}")
    print("-" * 90)
    
    for seq_len in args.seq_lengths:
        # Create test tensors
        q = torch.randn(args.batch_size, args.num_heads, seq_len, args.head_dim, 
                       device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Normalize to prevent overflow
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        
        # Calculate theoretical metrics
        theoretical_gflops = get_theoretical_flops(
            args.batch_size, args.num_heads, seq_len, args.head_dim, causal=True
        )
        
        # Test each kernel
        for kernel_name, kernel_func in kernels.items():
            time_ms, bandwidth_gbps = benchmark_kernel(
                kernel_func, q, k, v, causal=True, kernel_name=kernel_name
            )
            
            if time_ms is not None:
                achieved_gflops = theoretical_gflops / (time_ms / 1000)
                # Intel GPU Max 1550 has ~52 TFLOPS FP32 peak
                peak_percentage = (achieved_gflops / 52000) * 100
                
                results[kernel_name]['seq_lens'].append(seq_len)
                results[kernel_name]['times'].append(time_ms)
                results[kernel_name]['bandwidth'].append(bandwidth_gbps)
                results[kernel_name]['gflops'].append(achieved_gflops)
                
                print(f"{seq_len:<10} {kernel_name:<25} {time_ms:<15.2f} {bandwidth_gbps:<15.2f} "
                      f"{achieved_gflops:<15.2f} {peak_percentage:<10.2f}%")
    
    # Summary
    print("\nPerformance Summary:")
    print(f"{'Kernel':<25} {'Avg Time (ms)':<15} {'Avg BW (GB/s)':<15} {'Avg GFLOPS':<15}")
    print("-" * 70)
    
    for kernel_name in kernels.keys():
        if results[kernel_name]['times']:
            avg_time = np.mean(results[kernel_name]['times'])
            avg_bw = np.mean(results[kernel_name]['bandwidth'])
            avg_gflops = np.mean(results[kernel_name]['gflops'])
            print(f"{kernel_name:<25} {avg_time:<15.2f} {avg_bw:<15.2f} {avg_gflops:<15.2f}")
    
    # Generate plots if requested
    if args.plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Time vs Sequence Length
        for kernel_name in kernels.keys():
            if results[kernel_name]['times']:
                ax1.plot(results[kernel_name]['seq_lens'], 
                        results[kernel_name]['times'], 
                        marker='o', label=kernel_name)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Execution Time vs Sequence Length')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        
        # Memory Bandwidth
        for kernel_name in kernels.keys():
            if results[kernel_name]['bandwidth']:
                ax2.plot(results[kernel_name]['seq_lens'], 
                        results[kernel_name]['bandwidth'], 
                        marker='o', label=kernel_name)
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Memory Bandwidth (GB/s)')
        ax2.set_title('Achieved Memory Bandwidth')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xscale('log', base=2)
        
        # GFLOPS
        for kernel_name in kernels.keys():
            if results[kernel_name]['gflops']:
                ax3.plot(results[kernel_name]['seq_lens'], 
                        results[kernel_name]['gflops'], 
                        marker='o', label=kernel_name)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('GFLOPS')
        ax3.set_title('Achieved GFLOPS')
        ax3.legend()
        ax3.grid(True)
        ax3.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig('flash_attention_kernel_benchmark.png', dpi=150)
        print(f"\nPlots saved to flash_attention_kernel_benchmark.png")


if __name__ == "__main__":
    main()