#!/usr/bin/env python3
"""
Detailed memory bandwidth and performance analysis for SYCL Flash Attention
Helps identify the real bottlenecks for large sequences
"""

import os
import sys
import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ring_flash_attn.intel_flash_attn_sycl import (
    is_sycl_available,
    get_sycl_device_info,
    intel_flash_attn_forward_sycl,
)
from ring_flash_attn.intel_flash_attn import intel_flash_attn_forward


def get_memory_traffic(batch, heads, seq_len, head_dim, causal=True):
    """Calculate theoretical memory traffic for flash attention"""
    # Each block of Q needs to read all K and V
    # Q is read once per K/V block
    # K and V are read once per Q block
    
    # Assuming block sizes of 64x64 (typical)
    block_size = 64
    num_q_blocks = (seq_len + block_size - 1) // block_size
    num_kv_blocks = (seq_len + block_size - 1) // block_size
    
    # Memory reads:
    # - Q: read once per KV block = seq_len * head_dim * num_kv_blocks
    # - K: read once per Q block = seq_len * head_dim * num_q_blocks
    # - V: read once per Q block = seq_len * head_dim * num_q_blocks
    # Memory writes:
    # - O: written once = seq_len * head_dim
    # - LSE: written once = seq_len
    
    q_reads = batch * heads * seq_len * head_dim * num_kv_blocks * 4  # 4 bytes per float
    k_reads = batch * heads * seq_len * head_dim * num_q_blocks * 4
    v_reads = batch * heads * seq_len * head_dim * num_q_blocks * 4
    o_writes = batch * heads * seq_len * head_dim * 4
    lse_writes = batch * heads * seq_len * 4
    
    total_bytes = q_reads + k_reads + v_reads + o_writes + lse_writes
    return total_bytes / (1024**3)  # Convert to GB


def profile_kernel(kernel_name, kernel_func, q, k, v, causal=True, iterations=10):
    """Profile a kernel and return detailed metrics"""
    device = q.device
    batch, heads, seq_len, head_dim = q.shape
    
    # Warmup
    try:
        for _ in range(3):
            _ = kernel_func(q, k, v, causal=causal)
            if device.type == 'xpu':
                torch.xpu.synchronize()
    except Exception as e:
        return {
            'error': str(e),
            'time_ms': None,
            'bandwidth_gbps': None,
            'efficiency': None
        }
    
    # Time the kernel
    if device.type == 'xpu':
        torch.xpu.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        output, lse = kernel_func(q, k, v, causal=causal)
    
    if device.type == 'xpu':
        torch.xpu.synchronize()
    
    elapsed_ms = (time.time() - start) / iterations * 1000
    
    # Calculate achieved bandwidth
    memory_gb = get_memory_traffic(batch, heads, seq_len, head_dim, causal)
    bandwidth_gbps = memory_gb / (elapsed_ms / 1000)
    
    # Intel GPU Max 1550 has theoretical peak bandwidth of ~1.6 TB/s = 1600 GB/s
    efficiency = (bandwidth_gbps / 1600) * 100
    
    return {
        'time_ms': elapsed_ms,
        'bandwidth_gbps': bandwidth_gbps,
        'efficiency': efficiency,
        'memory_gb': memory_gb
    }


def main():
    if not is_sycl_available():
        print("SYCL is not available!")
        return
    
    # Get device info
    device_info = get_sycl_device_info()
    print("Intel GPU Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print()
    
    device = 'xpu' if torch.xpu.is_available() else 'cuda'
    
    # Test configurations focusing on the problematic range
    configs = [
        # Small sequences (where performance is good)
        {'batch': 1, 'heads': 32, 'seq_len': 32, 'head_dim': 128},
        {'batch': 1, 'heads': 32, 'seq_len': 64, 'head_dim': 128},
        # Transition region (where performance degrades)
        {'batch': 1, 'heads': 32, 'seq_len': 128, 'head_dim': 128},
        {'batch': 1, 'heads': 32, 'seq_len': 256, 'head_dim': 128},
        {'batch': 1, 'heads': 32, 'seq_len': 512, 'head_dim': 128},
        # Large sequences
        {'batch': 1, 'heads': 32, 'seq_len': 1024, 'head_dim': 128},
        {'batch': 1, 'heads': 32, 'seq_len': 2048, 'head_dim': 128},
    ]
    
    # Kernel implementations to test
    kernels = {
        "PyTorch": lambda q, k, v, causal: intel_flash_attn_forward(q, k, v, causal=causal),
        "SYCL Auto": lambda q, k, v, causal: intel_flash_attn_forward_sycl(q, k, v, causal=causal, kernel_type="auto"),
        "SYCL Basic": lambda q, k, v, causal: intel_flash_attn_forward_sycl(q, k, v, causal=causal, kernel_type="basic"),
        "SYCL Optimized": lambda q, k, v, causal: intel_flash_attn_forward_sycl(q, k, v, causal=causal, kernel_type="optimized_v3"),
        "SYCL Streaming": lambda q, k, v, causal: intel_flash_attn_forward_sycl(q, k, v, causal=causal, kernel_type="streaming"),
        "SYCL XMX": lambda q, k, v, causal: intel_flash_attn_forward_sycl(q, k, v, causal=causal, kernel_type="xmx"),
    }
    
    print("Memory Bandwidth Analysis")
    print("=" * 120)
    print(f"{'SeqLen':<8} {'Kernel':<20} {'Time(ms)':<12} {'BW(GB/s)':<12} {'BW Eff(%)':<12} {'Mem(GB)':<12} {'GB/s/seq':<12}")
    print("-" * 120)
    
    results = {kernel: {'seq_lens': [], 'times': [], 'bandwidth': [], 'efficiency': []} 
               for kernel in kernels.keys()}
    
    for config in configs:
        seq_len = config['seq_len']
        
        # Create tensors
        q = torch.randn(config['batch'], config['heads'], seq_len, config['head_dim'], 
                       device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Normalize
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        
        for kernel_name, kernel_func in kernels.items():
            metrics = profile_kernel(kernel_name, kernel_func, q, k, v, causal=True)
            
            if metrics['time_ms'] is not None:
                results[kernel_name]['seq_lens'].append(seq_len)
                results[kernel_name]['times'].append(metrics['time_ms'])
                results[kernel_name]['bandwidth'].append(metrics['bandwidth_gbps'])
                results[kernel_name]['efficiency'].append(metrics['efficiency'])
                
                # Bandwidth per sequence length (to see scaling)
                bw_per_seq = metrics['bandwidth_gbps'] / seq_len
                
                print(f"{seq_len:<8} {kernel_name:<20} {metrics['time_ms']:<12.2f} "
                      f"{metrics['bandwidth_gbps']:<12.2f} {metrics['efficiency']:<12.2f} "
                      f"{metrics['memory_gb']:<12.4f} {bw_per_seq:<12.4f}")
            else:
                print(f"{seq_len:<8} {kernel_name:<20} {'ERROR':<12} {'-':<12} {'-':<12} {'-':<12} {'-':<12}")
    
    print("\n" + "=" * 120)
    print("Analysis Summary:")
    print("-" * 120)
    
    # Find the sequence length where performance degrades
    for kernel_name in ["SYCL Auto", "SYCL Optimized", "SYCL XMX"]:
        if kernel_name in results and results[kernel_name]['times']:
            times = results[kernel_name]['times']
            seq_lens = results[kernel_name]['seq_lens']
            
            # Calculate time per element (should be constant for O(n²) algorithm)
            time_per_elem = [t / (s * s) for t, s in zip(times, seq_lens)]
            
            print(f"\n{kernel_name}:")
            print(f"  Time per element (ms/elem²):")
            for i, (seq, tpe) in enumerate(zip(seq_lens, time_per_elem)):
                print(f"    seq_len={seq}: {tpe:.6f}")
                if i > 0:
                    degradation = (tpe / time_per_elem[0] - 1) * 100
                    if degradation > 10:
                        print(f"      -> {degradation:.1f}% slower than seq_len={seq_lens[0]}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time vs sequence length
    for kernel_name in ["PyTorch", "SYCL Auto", "SYCL Optimized", "SYCL XMX"]:
        if kernel_name in results and results[kernel_name]['times']:
            ax1.plot(results[kernel_name]['seq_lens'], results[kernel_name]['times'], 
                    marker='o', label=kernel_name)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time vs Sequence Length')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # Bandwidth efficiency
    for kernel_name in ["SYCL Auto", "SYCL Optimized", "SYCL XMX"]:
        if kernel_name in results and results[kernel_name]['efficiency']:
            ax2.plot(results[kernel_name]['seq_lens'], results[kernel_name]['efficiency'], 
                    marker='o', label=kernel_name)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Bandwidth Efficiency (%)')
    ax2.set_title('Memory Bandwidth Efficiency')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xscale('log', base=2)
    ax2.axhline(y=80, color='r', linestyle='--', label='80% (Good)')
    ax2.axhline(y=50, color='orange', linestyle='--', label='50% (OK)')
    
    # Time per element squared (should be constant)
    for kernel_name in ["SYCL Auto", "SYCL Optimized", "SYCL XMX"]:
        if kernel_name in results and results[kernel_name]['times']:
            seq_lens = results[kernel_name]['seq_lens']
            times = results[kernel_name]['times']
            time_per_elem = [t / (s * s) for t, s in zip(times, seq_lens)]
            ax3.plot(seq_lens, time_per_elem, marker='o', label=kernel_name)
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Time per Element² (ms)')
    ax3.set_title('Normalized Time (Should be Constant for O(n²))')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xscale('log', base=2)
    
    # Speedup vs PyTorch
    pytorch_times = dict(zip(results['PyTorch']['seq_lens'], results['PyTorch']['times']))
    for kernel_name in ["SYCL Auto", "SYCL Optimized", "SYCL XMX"]:
        if kernel_name in results and results[kernel_name]['times']:
            speedups = []
            seq_lens = []
            for seq, time in zip(results[kernel_name]['seq_lens'], results[kernel_name]['times']):
                if seq in pytorch_times:
                    speedup = pytorch_times[seq] / time
                    speedups.append(speedup)
                    seq_lens.append(seq)
            ax4.plot(seq_lens, speedups, marker='o', label=kernel_name)
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Speedup vs PyTorch')
    ax4.set_title('SYCL Kernel Speedup')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xscale('log', base=2)
    ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bandwidth_analysis.png', dpi=150)
    print(f"\nPlots saved to bandwidth_analysis.png")


if __name__ == "__main__":
    main()