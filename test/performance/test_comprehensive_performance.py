#!/usr/bin/env python3
"""
Comprehensive performance test for all Flash Attention implementations
Tests PyTorch SDPA, PyTorch Baseline, all SYCL kernels, and oneDNN
Supports extended sequence lengths up to 8192
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ring_flash_attn.intel_flash_attn_sycl import (
    intel_flash_attn_forward_sycl, 
    is_sycl_available, 
    get_sycl_device_info
)
from ring_flash_attn.intel_flash_attn import intel_flash_attn_forward

def benchmark_kernel(kernel_func, q, k, v, causal, scale, kernel_name, warmup=3, num_runs=10):
    """Benchmark a single kernel implementation"""
    device = q.device
    
    # Skip if sequence length is too large for certain kernels
    seq_len = q.shape[2]
    if kernel_name in ["SYCL XMX", "SYCL Optimized V3", "SYCL Optimized V5"] and seq_len > 2048:
        return None, None, "Skipped (seq too long)"
    
    # Warmup
    try:
        for _ in range(warmup):
            if kernel_name == "PyTorch SDPA":
                # Use default SDPA settings (will select best kernel automatically)
                _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                   dropout_p=0.0, is_causal=causal, scale=scale)
            elif kernel_name == "PyTorch SDPA (math-only)":
                # Math-only mode for comparison
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                       dropout_p=0.0, is_causal=causal, scale=scale)
            elif kernel_name == "PyTorch Baseline":
                _ = intel_flash_attn_forward(q, k, v, causal=causal)
            elif kernel_name.startswith("SYCL"):
                kernel_type = kernel_name.replace("SYCL ", "").lower().replace(" ", "_")
                _ = intel_flash_attn_forward_sycl(q, k, v, causal=causal, 
                                                   softmax_scale=scale, kernel_type=kernel_type)
            else:  # oneDNN
                _ = intel_flash_attn_forward_sycl(q, k, v, causal=causal, 
                                                   softmax_scale=scale, kernel_type="onednn")
    except Exception as e:
        return None, None, f"Warmup failed: {str(e)[:50]}"
    
    # Timed runs
    try:
        torch.xpu.synchronize()
        start = time.time()
        
        for _ in range(num_runs):
            if kernel_name == "PyTorch SDPA":
                # Use default SDPA settings (will select best kernel automatically)
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                       dropout_p=0.0, is_causal=causal, scale=scale)
            elif kernel_name == "PyTorch SDPA (math-only)":
                # Math-only mode for comparison
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                           dropout_p=0.0, is_causal=causal, scale=scale)
            elif kernel_name == "PyTorch Baseline":
                output, _ = intel_flash_attn_forward(q, k, v, causal=causal)
            elif kernel_name.startswith("SYCL"):
                kernel_type = kernel_name.replace("SYCL ", "").lower().replace(" ", "_")
                output, _ = intel_flash_attn_forward_sycl(q, k, v, causal=causal, 
                                                          softmax_scale=scale, kernel_type=kernel_type)
            else:  # oneDNN
                output, _ = intel_flash_attn_forward_sycl(q, k, v, causal=causal, 
                                                          softmax_scale=scale, kernel_type="onednn")
        
        torch.xpu.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_runs * 1000  # ms
        return avg_time, output, "Success"
    except Exception as e:
        return None, None, f"Runtime failed: {str(e)[:50]}"

def calculate_metrics(time_ms, batch_size, num_heads, seq_len, head_dim):
    """Calculate bandwidth and GFLOPS"""
    if time_ms is None:
        return None, None
    
    # Memory access pattern for flash attention
    bytes_per_element = 4  # float32
    total_elements = batch_size * num_heads * seq_len * head_dim
    memory_accessed = 4 * total_elements * bytes_per_element  # Q, K, V, O
    
    # Bandwidth in GB/s
    bandwidth_gb_s = (memory_accessed / 1e9) / (time_ms / 1000)
    
    # FLOPs calculation for attention
    flops = batch_size * num_heads * seq_len * seq_len * head_dim * 4
    gflops = (flops / 1e9) / (time_ms / 1000)
    
    return bandwidth_gb_s, gflops

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Flash Attention performance test')
    parser.add_argument('--output', type=str, default='comprehensive_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance plots')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--num-heads', type=int, default=32,
                       help='Number of attention heads (default: 32)')
    parser.add_argument('--head-dim', type=int, default=128,
                       help='Head dimension (default: 128)')
    parser.add_argument('--normalize-inputs', action='store_true',
                       help='Normalize Q and K tensors to prevent numerical issues')
    parser.add_argument('--fresh-tensors', action='store_true',
                       help='Create fresh tensors for each implementation to avoid cache effects')
    
    args = parser.parse_args()
    
    print("Comprehensive Flash Attention Performance Test")
    print("=" * 80)
    
    # Check SYCL availability
    if not is_sycl_available():
        print("ERROR: SYCL Flash Attention not available!")
        return
    
    # Print device info
    print("Device Info:")
    device_info = get_sycl_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Test parameters
    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    causal = True
    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    # All implementations to test
    implementations = [
        "PyTorch SDPA",
        "PyTorch SDPA (math-only)",
        "PyTorch Baseline",
        "SYCL Auto-Select",
        "SYCL XMX",
        "SYCL Optimized V3",
        "SYCL Optimized V5",
        "oneDNN"
    ]
    
    print(f"Testing configurations:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Causal: {causal}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Implementations: {len(implementations)}")
    print(f"  Normalize inputs: {args.normalize_inputs}")
    print(f"  Fresh tensors: {args.fresh_tensors}")
    print()
    
    # Results storage
    results = []
    
    # Theoretical peak performance for Intel GPU Max 1550
    peak_gflops = 52000  # 52 TFLOPS FP32
    
    # Print header
    print(f"{'SeqLen':<8} {'Implementation':<20} {'Time (ms)':<12} {'BW (GB/s)':<12} "
          f"{'GFLOPS':<12} {'% Peak':<8} {'vs PyTorch':<12} {'Status':<20}")
    print("-" * 120)
    
    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 120)
        
        # Create base tensors
        q_base = torch.randn(batch_size, num_heads, seq_len, head_dim, device='xpu', dtype=torch.float32)
        k_base = torch.randn(batch_size, num_heads, seq_len, head_dim, device='xpu', dtype=torch.float32)
        v_base = torch.randn(batch_size, num_heads, seq_len, head_dim, device='xpu', dtype=torch.float32)
        
        # Normalize if requested
        if args.normalize_inputs:
            q_base = q_base / q_base.norm(dim=-1, keepdim=True)
            k_base = k_base / k_base.norm(dim=-1, keepdim=True)
        
        scale = 1.0 / np.sqrt(head_dim)
        
        # Store PyTorch SDPA time for comparison
        pytorch_time = None
        
        for impl in implementations:
            # Create fresh tensors if requested
            if args.fresh_tensors:
                q = q_base.clone()
                k = k_base.clone()
                v = v_base.clone()
            else:
                q = q_base
                k = k_base
                v = v_base
            
            time_ms, output, status = benchmark_kernel(None, q, k, v, causal, scale, impl)
            
            if time_ms is not None:
                bw_gbps, gflops = calculate_metrics(time_ms, batch_size, num_heads, seq_len, head_dim)
                peak_pct = (gflops / peak_gflops) * 100 if gflops else 0
                
                # Calculate speedup vs PyTorch SDPA
                if impl == "PyTorch SDPA":
                    pytorch_time = time_ms
                    speedup = 1.0
                else:
                    speedup = pytorch_time / time_ms if pytorch_time and time_ms else None
                
                # Store results
                result = {
                    'seq_len': seq_len,
                    'implementation': impl,
                    'time_ms': time_ms,
                    'bandwidth_gbps': bw_gbps,
                    'gflops': gflops,
                    'peak_pct': peak_pct,
                    'speedup_vs_pytorch': speedup,
                    'status': status
                }
                results.append(result)
                
                # Print results
                speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
                print(f"{seq_len:<8} {impl:<20} {time_ms:<12.2f} {bw_gbps:<12.2f} "
                      f"{gflops:<12.2f} {peak_pct:<8.2f} {speedup_str:<12} {status:<20}")
            else:
                # Store failed results
                result = {
                    'seq_len': seq_len,
                    'implementation': impl,
                    'time_ms': None,
                    'bandwidth_gbps': None,
                    'gflops': None,
                    'peak_pct': None,
                    'speedup_vs_pytorch': None,
                    'status': status
                }
                results.append(result)
                
                print(f"{seq_len:<8} {impl:<20} {'N/A':<12} {'N/A':<12} "
                      f"{'N/A':<12} {'N/A':<8} {'N/A':<12} {status:<20}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Best implementation for each sequence length
    print("\nBest Implementation by Sequence Length:")
    print(f"{'SeqLen':<8} {'Best Implementation':<20} {'Time (ms)':<12} {'Speedup vs PyTorch':<18}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        seq_results = df[(df['seq_len'] == seq_len) & (df['time_ms'].notna())]
        if not seq_results.empty:
            best = seq_results.loc[seq_results['time_ms'].idxmin()]
            speedup = best['speedup_vs_pytorch']
            speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
            print(f"{seq_len:<8} {best['implementation']:<20} {best['time_ms']:<12.2f} {speedup_str:<18}")
    
    # Average performance by implementation
    print("\nAverage Performance by Implementation (excluding failures):")
    print(f"{'Implementation':<20} {'Avg Time (ms)':<15} {'Avg GFLOPS':<15} {'Avg Speedup':<15}")
    print("-" * 65)
    
    for impl in implementations:
        impl_results = df[(df['implementation'] == impl) & (df['time_ms'].notna())]
        if not impl_results.empty:
            avg_time = impl_results['time_ms'].mean()
            avg_gflops = impl_results['gflops'].mean()
            avg_speedup = impl_results['speedup_vs_pytorch'].mean()
            print(f"{impl:<20} {avg_time:<15.2f} {avg_gflops:<15.2f} {avg_speedup:<15.2f}")
    
    # Generate plots if requested
    if args.plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Time vs Sequence Length
        for impl in implementations:
            impl_data = df[(df['implementation'] == impl) & (df['time_ms'].notna())]
            if not impl_data.empty:
                ax1.plot(impl_data['seq_len'], impl_data['time_ms'], 
                        marker='o', label=impl, linewidth=2)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Execution Time vs Sequence Length')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        
        # Plot 2: Speedup vs PyTorch
        for impl in implementations[1:]:  # Skip PyTorch SDPA itself
            impl_data = df[(df['implementation'] == impl) & (df['speedup_vs_pytorch'].notna())]
            if not impl_data.empty:
                ax2.plot(impl_data['seq_len'], impl_data['speedup_vs_pytorch'], 
                        marker='o', label=impl, linewidth=2)
        ax2.axhline(y=1.0, color='red', linestyle='--', label='PyTorch SDPA baseline')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Speedup vs PyTorch SDPA')
        ax2.set_title('Relative Performance vs PyTorch SDPA')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # Plot 3: GFLOPS
        for impl in implementations:
            impl_data = df[(df['implementation'] == impl) & (df['gflops'].notna())]
            if not impl_data.empty:
                ax3.plot(impl_data['seq_len'], impl_data['gflops'], 
                        marker='o', label=impl, linewidth=2)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('GFLOPS')
        ax3.set_title('Achieved GFLOPS vs Sequence Length')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        # Plot 4: Efficiency (% of Peak)
        for impl in implementations:
            impl_data = df[(df['implementation'] == impl) & (df['peak_pct'].notna())]
            if not impl_data.empty:
                ax4.plot(impl_data['seq_len'], impl_data['peak_pct'], 
                        marker='o', label=impl, linewidth=2)
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('% of Theoretical Peak')
        ax4.set_title('Efficiency vs Sequence Length')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        
        plt.tight_layout()
        plot_filename = f'comprehensive_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to {plot_filename}")

if __name__ == "__main__":
    main()