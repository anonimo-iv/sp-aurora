#!/usr/bin/env python3
"""
Test and benchmark oneDNN Flash Attention implementation
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from sp_aurora.intel_flash_attn_sycl import intel_flash_attn_forward_sycl, is_sycl_available, get_sycl_device_info

def benchmark_kernel(kernel_func, q, k, v, causal, scale, kernel_name, warmup=3, num_runs=10):
    """Benchmark a single kernel implementation"""
    
    # Warmup
    for _ in range(warmup):
        if kernel_name == "PyTorch SDPA":
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                   dropout_p=0.0, is_causal=causal, scale=scale)
        else:
            _ = kernel_func(q, k, v, causal=causal, softmax_scale=scale, kernel_type=kernel_name)
    
    # Timed runs
    torch.xpu.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        if kernel_name == "PyTorch SDPA":
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                       dropout_p=0.0, is_causal=causal, scale=scale)
        else:
            output, _ = kernel_func(q, k, v, causal=causal, softmax_scale=scale, kernel_type=kernel_name)
    
    torch.xpu.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time, output

def calculate_metrics(time_ms, batch_size, num_heads, seq_len, head_dim):
    """Calculate bandwidth and GFLOPS"""
    # Memory access pattern for flash attention
    # Read Q, K, V once, write O once
    bytes_per_element = 4  # float32
    total_elements = batch_size * num_heads * seq_len * head_dim
    memory_accessed = 4 * total_elements * bytes_per_element  # Q, K, V, O
    
    # Bandwidth in GB/s
    bandwidth_gb_s = (memory_accessed / 1e9) / (time_ms / 1000)
    
    # FLOPs calculation for attention
    # Q @ K^T: 2 * batch * heads * seq_len * seq_len * head_dim
    # Softmax: ~5 * batch * heads * seq_len * seq_len
    # Attn @ V: 2 * batch * heads * seq_len * seq_len * head_dim
    flops = batch_size * num_heads * seq_len * seq_len * head_dim * 4
    gflops = (flops / 1e9) / (time_ms / 1000)
    
    return bandwidth_gb_s, gflops

def main():
    print("Testing oneDNN Flash Attention Implementation")
    print("=" * 60)
    
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
    batch_size = 1
    num_heads = 32
    head_dim = 128
    causal = True
    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    
    # Test if oneDNN kernel is available
    try:
        q_test = torch.randn(1, 1, 32, 128, device='xpu', dtype=torch.float32)
        k_test = torch.randn(1, 1, 32, 128, device='xpu', dtype=torch.float32)
        v_test = torch.randn(1, 1, 32, 128, device='xpu', dtype=torch.float32)
        _, _ = intel_flash_attn_forward_sycl(q_test, k_test, v_test, kernel_type="onednn")
        has_onednn = True
        print("✓ oneDNN kernel is available\n")
    except Exception as e:
        has_onednn = False
        print(f"✗ oneDNN kernel not available: {e}\n")
    
    print(f"Testing configurations:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Causal: {causal}")
    print(f"  Sequence lengths: {seq_lengths}")
    print()
    
    # Results table header
    print("Results:")
    print("-" * 90)
    print(f"{'SeqLen':<10} {'Kernel':<25} {'Time (ms)':<15} {'BW (GB/s)':<15} {'GFLOPS':<15} {'% Peak':<10}")
    print("-" * 90)
    
    # Theoretical peak performance for Intel GPU Max 1550
    peak_gflops = 52000  # 52 TFLOPS FP32
    
    for seq_len in seq_lengths:
        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='xpu', dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='xpu', dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='xpu', dtype=torch.float32)
        
        scale = 1.0 / np.sqrt(head_dim)
        
        # Test PyTorch baseline
        try:
            time_pytorch, out_pytorch = benchmark_kernel(None, q, k, v, causal, scale, "PyTorch SDPA")
            bw_pytorch, gflops_pytorch = calculate_metrics(time_pytorch, batch_size, num_heads, seq_len, head_dim)
            peak_pct_pytorch = (gflops_pytorch / peak_gflops) * 100
            print(f"{seq_len:<10} {'PyTorch SDPA':<25} {time_pytorch:<15.2f} {bw_pytorch:<15.2f} "
                  f"{gflops_pytorch:<15.2f} {peak_pct_pytorch:<10.2f}%")
        except:
            print(f"{seq_len:<10} {'PyTorch SDPA':<25} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
        
        # Test SYCL V8 (current best)
        try:
            time_v8, out_v8 = benchmark_kernel(intel_flash_attn_forward_sycl, q, k, v, causal, scale, "optimized_v8")
            bw_v8, gflops_v8 = calculate_metrics(time_v8, batch_size, num_heads, seq_len, head_dim)
            peak_pct_v8 = (gflops_v8 / peak_gflops) * 100
            print(f"{seq_len:<10} {'SYCL Optimized V8':<25} {time_v8:<15.2f} {bw_v8:<15.2f} "
                  f"{gflops_v8:<15.2f} {peak_pct_v8:<10.2f}%")
        except Exception as e:
            print(f"{seq_len:<10} {'SYCL Optimized V8':<25} Error: {str(e)}")
        
        # Test oneDNN kernel
        if has_onednn:
            try:
                time_onednn, out_onednn = benchmark_kernel(intel_flash_attn_forward_sycl, q, k, v, causal, scale, "onednn")
                bw_onednn, gflops_onednn = calculate_metrics(time_onednn, batch_size, num_heads, seq_len, head_dim)
                peak_pct_onednn = (gflops_onednn / peak_gflops) * 100
                print(f"{seq_len:<10} {'SYCL oneDNN':<25} {time_onednn:<15.2f} {bw_onednn:<15.2f} "
                      f"{gflops_onednn:<15.2f} {peak_pct_onednn:<10.2f}%")
                
                # Calculate speedup vs PyTorch
                if 'time_pytorch' in locals():
                    speedup = time_pytorch / time_onednn
                    print(f"{'':<10} {'  → Speedup vs PyTorch':<25} {speedup:.2f}x")
            except Exception as e:
                print(f"{seq_len:<10} {'SYCL oneDNN':<25} Error: {str(e)}")
        
        print()  # Empty line between sequence lengths
    
    print("-" * 90)
    
    # Verify correctness for one configuration
    print("\nCorrectness check (seq_len=256):")
    q = torch.randn(1, 1, 256, 128, device='xpu', dtype=torch.float32)
    k = torch.randn(1, 1, 256, 128, device='xpu', dtype=torch.float32) 
    v = torch.randn(1, 1, 256, 128, device='xpu', dtype=torch.float32)
    
    # PyTorch reference
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    
    # Test each implementation
    implementations = [("optimized_v8", "SYCL V8")]
    if has_onednn:
        implementations.append(("onednn", "oneDNN"))
    
    for kernel_type, name in implementations:
        try:
            out_test, _ = intel_flash_attn_forward_sycl(q, k, v, causal=causal, 
                                                        softmax_scale=scale, kernel_type=kernel_type)
            max_diff = torch.max(torch.abs(out_ref - out_test)).item()
            mean_diff = torch.mean(torch.abs(out_ref - out_test)).item()
            print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        except Exception as e:
            print(f"  {name}: Error - {str(e)}")

if __name__ == "__main__":
    main()