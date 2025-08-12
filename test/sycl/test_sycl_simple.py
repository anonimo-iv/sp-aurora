#!/usr/bin/env python3
"""
Simple test to isolate SYCL Flash Attention numerical issues
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_aurora.intel_flash_attn_sycl import intel_flash_attn_forward_sycl
from sp_aurora.intel_flash_attn import intel_flash_attn_forward

def test_small_case():
    """Test with very small tensors to isolate issues"""
    device = 'xpu'
    
    # Very small test case
    batch = 1
    heads = 1
    seq_len = 4
    head_dim = 4
    
    # Create simple test tensors
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    # Normalize to prevent overflow
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    
    print("Input shapes:")
    print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    print(f"\nQ values:\n{q.squeeze()}")
    print(f"\nK values:\n{k.squeeze()}")
    print(f"\nV values:\n{v.squeeze()}")
    
    # Test without causal mask first
    print("\n=== Test 1: No causal mask ===")
    ref_out, ref_lse = intel_flash_attn_forward(q, k, v, causal=False)
    sycl_out, sycl_lse = intel_flash_attn_forward_sycl(q, k, v, causal=False)
    
    print(f"\nReference output:\n{ref_out.squeeze()}")
    print(f"\nSYCL output:\n{sycl_out.squeeze()}")
    print(f"\nDifference:\n{(ref_out - sycl_out).squeeze()}")
    print(f"\nMax difference: {torch.abs(ref_out - sycl_out).max().item():.6e}")
    
    print(f"\nReference LSE: {ref_lse.squeeze()}")
    print(f"SYCL LSE: {sycl_lse.squeeze()}")
    print(f"LSE difference: {torch.abs(ref_lse - sycl_lse).max().item():.6e}")
    
    # Test with causal mask
    print("\n=== Test 2: With causal mask ===")
    ref_out_causal, ref_lse_causal = intel_flash_attn_forward(q, k, v, causal=True)
    sycl_out_causal, sycl_lse_causal = intel_flash_attn_forward_sycl(q, k, v, causal=True)
    
    print(f"\nReference output (causal):\n{ref_out_causal.squeeze()}")
    print(f"\nSYCL output (causal):\n{sycl_out_causal.squeeze()}")
    print(f"\nDifference:\n{(ref_out_causal - sycl_out_causal).squeeze()}")
    print(f"\nMax difference: {torch.abs(ref_out_causal - sycl_out_causal).max().item():.6e}")
    
    # Manual computation for verification
    print("\n=== Manual computation (no mask) ===")
    scale = head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    print(f"Scores:\n{scores.squeeze()}")
    
    probs = torch.softmax(scores, dim=-1)
    print(f"Softmax probs:\n{probs.squeeze()}")
    
    manual_out = torch.matmul(probs, v)
    print(f"Manual output:\n{manual_out.squeeze()}")
    print(f"Manual vs Reference diff: {torch.abs(manual_out - ref_out).max().item():.6e}")

if __name__ == "__main__":
    test_small_case()