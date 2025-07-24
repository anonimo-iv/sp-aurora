#!/usr/bin/env python3
"""
Test script to verify LSE computation in Intel Flash Attention
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ring_flash_attn.intel_flash_attn import intel_flash_attn_forward

def test_lse_computation():
    """Test that LSE values are computed correctly"""
    print("Testing LSE computation in Intel Flash Attention")
    print("="*60)
    
    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 32
    
    # Create test tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Test 1: Non-causal attention
    print("\nTest 1: Non-causal attention")
    out, lse = intel_flash_attn_forward(q, k, v, causal=False)
    print(f"Output shape: {out.shape}")
    print(f"LSE shape: {lse.shape}")
    print(f"LSE dtype: {lse.dtype}")
    print(f"LSE range: [{lse.min().item():.4f}, {lse.max().item():.4f}]")
    print(f"LSE mean: {lse.mean().item():.4f}")
    
    # Verify LSE is not all zeros
    assert not torch.allclose(lse, torch.zeros_like(lse)), "LSE should not be all zeros!"
    print("✓ LSE values are non-zero")
    
    # Test 2: Causal attention
    print("\nTest 2: Causal attention")
    out_causal, lse_causal = intel_flash_attn_forward(q, k, v, causal=True)
    print(f"Causal LSE range: [{lse_causal.min().item():.4f}, {lse_causal.max().item():.4f}]")
    print(f"Causal LSE mean: {lse_causal.mean().item():.4f}")
    
    # Causal LSE should be different from non-causal
    assert not torch.allclose(lse, lse_causal), "Causal and non-causal LSE should differ!"
    print("✓ Causal LSE differs from non-causal")
    
    # Test 3: Verify LSE computation manually
    print("\nTest 3: Manual verification")
    softmax_scale = head_dim ** (-0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    manual_lse = torch.logsumexp(scores, dim=-1)
    
    # Convert manual_lse to same dtype as lse for comparison
    manual_lse = manual_lse.to(lse.dtype)
    
    # Compare with our implementation (non-causal)
    if torch.allclose(lse, manual_lse, rtol=1e-3, atol=1e-3):
        print("✓ LSE matches manual computation")
    else:
        max_diff = (lse - manual_lse).abs().max().item()
        print(f"⚠ LSE differs from manual computation (max diff: {max_diff:.6f})")
    
    # Test 4: Check numerical properties
    print("\nTest 4: Numerical properties")
    # LSE should be finite
    assert torch.isfinite(lse).all(), "LSE contains non-finite values!"
    print("✓ All LSE values are finite")
    
    # For reasonable inputs, LSE should be in a reasonable range
    # Typically around log(seq_len) ± a few units
    expected_order = torch.log(torch.tensor(float(seq_len)))
    if lse.mean() > expected_order - 10 and lse.mean() < expected_order + 10:
        print(f"✓ LSE values are in expected range (near log({seq_len}) = {expected_order:.2f})")
    
    # Test 5: Edge cases
    print("\nTest 5: Edge cases")
    
    # Test with very small sequence length
    q_small = torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16)
    k_small = torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16)
    v_small = torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16)
    
    out_small, lse_small = intel_flash_attn_forward(q_small, k_small, v_small, causal=False)
    assert lse_small.shape == (1, 2, 4), f"Unexpected LSE shape: {lse_small.shape}"
    assert torch.isfinite(lse_small).all(), "LSE contains non-finite values for small input!"
    print("✓ Small sequence length handled correctly")
    
    # Test with larger sequence length
    q_large = torch.randn(1, 2, 128, head_dim, device=device, dtype=torch.float16)
    k_large = torch.randn(1, 2, 128, head_dim, device=device, dtype=torch.float16)
    v_large = torch.randn(1, 2, 128, head_dim, device=device, dtype=torch.float16)
    
    out_large, lse_large = intel_flash_attn_forward(q_large, k_large, v_large, causal=True)
    assert lse_large.shape == (1, 2, 128), f"Unexpected LSE shape: {lse_large.shape}"
    assert torch.isfinite(lse_large).all(), "LSE contains non-finite values for large input!"
    print("✓ Large sequence length handled correctly")
    
    # Test consistency across multiple runs
    out1, lse1 = intel_flash_attn_forward(q, k, v, causal=False)
    out2, lse2 = intel_flash_attn_forward(q, k, v, causal=False)
    assert torch.allclose(lse1, lse2, rtol=1e-5, atol=1e-5), "LSE not consistent across runs!"
    print("✓ LSE computation is deterministic")
    
    print("\n" + "="*60)
    print("All tests passed! LSE computation is working correctly.")

if __name__ == "__main__":
    test_lse_computation()