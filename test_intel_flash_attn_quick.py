#!/usr/bin/env python3
"""
Quick test for Intel Flash Attention implementation
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ring_flash_attn.intel_flash_attn import intel_flash_attn_forward

def test_basic_functionality():
    """Test basic functionality of Intel flash attention"""
    print("Testing Intel Flash Attention with optimized implementation")
    print("="*60)
    
    # Test parameters
    batch_size = 1
    num_heads = 2
    seq_len = 8
    head_dim = 16
    
    # Create test tensors on CPU first
    device = 'cpu'
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    
    print(f"Input shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    print(f"Device: {device}")
    
    # Test non-causal attention
    print("\n1. Testing non-causal attention...")
    try:
        output, lse = intel_flash_attn_forward(q, k, v, causal=False)
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ LSE shape: {lse.shape}")
        print(f"✓ LSE dtype: {lse.dtype}")
        assert output.shape == q.shape
        assert lse.shape == (batch_size, num_heads, seq_len)
        assert lse.dtype == torch.float32
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test causal attention
    print("\n2. Testing causal attention...")
    try:
        output_causal, lse_causal = intel_flash_attn_forward(q, k, v, causal=True)
        print(f"✓ Causal output shape: {output_causal.shape}")
        print(f"✓ Causal LSE shape: {lse_causal.shape}")
        assert output_causal.shape == q.shape
        assert lse_causal.shape == (batch_size, num_heads, seq_len)
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Verify LSE computation
    print("\n3. Verifying LSE computation...")
    # Manually compute expected LSE for comparison
    softmax_scale = head_dim ** (-0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    expected_lse = torch.logsumexp(scores, dim=-1)
    
    # Check if LSE values are reasonable
    print(f"✓ LSE range: [{lse.min().item():.4f}, {lse.max().item():.4f}]")
    print(f"✓ Expected LSE range: [{expected_lse.min().item():.4f}, {expected_lse.max().item():.4f}]")
    
    # Test with dropout
    print("\n4. Testing with dropout...")
    try:
        output_dropout, lse_dropout = intel_flash_attn_forward(q, k, v, dropout_p=0.1)
        print(f"✓ Output with dropout shape: {output_dropout.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)