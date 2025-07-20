#!/usr/bin/env python3
"""
Test script for Intel GPU Ring Flash Attention implementation
"""

import torch
import torch.distributed as dist
import os
import sys

try:
    import intel_extension_for_pytorch as ipex
    print(f"Intel Extension for PyTorch version: {ipex.__version__}")
except ImportError:
    print("Intel Extension for PyTorch not available")
    sys.exit(1)

# Test basic Intel GPU functionality
def test_intel_gpu_basic():
    """Test basic Intel GPU/XPU functionality"""
    print("\n=== Testing Intel GPU Basic Functionality ===")
    
    if not hasattr(torch, 'xpu'):
        print("❌ XPU not available in PyTorch")
        return False
        
    if not torch.xpu.is_available():
        print("❌ Intel GPU (XPU) not available")
        return False
    
    print(f"✅ Intel GPU (XPU) available")
    print(f"✅ XPU device count: {torch.xpu.device_count()}")
    
    # Test tensor operations on XPU
    try:
        x = torch.randn(4, 4).to('xpu')
        y = torch.randn(4, 4).to('xpu')
        z = torch.matmul(x, y)
        print(f"✅ Basic XPU tensor operations work")
        print(f"✅ Result device: {z.device}")
        return True
    except Exception as e:
        print(f"❌ XPU tensor operations failed: {e}")
        return False


def test_ring_flash_attn_import():
    """Test importing ring flash attention with Intel backend"""
    print("\n=== Testing Ring Flash Attention Import ===")
    
    try:
        import ring_flash_attn
        print("✅ Ring Flash Attention imported successfully")
        
        # Check if Intel backend is detected
        if hasattr(ring_flash_attn, 'BACKEND'):
            print(f"✅ Detected backend: {ring_flash_attn.BACKEND}")
            return ring_flash_attn.BACKEND == 'intel'
        else:
            print("⚠️  Backend detection not available")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import ring_flash_attn: {e}")
        return False


def test_ring_flash_attn_basic():
    """Test basic ring flash attention functionality"""
    print("\n=== Testing Ring Flash Attention Basic Functionality ===")
    
    if not torch.xpu.is_available():
        print("❌ XPU not available, skipping test")
        return False
    
    try:
        from ring_flash_attn import ring_flash_attn_func
        
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        # Create test tensors on XPU
        device = 'xpu'
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        print(f"✅ Created test tensors on {device}")
        print(f"✅ Tensor shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # Test without distributed (single process)
        try:
            output = ring_flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                causal=True,
                group=None  # No process group for single process
            )
            print(f"✅ Ring Flash Attention executed successfully")
            print(f"✅ Output shape: {output.shape}")
            print(f"✅ Output device: {output.device}")
            return True
            
        except Exception as e:
            print(f"❌ Ring Flash Attention execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scaled_dot_product_attention():
    """Test Intel's scaled dot product attention directly"""
    print("\n=== Testing Intel Scaled Dot Product Attention ===")
    
    if not torch.xpu.is_available():
        print("❌ XPU not available, skipping test")
        return False
    
    try:
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        device = 'xpu'
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        with torch.backends.xpu.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        
        print(f"✅ Intel SDP Attention executed successfully")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Output device: {output.device}")
        return True
        
    except Exception as e:
        print(f"❌ Intel SDP Attention failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 Testing Intel GPU Ring Flash Attention Implementation")
    print("=" * 60)
    
    tests = [
        test_intel_gpu_basic,
        test_scaled_dot_product_attention,
        test_ring_flash_attn_import,
        test_ring_flash_attn_basic,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Intel GPU Ring Flash Attention is working!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())