#!/usr/bin/env python3
"""
Check which backend PyTorch's scaled_dot_product_attention is using
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def check_sdpa_backends():
    """Check which SDPA backends are available and being used"""
    
    print("=== PyTorch SDPA Backend Information ===\n")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check device
    if torch.xpu.is_available():
        device = 'xpu'
        print(f"Device: Intel GPU (XPU)")
        print(f"XPU device: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"Device: CUDA GPU")
    else:
        device = 'cpu'
        print(f"Device: CPU")
    
    print("\n=== Available Backends ===")
    
    # Check which backends are available
    backends = {
        "MATH": SDPBackend.MATH,
        "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
        "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
    }
    
    for name, backend in backends.items():
        print(f"{name}: {backend}")
    
    print("\n=== Testing Backend Selection ===")
    
    # Create test tensors
    batch_size = 1
    num_heads = 16
    seq_len = 512
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    # Test default backend
    print("\n1. Default backend (no context manager):")
    try:
        with torch.no_grad():
            output = F.scaled_dot_product_attention(q, k, v)
        print("   Success - but can't determine which backend was used")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test each backend explicitly
    print("\n2. Testing each backend explicitly:")
    
    for name, backend in backends.items():
        print(f"\n   Testing {name}:")
        try:
            with sdpa_kernel(backends=[backend]):
                with torch.no_grad():
                    output = F.scaled_dot_product_attention(q, k, v)
            print(f"   ✓ {name} works!")
        except RuntimeError as e:
            if "No available kernel" in str(e):
                print(f"   ✗ {name} not available: No kernel for this backend")
            else:
                print(f"   ✗ {name} error: {e}")
        except Exception as e:
            print(f"   ✗ {name} error: {e}")
    
    # Test with different data types
    print("\n3. Testing with different data types:")
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"\n   Testing dtype: {dtype}")
        try:
            q_typed = q.to(dtype)
            k_typed = k.to(dtype)
            v_typed = v.to(dtype)
            
            with torch.no_grad():
                output = F.scaled_dot_product_attention(q_typed, k_typed, v_typed)
            print(f"   ✓ {dtype} works with default backend")
            
            # Try to force flash attention with this dtype
            try:
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    with torch.no_grad():
                        output = F.scaled_dot_product_attention(q_typed, k_typed, v_typed)
                print(f"   ✓ {dtype} works with FLASH_ATTENTION")
            except:
                print(f"   ✗ {dtype} doesn't work with FLASH_ATTENTION")
                
        except Exception as e:
            print(f"   ✗ {dtype} error: {e}")
    
    # Check backend hints
    print("\n4. Backend capability hints:")
    
    # Check if we can get any hints about which backend is preferred
    if hasattr(torch.backends, 'cuda'):
        print(f"   CUDA Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"   CUDA Mem Efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"   CUDA Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")
    
    # Additional Intel-specific checks
    print("\n5. Intel-specific information:")
    
    try:
        import intel_extension_for_pytorch as ipex
        print(f"   Intel Extension for PyTorch version: {ipex.__version__}")
        
        # Check if IPEX has its own SDPA
        if hasattr(ipex, 'llm') and hasattr(ipex.llm, 'functional'):
            print("   IPEX LLM functional module available")
            if hasattr(ipex.llm.functional, 'scaled_dot_product_attention'):
                print("   ✓ IPEX has scaled_dot_product_attention")
        
    except ImportError:
        print("   Intel Extension for PyTorch not installed")
    
    # Try to detect which backend is actually being used
    print("\n6. Trying to detect actual backend used:")
    
    # Method 1: Profile and look for kernel names
    if device == 'xpu':
        print("   On Intel GPU, likely using oneDNN primitives through math backend")
    
    # Method 2: Time different backends
    import time
    
    print("\n7. Performance comparison:")
    iterations = 10
    
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(q, k, v)
    
    # Time default
    if device == 'xpu':
        torch.xpu.synchronize()
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(q, k, v)
    if device == 'xpu':
        torch.xpu.synchronize()
    default_time = (time.time() - start) / iterations * 1000
    
    print(f"   Default backend: {default_time:.2f} ms")
    
    # Time math backend explicitly
    try:
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            if device == 'xpu':
                torch.xpu.synchronize()
            start = time.time()
            for _ in range(iterations):
                with torch.no_grad():
                    _ = F.scaled_dot_product_attention(q, k, v)
            if device == 'xpu':
                torch.xpu.synchronize()
            math_time = (time.time() - start) / iterations * 1000
        print(f"   MATH backend: {math_time:.2f} ms")
        
        if abs(default_time - math_time) < 0.1:  # Within 0.1ms
            print("\n   → Default is likely using MATH backend (times are very close)")
        
    except:
        print("   MATH backend timing failed")


if __name__ == "__main__":
    check_sdpa_backends()