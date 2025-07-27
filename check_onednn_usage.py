#!/usr/bin/env python3
"""
Check if PyTorch uses oneDNN (formerly MKL-DNN) on Intel GPU
"""
import torch
import torch.nn.functional as F
import sys
import os

def check_onednn_usage():
    print("=== Checking oneDNN/MKL-DNN Usage in PyTorch ===\n")
    
    # 1. Check PyTorch build configuration
    print("1. PyTorch Build Information:")
    print(f"   PyTorch version: {torch.__version__}")
    
    # Check if PyTorch was built with MKL-DNN/oneDNN
    if hasattr(torch, '_C') and hasattr(torch._C, '_has_mkldnn'):
        print(f"   Built with MKL-DNN: {torch._C._has_mkldnn}")
    else:
        print("   MKL-DNN build flag not found")
    
    # Check MKL availability
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkl'):
        print(f"   MKL is available: {torch.backends.mkl.is_available()}")
    
    # Check for oneDNN specific functions
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkldnn'):
        print(f"   MKLDNN backend exists: True")
    
    # 2. Check environment variables
    print("\n2. Environment Variables:")
    onednn_vars = ['DNNL_VERBOSE', 'MKLDNN_VERBOSE', 'ONEDNN_VERBOSE']
    for var in onednn_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # 3. Try to enable verbose mode and run attention
    print("\n3. Testing with oneDNN verbose mode:")
    
    # Set verbose mode
    os.environ['DNNL_VERBOSE'] = '1'
    os.environ['MKLDNN_VERBOSE'] = '1'
    os.environ['ONEDNN_VERBOSE'] = '1'
    
    device = 'xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    # Create test tensors
    batch, heads, seq_len, head_dim = 1, 16, 128, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    print("\n   Running scaled_dot_product_attention...")
    print("   (If oneDNN is used, verbose output should appear below)")
    print("   " + "-" * 50)
    
    # Run attention - if oneDNN is used, verbose output should appear
    with torch.no_grad():
        output = F.scaled_dot_product_attention(q, k, v)
    
    print("   " + "-" * 50)
    
    # 4. Check Intel Extension for PyTorch
    print("\n4. Intel Extension for PyTorch (IPEX):")
    try:
        import intel_extension_for_pytorch as ipex
        print(f"   IPEX version: {ipex.__version__}")
        
        # Check if IPEX uses oneDNN
        if hasattr(ipex, '_C'):
            # Try to access build configuration
            if hasattr(ipex._C, 'get_onednn_version'):
                print(f"   oneDNN version: {ipex._C.get_onednn_version()}")
            
            # Check available backends
            if hasattr(ipex._C, 'is_onednn_available'):
                print(f"   oneDNN available in IPEX: {ipex._C.is_onednn_available()}")
        
        # Check for XPU backend info
        if hasattr(ipex, 'xpu'):
            print(f"   XPU backend available: True")
            if hasattr(ipex.xpu, 'get_device_name'):
                print(f"   XPU device: {ipex.xpu.get_device_name(0)}")
                
    except ImportError:
        print("   IPEX not installed")
    except Exception as e:
        print(f"   Error checking IPEX: {e}")
    
    # 5. Try to trace operations
    print("\n5. Operation Profiling:")
    
    # Try using PyTorch profiler
    try:
        from torch.profiler import profile, ProfilerActivity
        
        activities = []
        if device == 'xpu':
            # Note: XPU profiling might not be fully supported
            activities = [ProfilerActivity.CPU]
        elif device == 'cuda':
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        else:
            activities = [ProfilerActivity.CPU]
        
        with profile(activities=activities, record_shapes=True) as prof:
            with torch.no_grad():
                for _ in range(5):
                    output = F.scaled_dot_product_attention(q, k, v)
        
        print("\n   Profiler Key Averages:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        # Look for oneDNN/MKL kernels in the trace
        events = prof.events()
        onednn_events = [e for e in events if any(keyword in e.name.lower() 
                        for keyword in ['mkl', 'dnnl', 'onednn', 'mkldnn'])]
        
        if onednn_events:
            print("\n   Found oneDNN/MKL related events:")
            for event in onednn_events[:5]:  # Show first 5
                print(f"     - {event.name}")
        else:
            print("\n   No oneDNN/MKL events found in profiler trace")
            
    except Exception as e:
        print(f"   Profiling failed: {e}")
    
    # 6. Check for specific Intel optimizations
    print("\n6. Intel-specific Optimizations:")
    
    # Check if we can access low-level info
    if device == 'xpu':
        print("   On Intel XPU - checking for optimization paths...")
        
        # Try to check if specific Intel paths are available
        if hasattr(torch.ops, 'aten'):
            # Check for Intel-specific implementations
            aten_ops = dir(torch.ops.aten)
            intel_ops = [op for op in aten_ops if 'mkl' in op.lower() or 'dnnl' in op.lower()]
            
            if intel_ops:
                print(f"   Found {len(intel_ops)} Intel-specific operators")
                print("   Examples:", intel_ops[:3])
            else:
                print("   No explicit Intel-specific operators found")
    
    # Reset verbose mode
    os.environ.pop('DNNL_VERBOSE', None)
    os.environ.pop('MKLDNN_VERBOSE', None)
    os.environ.pop('ONEDNN_VERBOSE', None)

if __name__ == "__main__":
    check_onednn_usage()