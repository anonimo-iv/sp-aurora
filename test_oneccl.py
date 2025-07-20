#!/usr/bin/env python3
"""Test oneCCL functionality for Intel GPUs"""

import os
import sys

# Set environment variables for oneCCL
os.environ['CCL_ROOT'] = '/opt/aurora/24.347.0/oneapi/ccl/latest'
os.environ['I_MPI_ROOT'] = '/opt/aurora/24.347.0/oneapi/mpi/latest'

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Import oneCCL bindings
    import oneccl_bindings_for_pytorch
    print("✅ oneCCL bindings imported successfully")
    
    # Check distributed backends
    available_backends = []
    if torch.distributed.is_available():
        print("✅ PyTorch distributed is available")
        
        # Check for CCL backend
        if hasattr(torch.distributed, 'Backend'):
            if hasattr(torch.distributed.Backend, 'CCL'):
                available_backends.append('ccl')
            if hasattr(torch.distributed.Backend, 'NCCL'):
                available_backends.append('nccl')
            if hasattr(torch.distributed.Backend, 'GLOO'):
                available_backends.append('gloo')
        
        print(f"Available backends: {available_backends}")
    
    # Test Intel GPU
    try:
        import intel_extension_for_pytorch as ipex
        print(f"✅ Intel Extension for PyTorch: {ipex.__version__}")
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"✅ Intel GPU available, device count: {torch.xpu.device_count()}")
            
            # Test basic operation
            x = torch.randn(2, 2, device='xpu')
            y = torch.randn(2, 2, device='xpu')
            z = torch.matmul(x, y)
            print(f"✅ Basic operations work on XPU")
            
        else:
            print("❌ Intel GPU not available")
    
    except ImportError as e:
        print(f"❌ Intel Extension not available: {e}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()