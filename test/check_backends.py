#!/usr/bin/env python3
import torch
import torch.distributed as dist

print("Available distributed backends:")
print(f"- is_available(): {dist.is_available()}")

if hasattr(dist, 'is_gloo_available'):
    print(f"- Gloo: {dist.is_gloo_available()}")
if hasattr(dist, 'is_nccl_available'):
    print(f"- NCCL: {dist.is_nccl_available()}")
if hasattr(dist, 'is_mpi_available'):
    print(f"- MPI: {dist.is_mpi_available()}")

# Check for oneCCL
try:
    import oneccl_bindings_for_pytorch
    print("- oneCCL bindings: Available")
except ImportError:
    print("- oneCCL bindings: Not available")

# Check Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    print(f"- Intel Extension for PyTorch: {ipex.__version__}")
    
    # Check if CCL is available through IPEX
    if hasattr(ipex, 'xpu') and hasattr(ipex.xpu, 'is_oneccl_available'):
        print(f"- IPEX oneCCL support: {ipex.xpu.is_oneccl_available()}")
except ImportError:
    print("- Intel Extension for PyTorch: Not available")

# List all available backends
print("\nRegistered backends:")
if hasattr(dist, 'Backend'):
    for attr in dir(dist.Backend):
        if not attr.startswith('_') and attr.isupper():
            print(f"  - {attr}: {getattr(dist.Backend, attr)}")

# Check environment
import os
print("\nRelevant environment variables:")
for key in os.environ:
    if 'CCL' in key or 'MPI' in key:
        print(f"  {key}={os.environ[key]}")