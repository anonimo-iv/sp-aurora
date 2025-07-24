#!/usr/bin/env python3
"""Check environment for Intel GPU and MPI setup"""

import sys
import os

print("=== Environment Check ===")
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")

# Check Intel Extension
try:
    import intel_extension_for_pytorch as ipex
    print(f"✓ Intel Extension for PyTorch: {ipex.__version__}")
    if hasattr(torch, 'xpu'):
        print(f"  XPU available: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            print(f"  XPU device count: {torch.xpu.device_count()}")
    else:
        print("  XPU module not found in torch")
except ImportError as e:
    print(f"✗ Intel Extension not available: {e}")

# Check oneCCL bindings
try:
    import oneccl_bindings_for_pytorch
    print("✓ oneCCL bindings available")
except ImportError as e:
    print(f"✗ oneCCL bindings not available: {e}")

# Check MPI
try:
    from mpi4py import MPI
    print(f"✓ mpi4py available: {MPI.Get_version()}")
    comm = MPI.COMM_WORLD
    print(f"  MPI initialized: rank={comm.Get_rank()}, size={comm.Get_size()}")
except ImportError as e:
    print(f"✗ mpi4py not available: {e}")
except Exception as e:
    print(f"✗ MPI initialization failed: {e}")

# Check distributed
try:
    import torch.distributed as dist
    print("✓ torch.distributed available")
    if dist.is_available():
        print("  Distributed backend available")
        print(f"  Available backends: {[b for b in ['gloo', 'nccl', 'mpi', 'ccl'] if dist.is_backend_available(b)]}")
except Exception as e:
    print(f"✗ torch.distributed error: {e}")

# Check environment variables
print("\n=== Relevant Environment Variables ===")
env_vars = ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 
            'CCL_PROCESS_LAUNCHER', 'CCL_ATL_TRANSPORT', 'CCL_KVS_MODE',
            'LD_LIBRARY_PATH', 'PMI_RANK', 'PMI_SIZE']
for var in env_vars:
    value = os.environ.get(var, "NOT SET")
    if var == 'LD_LIBRARY_PATH' and value != "NOT SET":
        # Truncate long paths
        paths = value.split(':')
        if len(paths) > 3:
            value = ':'.join(paths[:2]) + f"... ({len(paths)} paths total)"
    print(f"  {var}: {value}")

print("\n=== Library Check ===")
# Check for missing libraries
import subprocess
try:
    result = subprocess.run(['ldd', sys.executable], capture_output=True, text=True)
    if 'not found' in result.stdout:
        print("⚠️  Missing libraries detected:")
        for line in result.stdout.split('\n'):
            if 'not found' in line:
                print(f"  {line.strip()}")
    else:
        print("✓ All libraries found for Python executable")
except:
    print("Could not check libraries with ldd")