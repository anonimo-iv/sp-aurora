#!/usr/bin/env python3
"""
Diagnostic script to identify where CCL hang occurs
"""

import sys
import os
import torch
import torch.distributed as dist
from mpi4py import MPI

print(f"[{MPI.COMM_WORLD.Get_rank()}] Script started", flush=True)

# Test 1: Basic imports
print(f"[{MPI.COMM_WORLD.Get_rank()}] Testing basic imports...", flush=True)
try:
    import intel_extension_for_pytorch as ipex
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✓ IPEX imported", flush=True)
except Exception as e:
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✗ IPEX import failed: {e}", flush=True)

try:
    import oneccl_bindings_for_pytorch
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✓ oneCCL bindings imported", flush=True)
except Exception as e:
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✗ oneCCL import failed: {e}", flush=True)

# Test 2: Device availability
print(f"[{MPI.COMM_WORLD.Get_rank()}] Checking XPU availability...", flush=True)
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✓ XPU available, count: {torch.xpu.device_count()}", flush=True)
else:
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✗ XPU not available", flush=True)

# Test 3: Initialize distributed with CCL
print(f"[{MPI.COMM_WORLD.Get_rank()}] Setting up distributed environment...", flush=True)
os.environ['RANK'] = str(MPI.COMM_WORLD.Get_rank())
os.environ['WORLD_SIZE'] = str(MPI.COMM_WORLD.Get_size())
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if MPI.COMM_WORLD.Get_size() > 1 and MPI.COMM_WORLD.Get_rank() == 0:
    import socket
    os.environ['MASTER_ADDR'] = socket.gethostname()

# Broadcast master address
if MPI.COMM_WORLD.Get_size() > 1:
    master_addr = MPI.COMM_WORLD.bcast(os.environ['MASTER_ADDR'], root=0)
    os.environ['MASTER_ADDR'] = master_addr

print(f"[{MPI.COMM_WORLD.Get_rank()}] Initializing process group with CCL...", flush=True)
try:
    dist.init_process_group(
        backend='ccl',
        init_method='env://',
        world_size=MPI.COMM_WORLD.Get_size(),
        rank=MPI.COMM_WORLD.Get_rank()
    )
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✓ Process group initialized", flush=True)
except Exception as e:
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✗ Process group init failed: {e}", flush=True)
    sys.exit(1)

# Test 4: Simple collective on CPU
print(f"[{MPI.COMM_WORLD.Get_rank()}] Testing CPU all-to-all...", flush=True)
cpu_tensor = torch.ones(4)
try:
    dist.all_to_all_single(cpu_tensor, cpu_tensor)
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✓ CPU all-to-all succeeded", flush=True)
except Exception as e:
    print(f"[{MPI.COMM_WORLD.Get_rank()}] ✗ CPU all-to-all failed: {e}", flush=True)

# Test 5: Simple collective on XPU
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f"[{MPI.COMM_WORLD.Get_rank()}] Testing XPU all-to-all...", flush=True)
    device = torch.device(f'xpu:{MPI.COMM_WORLD.Get_rank()}')
    xpu_tensor = torch.ones(4, device=device)
    try:
        dist.all_to_all_single(xpu_tensor, xpu_tensor)
        print(f"[{MPI.COMM_WORLD.Get_rank()}] ✓ XPU all-to-all succeeded", flush=True)
    except Exception as e:
        print(f"[{MPI.COMM_WORLD.Get_rank()}] ✗ XPU all-to-all failed: {e}", flush=True)

# Test 6: Larger tensor on XPU
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f"[{MPI.COMM_WORLD.Get_rank()}] Testing larger XPU tensor all-to-all...", flush=True)
    device = torch.device(f'xpu:{MPI.COMM_WORLD.Get_rank()}')
    large_tensor = torch.randn(1024, 1024, device=device)
    try:
        dist.all_to_all_single(large_tensor.flatten(), large_tensor.flatten())
        print(f"[{MPI.COMM_WORLD.Get_rank()}] ✓ Large XPU all-to-all succeeded", flush=True)
    except Exception as e:
        print(f"[{MPI.COMM_WORLD.Get_rank()}] ✗ Large XPU all-to-all failed: {e}", flush=True)

print(f"[{MPI.COMM_WORLD.Get_rank()}] All tests completed", flush=True)

# Cleanup
if dist.is_initialized():
    dist.destroy_process_group()

print(f"[{MPI.COMM_WORLD.Get_rank()}] Script finished", flush=True)