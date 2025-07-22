#!/usr/bin/env python3
"""
Minimal test to isolate CCL/MPI initialization issue
"""

import os
import sys

# Set environment before any torch import
os.environ['USE_CCL'] = '1'

import torch
from mpi4py import MPI

# Import Intel Extension for PyTorch first
try:
    import intel_extension_for_pytorch as ipex
    print(f"✅ Intel Extension for PyTorch imported: {ipex.__version__}")
except ImportError as e:
    print(f"❌ Failed to import Intel Extension for PyTorch: {e}")
    sys.exit(1)

# Import oneCCL bindings to register the backend
try:
    import oneccl_bindings_for_pytorch
    print("✅ oneCCL bindings imported successfully")
except ImportError as e:
    print(f"❌ Failed to import oneCCL bindings: {e}")
    sys.exit(1)

import torch.distributed as dist
import time

def main():
    print(f"Process PID: {os.getpid()}")
    
    # Check Intel GPU
    if not torch.xpu.is_available():
        print("❌ Intel GPU not available")
        return 1
    
    print(f"✅ Intel GPU available: {torch.xpu.device_count()} devices")
    
    # Try to initialize CCL backend
    try:
        print("Initializing CCL backend...")
        
        # Set minimal environment variables
        os.environ['CCL_LOG_LEVEL'] = 'debug'
        os.environ['CCL_PROCESS_LAUNCHER'] = 'none'
        
        # Try with xpu:ccl format as suggested by warning
        dist.init_process_group(backend='xpu:ccl')
        print("✅ CCL backend initialized successfully")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Rank: {rank}, World size: {world_size}")
        
        # Test basic communication
        print(f"[Rank {rank}] Creating test tensor on XPU")
        tensor = torch.ones(10, device='xpu') * rank
        print(f"[Rank {rank}] Tensor: {tensor}")
        
        print(f"[Rank {rank}] Testing all_reduce...")
        dist.all_reduce(tensor)
        print(f"[Rank {rank}] After all_reduce: {tensor}")
        
        # Test P2P communication
        if world_size == 2:
            print(f"[Rank {rank}] Testing P2P communication...")
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1) % world_size
            
            send_tensor = torch.ones(5, device='xpu') * rank
            recv_tensor = torch.empty(5, device='xpu')
            
            print(f"[Rank {rank}] Sending to {send_rank}, receiving from {recv_rank}")
            
            if rank == 0:
                dist.send(send_tensor, dst=send_rank)
                dist.recv(recv_tensor, src=recv_rank)
            else:
                dist.recv(recv_tensor, src=recv_rank)
                dist.send(send_tensor, dst=send_rank)
            
            print(f"[Rank {rank}] Received: {recv_tensor}")
        
        print(f"[Rank {rank}] ✅ All tests passed")
        
        # Cleanup
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())