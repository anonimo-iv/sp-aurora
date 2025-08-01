#!/usr/bin/env python3
"""
Test various workarounds for CCL all-to-all on XPU
"""

import os
import sys
import torch
import torch.distributed as dist
from mpi4py import MPI

# Try different CCL environment settings
ccl_configs = [
    {
        "name": "Default CCL",
        "env": {}
    },
    {
        "name": "CCL with experimental features",
        "env": {
            "CCL_ENABLE_EXPERIMENTAL": "1",
            "CCL_GPU_DEVICES": "all",
            "CCL_USE_DEVICE_MEM": "1"
        }
    },
    {
        "name": "CCL with Ze backend forced",
        "env": {
            "CCL_BACKEND": "ze",
            "CCL_ENABLE_ZE": "1",
            "CCL_ZE_ENABLE": "1"
        }
    },
    {
        "name": "CCL with CPU fallback",
        "env": {
            "CCL_ALLTOALL_DEVICE": "cpu",
            "CCL_USE_DEVICE_MEM": "0"
        }
    }
]

def test_alltoall_with_config(config_name, env_vars):
    """Test all-to-all with specific CCL configuration"""
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    print(f"\n[Rank {rank}] Testing: {config_name}", flush=True)
    
    # Apply environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"[Rank {rank}] Set {key}={value}", flush=True)
    
    # Import oneccl after setting env vars
    import oneccl_bindings_for_pytorch
    
    # Initialize process group
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    if size > 1 and rank == 0:
        import socket
        os.environ['MASTER_ADDR'] = socket.gethostname()
    
    if size > 1:
        master_addr = MPI.COMM_WORLD.bcast(os.environ['MASTER_ADDR'], root=0)
        os.environ['MASTER_ADDR'] = master_addr
    
    try:
        dist.init_process_group(backend='ccl', init_method='env://')
        print(f"[Rank {rank}] Process group initialized", flush=True)
        
        # Test on XPU
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device(f'xpu:{rank}')
            tensor = torch.ones(4, device=device)
            
            try:
                dist.all_to_all_single(tensor, tensor)
                print(f"[Rank {rank}] ✓ XPU all-to-all succeeded!", flush=True)
                return True
            except Exception as e:
                print(f"[Rank {rank}] ✗ XPU all-to-all failed: {e}", flush=True)
                return False
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ Process group init failed: {e}", flush=True)
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    
    return False

def main():
    rank = MPI.COMM_WORLD.Get_rank()
    
    print(f"[Rank {rank}] Testing CCL all-to-all workarounds", flush=True)
    
    # Test 1: Check if we can use all_to_all (not all_to_all_single)
    print(f"\n[Rank {rank}] Test 1: Using all_to_all instead of all_to_all_single", flush=True)
    
    import oneccl_bindings_for_pytorch
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(MPI.COMM_WORLD.Get_size())
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    
    if MPI.COMM_WORLD.Get_size() > 1 and rank == 0:
        import socket
        os.environ['MASTER_ADDR'] = socket.gethostname()
    
    if MPI.COMM_WORLD.Get_size() > 1:
        master_addr = MPI.COMM_WORLD.bcast(os.environ['MASTER_ADDR'], root=0)
        os.environ['MASTER_ADDR'] = master_addr
    
    try:
        dist.init_process_group(backend='ccl', init_method='env://')
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device(f'xpu:{rank}')
            
            # Try all_to_all with list of tensors
            tensors_in = [torch.ones(2, 2, device=device) for _ in range(MPI.COMM_WORLD.Get_size())]
            tensors_out = [torch.zeros(2, 2, device=device) for _ in range(MPI.COMM_WORLD.Get_size())]
            
            try:
                dist.all_to_all(tensors_out, tensors_in)
                print(f"[Rank {rank}] ✓ all_to_all (list version) succeeded!", flush=True)
            except Exception as e:
                print(f"[Rank {rank}] ✗ all_to_all (list version) failed: {e}", flush=True)
        
        dist.destroy_process_group()
    except Exception as e:
        print(f"[Rank {rank}] ✗ Test 1 failed: {e}", flush=True)
    
    # Test different configurations
    for config in ccl_configs:
        success = test_alltoall_with_config(config["name"], config["env"])
        if success:
            print(f"[Rank {rank}] ✅ Found working configuration: {config['name']}", flush=True)
            break
    
    print(f"\n[Rank {rank}] All tests completed", flush=True)

if __name__ == "__main__":
    main()