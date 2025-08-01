#!/usr/bin/env python3
"""
Minimal test to reproduce Ulysses hang issue.

Usage:
    mpirun -n 2 python test_ulysses_minimal_hang.py
"""

import torch
import torch.distributed as dist
import os
import sys
from mpi4py import MPI
import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
    INTEL_GPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
except ImportError as e:
    print(f"Warning: Intel Extension not available: {e}")
    INTEL_GPU_AVAILABLE = False


def main():
    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()
    
    print(f"[Rank {rank}] Starting Ulysses minimal hang test")
    
    # Setup distributed (minimal, like test_intel_ulysses_attn.py)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Broadcast master address and port from rank 0
    if world_size > 1:
        if rank == 0:
            master_addr = 'localhost'  # Using localhost instead of socket.gethostname()
            master_port = 12360
        else:
            master_addr = None
            master_port = None
        
        master_addr = mpi_comm.bcast(master_addr, root=0)
        master_port = mpi_comm.bcast(master_port, root=0)
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
    
    print(f"[Rank {rank}] Before MPI barrier")
    mpi_comm.Barrier()
    print(f"[Rank {rank}] After MPI barrier")
    
    # Set device
    if INTEL_GPU_AVAILABLE:
        device = torch.device(f'xpu:{rank % torch.xpu.device_count()}')
        torch.xpu.set_device(device)
        backend = 'ccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    
    print(f"[Rank {rank}] Device: {device}, Backend: {backend}")
    
    # Initialize process group
    print(f"[Rank {rank}] Initializing process group...")
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=30)
    )
    print(f"[Rank {rank}] Process group initialized")
    
    # Try importing ring_flash_attn modules (like in original test)
    print(f"[Rank {rank}] Importing ring_flash_attn modules...")
    from ring_flash_attn import ulysses_flash_attn_func, ulysses_flash_attn_qkvpacked_func, ulysses_flash_attn_kvpacked_func
    from ring_flash_attn.intel_ulysses_attn import IntelSeqAllToAll4D, intel_all_to_all_4d
    print(f"[Rank {rank}] Imports successful")
    
    print(f"[Rank {rank}] Test completed without hang!")
    
    # Cleanup
    dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())