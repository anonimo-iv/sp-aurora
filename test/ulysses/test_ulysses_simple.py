#!/usr/bin/env python3
"""
Simple test for Ulysses attention function.

Usage:
    mpirun -n 2 python test_ulysses_simple.py
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
    
    print(f"\n[Rank {rank}] Simple Ulysses Test")
    print(f"[Rank {rank}] World size: {world_size}")
    
    # Setup distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    if rank == 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12359'
    
    master_addr = mpi_comm.bcast(os.environ.get('MASTER_ADDR'), root=0)
    master_port = mpi_comm.bcast(os.environ.get('MASTER_PORT'), root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    mpi_comm.Barrier()
    
    # Set device
    if INTEL_GPU_AVAILABLE:
        device = torch.device(f'xpu:{rank}')
        torch.xpu.set_device(device)
        backend = 'ccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    
    print(f"[Rank {rank}] Device: {device}, Backend: {backend}")
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=60)
    )
    
    print(f"[Rank {rank}] Process group initialized")
    
    # Import Ulysses functions after dist init
    print(f"[Rank {rank}] Importing Ulysses functions...")
    from ring_flash_attn.intel_ulysses_attn import intel_ulysses_flash_attn_func
    print(f"[Rank {rank}] Import successful")
    
    # Test with small tensors
    print(f"\n[Rank {rank}] Testing Ulysses attention...")
    
    try:
        # Create small test tensors
        batch_size = 1
        seq_len_per_rank = 8  # Each rank has 8 tokens
        num_heads = 4
        head_dim = 64
        
        # Input shape: (batch, seq_len/P, num_heads, head_dim)
        q = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim,
                       device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        print(f"[Rank {rank}] Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # Call Ulysses attention
        output = intel_ulysses_flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=True,
            group=None  # Use default process group
        )
        
        print(f"[Rank {rank}] ✓ Ulysses attention SUCCESS!")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ Ulysses attention FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    dist.destroy_process_group()
    print(f"[Rank {rank}] Test completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())