#!/usr/bin/env python3
"""
Test the specific intel_all_to_all_4d function from Ulysses.

Usage:
    mpirun -n 2 python test_intel_alltoall_4d.py
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
    
    print(f"\n[Rank {rank}] Testing intel_all_to_all_4d function")
    print(f"[Rank {rank}] World size: {world_size}")
    
    # Setup distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    if rank == 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12358'
    
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
    
    # Import the function after dist init
    print(f"[Rank {rank}] Importing intel_all_to_all_4d...")
    from sp_aurora.intel_ulysses_attn import intel_all_to_all_4d
    print(f"[Rank {rank}] Import successful")
    
    # Test 1: Direct call to intel_all_to_all_4d
    print(f"\n[Rank {rank}] Test 1: Direct intel_all_to_all_4d")
    try:
        # Create 4D tensor matching Ulysses pattern
        batch_size = 2
        seq_len_per_rank = 4
        num_heads = 8
        head_dim = 64
        
        input_tensor = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim,
                                  device=device, dtype=torch.float16)
        
        print(f"[Rank {rank}] Input shape: {input_tensor.shape}")
        
        # Call the function
        output = intel_all_to_all_4d(input_tensor, scatter_idx=2, gather_idx=1)
        
        print(f"[Rank {rank}] ✓ intel_all_to_all_4d SUCCESS!")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        
        # Expected output shape: (batch, seq_len_total, num_heads/P, head_dim)
        expected_shape = (batch_size, seq_len_per_rank * world_size, num_heads // world_size, head_dim)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ intel_all_to_all_4d FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: With dummy all_reduce first
    print(f"\n[Rank {rank}] Test 2: intel_all_to_all_4d with dummy all_reduce")
    try:
        # Do dummy all_reduce first
        dummy = torch.tensor([1.0], device=device, dtype=torch.float32)
        dist.all_reduce(dummy)
        print(f"[Rank {rank}] Dummy all_reduce done")
        
        # Now try again
        input_tensor = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim,
                                  device=device, dtype=torch.float16)
        
        output = intel_all_to_all_4d(input_tensor, scatter_idx=2, gather_idx=1)
        
        print(f"[Rank {rank}] ✓ intel_all_to_all_4d after dummy all_reduce SUCCESS!")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ intel_all_to_all_4d after dummy failed: {e}")
    
    # Test 3: Backward pattern
    print(f"\n[Rank {rank}] Test 3: intel_all_to_all_4d backward pattern")
    try:
        # Input for backward: (batch, seq_len_total, num_heads/P, head_dim)
        seq_len_total = seq_len_per_rank * world_size
        shard_heads = num_heads // world_size
        
        input_backward = torch.randn(batch_size, seq_len_total, shard_heads, head_dim,
                                    device=device, dtype=torch.float16)
        
        print(f"[Rank {rank}] Backward input shape: {input_backward.shape}")
        
        # Call with backward indices
        output_backward = intel_all_to_all_4d(input_backward, scatter_idx=1, gather_idx=2)
        
        print(f"[Rank {rank}] ✓ intel_all_to_all_4d backward SUCCESS!")
        print(f"[Rank {rank}] Backward output shape: {output_backward.shape}")
        
        # Expected: back to (batch, seq_len/P, num_heads, head_dim)
        expected_back = (batch_size, seq_len_per_rank, num_heads, head_dim)
        assert output_backward.shape == expected_back, f"Shape mismatch: {output_backward.shape} != {expected_back}"
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ intel_all_to_all_4d backward failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())