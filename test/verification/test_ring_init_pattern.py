#!/usr/bin/env python3
"""
Test that mimics Ring's exact initialization pattern to see if it enables all-to-all.
This test follows the exact sequence Ring uses.

Usage:
    mpirun -n 2 python test_ring_init_pattern.py
"""

import torch
import torch.distributed as dist
import os
import sys
from mpi4py import MPI
import datetime

# Add parent directory to path (exactly like Ring test)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for Intel GPU support (exactly like Ring test)
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
    INTEL_GPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
except ImportError as e:
    print(f"Warning: Intel Extension not available: {e}")
    INTEL_GPU_AVAILABLE = False

# Import ring flash attention modules (like in test_ring.py)
from ring_flash_attn.intel_ring_flash_attn import intel_ring_flash_attn_func
from ring_flash_attn.intel_utils import IntelRingComm


def main():
    # Initialize MPI (exactly like Ring)
    mpi_comm = MPI.COMM_WORLD
    RANK = mpi_comm.Get_rank()
    WORLD_SIZE = mpi_comm.Get_size()
    
    print(f"[Rank {RANK}] Ring Initialization Pattern Test")
    print(f"[Rank {RANK}] World size: {WORLD_SIZE}")
    
    # Setup environment (exactly like Ring)
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    
    if RANK == 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
    
    if WORLD_SIZE > 1:
        master_addr = mpi_comm.bcast(os.environ.get('MASTER_ADDR'), root=0)
        master_port = mpi_comm.bcast(os.environ.get('MASTER_PORT'), root=0)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        mpi_comm.Barrier()
    
    # Set device (exactly like Ring)
    if INTEL_GPU_AVAILABLE:
        DEVICE = torch.device(f"xpu:{RANK % torch.xpu.device_count()}")
        torch.xpu.set_device(DEVICE)
        print(f"[Rank {RANK}] XPU device set: {DEVICE}")
    else:
        DEVICE = torch.device('cpu')
        print(f"[Rank {RANK}] Using CPU device")
    
    # Initialize process group (exactly like Ring)
    if WORLD_SIZE > 1:
        backend = "ccl" if INTEL_GPU_AVAILABLE else "gloo"
        print(f"[Rank {RANK}] Using backend: {backend}")
        
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=WORLD_SIZE,
            rank=RANK,
            timeout=datetime.timedelta(seconds=360)
        )
        
        if WORLD_SIZE > 1:
            mpi_comm.Barrier()
    
    # Now test the exact sequence Ring uses
    print(f"\n[Rank {RANK}] Testing Ring's initialization sequence...")
    
    try:
        # Step 1: Initialize communicators with broadcast (from test_ring.py test_ring_comm)
        dummy = torch.tensor([1.0], device=DEVICE, dtype=torch.float32)
        dist.broadcast(dummy, src=0)
        print(f"[Rank {RANK}] ✓ Broadcast succeeded, dummy={dummy.item()}")
        
        # Step 2: Initialize ring communicator
        comm = IntelRingComm(None)
        print(f"[Rank {RANK}] ✓ RingComm initialized")
        
        # Step 3: Test the dummy all_reduce that Ring does in forward pass
        dummy_tensor = torch.tensor([1.0], device='xpu' if INTEL_GPU_AVAILABLE else 'cpu')
        dist.all_reduce(dummy_tensor)
        print(f"[Rank {RANK}] ✓ Dummy all_reduce succeeded")
        
        # Step 4: Now test all_to_all
        print(f"\n[Rank {RANK}] Testing all_to_all after Ring initialization...")
        
        # Test with same tensor shapes as Ulysses
        batch_size = 2
        seq_len_per_rank = 4  
        num_heads = 8
        head_dim = 64
        
        # Create 4D tensor
        input_4d = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim,
                              device=DEVICE, dtype=torch.float16)
        
        # Reshape for all-to-all (following Ulysses pattern)
        shard_heads = num_heads // WORLD_SIZE
        input_reshaped = input_4d.reshape(batch_size, seq_len_per_rank, WORLD_SIZE, shard_heads, head_dim)
        input_t = input_reshaped.transpose(0, 2).contiguous()
        
        # Try all_to_all_single
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t)
        
        print(f"[Rank {RANK}] ✓ all_to_all_single SUCCEEDED after Ring init!")
        
        # Also test if isend/irecv work
        print(f"\n[Rank {RANK}] Testing isend/irecv...")
        test_tensor = torch.tensor([float(RANK)], device=DEVICE)
        next_rank = (RANK + 1) % WORLD_SIZE
        prev_rank = (RANK - 1) % WORLD_SIZE
        
        if RANK % 2 == 0:
            send_req = dist.isend(test_tensor, next_rank)
            recv_req = dist.irecv(test_tensor, prev_rank)
        else:
            recv_req = dist.irecv(test_tensor, prev_rank)
            send_req = dist.isend(test_tensor, next_rank)
            
        send_req.wait()
        recv_req.wait()
        print(f"[Rank {RANK}] ✓ isend/irecv SUCCEEDED!")
        
    except Exception as e:
        print(f"[Rank {RANK}] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())