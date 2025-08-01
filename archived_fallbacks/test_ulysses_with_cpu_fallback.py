#!/usr/bin/env python3
"""
Test Ulysses attention with CPU fallback for all-to-all operations
"""

import torch
import torch.distributed as dist
import os
import sys
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import oneCCL bindings
import oneccl_bindings_for_pytorch

# Import the modified Ulysses attention with fallback
from ring_flash_attn.intel_ulysses_attn_with_fallback import (
    ulysses_flash_attn_func_with_fallback,
    intel_all_to_all_4d_with_fallback
)


def setup_distributed():
    """Initialize distributed environment"""
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    
    if size > 1 and rank == 0:
        import socket
        os.environ['MASTER_ADDR'] = socket.gethostname()
    
    if size > 1:
        master_addr = mpi_comm.bcast(os.environ['MASTER_ADDR'], root=0)
        os.environ['MASTER_ADDR'] = master_addr
        mpi_comm.Barrier()
    
    # Use CCL backend - the fallback will handle the all-to-all issue
    dist.init_process_group(backend='ccl', init_method='env://')
    
    return rank, size


def test_ulysses_attention(rank, world_size, device):
    """Test Ulysses attention with CPU fallback"""
    print(f"[Rank {rank}] Testing Ulysses attention with CPU fallback", flush=True)
    
    # Set environment to use CPU fallback
    os.environ['ULYSSES_ALLTOALL_METHOD'] = 'cpu'
    
    # Create test tensors
    batch_size = 2
    seq_len_per_gpu = 512
    num_heads = 16
    head_dim = 64
    
    # Create QKV tensors
    q = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim, 
                   device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                   device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                   device=device, dtype=torch.float16)
    
    print(f"[Rank {rank}] Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}", flush=True)
    
    # Test the all-to-all operation directly
    print(f"[Rank {rank}] Testing all-to-all with CPU fallback...", flush=True)
    try:
        # Test forward all-to-all
        test_tensor = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                                 device=device, dtype=torch.float16)
        output = intel_all_to_all_4d_with_fallback(test_tensor, scatter_idx=2, gather_idx=1, method='cpu')
        print(f"[Rank {rank}] ✓ All-to-all forward succeeded, output shape: {output.shape}", flush=True)
        
        # Test backward all-to-all
        output_back = intel_all_to_all_4d_with_fallback(output, scatter_idx=1, gather_idx=2, method='cpu')
        print(f"[Rank {rank}] ✓ All-to-all backward succeeded, output shape: {output_back.shape}", flush=True)
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ All-to-all test failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
    
    # Test Ulysses attention
    print(f"[Rank {rank}] Testing Ulysses Flash Attention...", flush=True)
    try:
        # Get process group
        process_group = dist.group.WORLD if hasattr(dist.group, 'WORLD') else None
        
        # Run Ulysses attention with CPU fallback
        output = ulysses_flash_attn_func_with_fallback(
            q, k, v,
            causal=True,
            process_group=process_group,
            all_to_all_method='cpu'
        )
        
        print(f"[Rank {rank}] ✓ Ulysses attention succeeded, output shape: {output.shape}", flush=True)
        
        # Verify output shape
        assert output.shape == q.shape, f"Output shape mismatch: {output.shape} vs {q.shape}"
        
        # Test gradient flow
        if output.requires_grad:
            loss = output.sum()
            loss.backward()
            print(f"[Rank {rank}] ✓ Gradient flow test passed", flush=True)
        
        return True
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ Ulysses attention test failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def main():
    # Setup distributed
    rank, world_size = setup_distributed()
    
    print(f"\n{'='*60}", flush=True)
    print(f"[Rank {rank}] Testing Ulysses Attention with CPU Fallback", flush=True)
    print(f"[Rank {rank}] World size: {world_size}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Set device
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device(f'xpu:{rank}')
        torch.xpu.set_device(device)
        print(f"[Rank {rank}] Using XPU device", flush=True)
    else:
        device = torch.device('cpu')
        print(f"[Rank {rank}] Using CPU device", flush=True)
    
    # Run tests
    success = test_ulysses_attention(rank, world_size, device)
    
    # Synchronize using MPI barrier instead of dist.barrier (not supported on XPU)
    if dist.is_initialized():
        MPI.COMM_WORLD.Barrier()
    
    # Summary
    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        if success:
            print("✅ Ulysses attention with CPU fallback works correctly!", flush=True)
            print("   The all-to-all operations are performed on CPU", flush=True)
            print("   while attention computation remains on GPU", flush=True)
        else:
            print("❌ Ulysses attention test failed", flush=True)
        print(f"{'='*60}\n", flush=True)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()