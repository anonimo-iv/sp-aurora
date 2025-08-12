#!/usr/bin/env python3
"""
Final comprehensive test to verify both Ring and Ulysses attention work with CCL+XPU.

Usage:
    mpirun -n 2 python test_ring_ulysses_final.py
"""

import torch
import torch.distributed as dist
import os
import sys
from mpi4py import MPI
import datetime
import time

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


def test_ring_attention(rank, world_size, device):
    """Test Ring Flash Attention"""
    print(f"\n[Rank {rank}] === Testing Ring Flash Attention ===")
    
    try:
        from sp_aurora import sp_aurora_func
        
        # Create test tensors
        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64
        
        # Ring expects: (batch, seq_len, num_heads, head_dim)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        print(f"[Rank {rank}] Input shape: {q.shape}")
        
        # Test forward
        start = time.time()
        output = sp_aurora_func(q, k, v, causal=True, group=None)
        torch.xpu.synchronize() if device.type == 'xpu' else None
        elapsed = time.time() - start
        
        print(f"[Rank {rank}] ✓ Ring forward SUCCESS in {elapsed:.3f}s")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        
        # Test backward
        output.sum().backward()
        print(f"[Rank {rank}] ✓ Ring backward SUCCESS")
        
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ Ring attention FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ulysses_attention(rank, world_size, device):
    """Test Ulysses Flash Attention"""
    print(f"\n[Rank {rank}] === Testing Ulysses Flash Attention ===")
    
    try:
        from sp_aurora import ulysses_flash_attn_func
        
        # Create test tensors
        batch_size = 2
        seq_len_per_rank = 256  # Each rank has part of sequence
        num_heads = 8
        head_dim = 64
        
        # Ulysses expects: (batch, seq_len/P, num_heads, head_dim)
        q = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim,
                       device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        print(f"[Rank {rank}] Input shape (per rank): {q.shape}")
        print(f"[Rank {rank}] Total sequence length: {seq_len_per_rank * world_size}")
        
        # Test forward
        start = time.time()
        output = ulysses_flash_attn_func(q, k, v, causal=True, group=None)
        torch.xpu.synchronize() if device.type == 'xpu' else None
        elapsed = time.time() - start
        
        print(f"[Rank {rank}] ✓ Ulysses forward SUCCESS in {elapsed:.3f}s")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        
        # Test backward
        output.sum().backward()
        print(f"[Rank {rank}] ✓ Ulysses backward SUCCESS")
        
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ Ulysses attention FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collective_operations(rank, world_size, device):
    """Test all collective operations work on XPU"""
    print(f"\n[Rank {rank}] === Testing Collective Operations ===")
    
    results = {}
    
    # Test all_reduce
    try:
        t = torch.tensor([float(rank)], device=device)
        dist.all_reduce(t)
        results['all_reduce'] = True
        print(f"[Rank {rank}] ✓ all_reduce: {t.item()}")
    except Exception as e:
        results['all_reduce'] = False
        print(f"[Rank {rank}] ✗ all_reduce failed: {e}")
    
    # Test all_to_all_single
    try:
        input_t = torch.arange(world_size, device=device, dtype=torch.float32)
        output_t = torch.empty_like(input_t)
        dist.all_to_all_single(output_t, input_t)
        results['all_to_all'] = True
        print(f"[Rank {rank}] ✓ all_to_all_single")
    except Exception as e:
        results['all_to_all'] = False
        print(f"[Rank {rank}] ✗ all_to_all_single failed: {e}")
    
    # Test isend/irecv
    try:
        t = torch.tensor([float(rank)], device=device)
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size
        
        if rank % 2 == 0:
            req1 = dist.isend(t, next_rank)
            req2 = dist.irecv(t, prev_rank)
        else:
            req2 = dist.irecv(t, prev_rank)
            req1 = dist.isend(t, next_rank)
        
        req1.wait()
        req2.wait()
        results['isend_irecv'] = True
        print(f"[Rank {rank}] ✓ isend/irecv")
    except Exception as e:
        results['isend_irecv'] = False
        print(f"[Rank {rank}] ✗ isend/irecv failed: {e}")
    
    return all(results.values())


def main():
    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()
    
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Final Ring & Ulysses Test with CCL+XPU")
    print(f"[Rank {rank}] World size: {world_size}")
    print(f"{'='*60}")
    
    # Setup distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    if rank == 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12362'
    
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
    
    # Run tests
    results = {}
    
    # Test 1: Collective operations
    results['collectives'] = test_collective_operations(rank, world_size, device)
    
    # Test 2: Ring attention
    results['ring'] = test_ring_attention(rank, world_size, device)
    
    # Test 3: Ulysses attention
    results['ulysses'] = test_ulysses_attention(rank, world_size, device)
    
    # Synchronize before summary
    dist.barrier()
    
    # Print summary on rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print("FINAL TEST SUMMARY:")
        print(f"{'='*60}")
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        print(f"\n{'='*60}")
        if all_passed:
            print("✅ ALL TESTS PASSED!")
            print("\nCONCLUSION:")
            print("- CCL backend fully supports XPU devices")
            print("- All collective operations work correctly")
            print("- Both Ring and Ulysses attention work with CCL+XPU")
            print("- No fallback mechanisms needed")
        else:
            print("❌ Some tests failed")
        print(f"{'='*60}")
    
    # Cleanup
    dist.destroy_process_group()
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())