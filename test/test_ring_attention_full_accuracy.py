#!/usr/bin/env python3
"""
Fixed full accuracy test for ring attention comparing with reference implementation.
Usage: 
    mpirun -n 1 python test_ring_attention_full_accuracy_fixed.py
    mpirun -n 2 python test_ring_attention_full_accuracy_fixed.py
"""

import os
import sys
import torch
import torch.distributed as dist
from mpi4py import MPI
import datetime
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import intel_extension_for_pytorch as ipex
    HAS_XPU = torch.xpu.is_available()
except:
    HAS_XPU = False

from sp_aurora.intel_flash_attn import intel_flash_attn_forward
from sp_aurora.intel_ring_flash_attn import intel_ring_flash_attn_forward


def test_ring_accuracy_distributed():
    """Test ring attention accuracy in distributed setting"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Setup distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12348'
    
    backend = 'ccl' if HAS_XPU else 'gloo'
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60)
    )
    
    device = f'xpu:{rank % torch.xpu.device_count()}' if HAS_XPU else 'cpu'
    if HAS_XPU:
        torch.xpu.set_device(device)
        print(f"[Rank {rank}] Using device: {device}")
    
    # Test parameters
    batch_size = 2
    seq_len_per_rank = 128
    num_heads = 8
    head_dim = 64
    dtype = torch.float32
    
    if rank == 0:
        print(f"\nRing Attention Distributed Accuracy Test")
        print(f"World size: {world_size}")
        print(f"Sequence length per rank: {seq_len_per_rank}")
        print(f"Total sequence length: {seq_len_per_rank * world_size}")
        print("="*60)
    
    # Create test data in [batch, num_heads, seq_len, head_dim] format
    # This is the format expected by intel_ring_flash_attn_forward
    torch.manual_seed(42 + rank)  # Different seed per rank
    q_local = torch.randn(batch_size, num_heads, seq_len_per_rank, head_dim, 
                          device=device, dtype=dtype)
    k_local = torch.randn(batch_size, num_heads, seq_len_per_rank, head_dim, 
                          device=device, dtype=dtype)
    v_local = torch.randn(batch_size, num_heads, seq_len_per_rank, head_dim, 
                          device=device, dtype=dtype)
    
    # Normalize to prevent overflow
    q_local = q_local / q_local.norm(dim=-1, keepdim=True)
    k_local = k_local / k_local.norm(dim=-1, keepdim=True)
    
    print(f"[Rank {rank}] Local tensor shape: {q_local.shape}")
    
    # Synchronize before computation
    dist.barrier()
    
    # Run ring attention
    if rank == 0:
        print("\nRunning ring attention...")
    
    ring_out, ring_lse = intel_ring_flash_attn_forward(
        None,  # process_group
        q_local, k_local, v_local,
        softmax_scale=1.0 / math.sqrt(head_dim),
        dropout_p=0.0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False
    )
    
    print(f"[Rank {rank}] Ring output shape: {ring_out.shape}")
    print(f"[Rank {rank}] Ring LSE shape: {ring_lse.shape}")
    
    # Synchronize before gathering
    dist.barrier()
    
    # Gather all inputs and outputs for comparison
    if world_size > 1:
        # Create lists for all_gather
        q_list = [torch.empty_like(q_local) for _ in range(world_size)]
        k_list = [torch.empty_like(k_local) for _ in range(world_size)]
        v_list = [torch.empty_like(v_local) for _ in range(world_size)]
        out_list = [torch.empty_like(ring_out) for _ in range(world_size)]
        
        # All gather - ensure contiguous tensors
        dist.all_gather(q_list, q_local.contiguous())
        dist.all_gather(k_list, k_local.contiguous())
        dist.all_gather(v_list, v_local.contiguous())
        dist.all_gather(out_list, ring_out.contiguous())
        
        if rank == 0:
            # Concatenate along sequence dimension (dim=2)
            q_all = torch.cat(q_list, dim=2)
            k_all = torch.cat(k_list, dim=2)
            v_all = torch.cat(v_list, dim=2)
            ring_out_all = torch.cat(out_list, dim=2)
            
            print(f"\nGathered tensor shapes:")
            print(f"Q all: {q_all.shape}")
            print(f"Ring output all: {ring_out_all.shape}")
    else:
        q_all = q_local
        k_all = k_local
        v_all = v_local
        ring_out_all = ring_out
    
    # Compute reference on rank 0
    if rank == 0:
        print("\nComputing reference attention on full sequence...")
        ref_out, ref_lse = intel_flash_attn_forward(
            q_all, k_all, v_all,
            dropout_p=0.0,
            causal=True,
            softmax_scale=1.0 / math.sqrt(head_dim)
        )
        
        print(f"Reference output shape: {ref_out.shape}")
        
        # Compare results
        abs_diff = (ring_out_all - ref_out).abs()
        rel_diff = abs_diff / (ref_out.abs() + 1e-8)
        
        max_abs_error = abs_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        max_rel_error = rel_diff.max().item()
        mean_rel_error = rel_diff.mean().item()
        
        print(f"\nAccuracy Results:")
        print(f"Max absolute error: {max_abs_error:.6e}")
        print(f"Mean absolute error: {mean_abs_error:.6e}")
        print(f"Max relative error: {max_rel_error:.6e}")
        print(f"Mean relative error: {mean_rel_error:.6e}")
        
        # Check if within tolerance
        tolerance = 1e-3 if dtype == torch.float32 else 1e-2
        
        if max_abs_error < tolerance:
            print(f"\n✅ PASSED: Ring attention matches reference within tolerance ({tolerance})")
        else:
            print(f"\n❌ FAILED: Ring attention exceeds error tolerance ({tolerance})")
            
            # Find worst errors
            worst_idx = torch.unravel_index(abs_diff.argmax(), abs_diff.shape)
            print(f"\nWorst error at index {worst_idx}:")
            print(f"Ring value: {ring_out_all[worst_idx].item():.6f}")
            print(f"Ref value: {ref_out[worst_idx].item():.6f}")
            print(f"Difference: {abs_diff[worst_idx].item():.6f}")
        
        # Check for NaN/Inf
        if torch.isnan(ring_out_all).any():
            print("\n⚠️  WARNING: Ring output contains NaN!")
        if torch.isinf(ring_out_all).any():
            print("\n⚠️  WARNING: Ring output contains Inf!")
    
    # Final synchronization
    dist.barrier()
    
    # Cleanup
    dist.destroy_process_group()
    
    if rank == 0:
        print("\nTest completed")


def test_single_process_accuracy():
    """Test accuracy in single process mode"""
    print("Single Process Accuracy Test")
    print("="*60)
    
    device = 'xpu' if HAS_XPU else 'cpu'
    
    # Test parameters
    batch_size = 2
    seq_len = 256
    num_heads = 8
    head_dim = 64
    
    # Create test data
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Normalize
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    
    print(f"Input shape: {q.shape}")
    
    # Run flash attention
    out, lse = intel_flash_attn_forward(q, k, v, dropout_p=0.0, causal=True)
    
    print(f"Output shape: {out.shape}")
    print(f"LSE shape: {lse.shape}")
    
    # Basic sanity checks
    if torch.isnan(out).any():
        print("❌ FAILED: Output contains NaN")
    elif torch.isinf(out).any():
        print("❌ FAILED: Output contains Inf")
    else:
        print("✅ PASSED: Output is valid")


def main():
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    
    if world_size > 1:
        test_ring_accuracy_distributed()
    else:
        test_single_process_accuracy()
        print("\nFor distributed test, run with: mpirun -n 2 python test_ring_attention_full_accuracy_fixed.py")


if __name__ == "__main__":
    main()