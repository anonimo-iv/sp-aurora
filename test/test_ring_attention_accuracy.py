#!/usr/bin/env python3
"""
Test ring attention accuracy.
Usage: mpirun -n 1 python test_ring_attention_accuracy.py
"""

import os
import sys
import torch
import torch.distributed as dist
from mpi4py import MPI
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import intel_extension_for_pytorch as ipex
    HAS_XPU = torch.xpu.is_available()
except:
    HAS_XPU = False

from sp_aurora.intel_flash_attn import intel_flash_attn_forward
from sp_aurora.utils import update_out_and_lse


def reference_attention(q, k, v, causal=True):
    """Standard attention for comparison"""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if causal:
        mask = torch.triu(torch.ones_like(scores), diagonal=1)
        scores.masked_fill_(mask.bool(), float('-inf'))
    
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.matmul(attn, v)
    
    return out, lse


def test_accuracy():
    """Test flash attention accuracy"""
    device = 'xpu' if HAS_XPU else 'cpu'
    
    # Test cases
    for dtype in [torch.float32, torch.float16]:
        print(f"\nTesting dtype: {dtype}")
        
        # Small test case
        b, h, s, d = 2, 8, 256, 64
        q = torch.randn(b, h, s, d, device=device, dtype=dtype)
        k = torch.randn(b, h, s, d, device=device, dtype=dtype)
        v = torch.randn(b, h, s, d, device=device, dtype=dtype)
        
        # Normalize to prevent overflow
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        
        # Reference
        ref_out, ref_lse = reference_attention(q, k, v)
        
        # Flash attention
        flash_out, flash_lse = intel_flash_attn_forward(q, k, v, causal=True)
        
        # Compare
        out_diff = (flash_out - ref_out).abs().max().item()
        lse_diff = (flash_lse - ref_lse).abs().max().item()
        
        print(f"Output max diff: {out_diff:.6f}")
        print(f"LSE max diff: {lse_diff:.6f}")
        
        # Tolerances
        tol = 1e-3 if dtype == torch.float32 else 1e-2
        
        if out_diff < tol and lse_diff < tol:
            print("✓ PASSED")
        else:
            print("✗ FAILED")


def main():
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    if rank == 0:
        print("Ring Attention Accuracy Test")
        print("="*40)
    
    # Distributed setup if needed
    if world_size > 1:
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        
        backend = 'ccl' if HAS_XPU else 'gloo'
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
    
    # Run tests only on rank 0
    if rank == 0:
        test_accuracy()
    
    # Cleanup
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()