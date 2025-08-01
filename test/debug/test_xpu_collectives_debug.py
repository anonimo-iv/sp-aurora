#!/usr/bin/env python3
"""
Debug test to understand which collective operations work on XPU with CCL.
Tests all collective operations mentioned in the problem list.

Usage:
    mpirun -n 2 python test_xpu_collectives_debug.py
"""

import torch
import torch.distributed as dist
import os
import sys
from mpi4py import MPI
import datetime

# Check for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
    INTEL_GPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
except ImportError as e:
    print(f"Warning: Intel Extension not available: {e}")
    INTEL_GPU_AVAILABLE = False


def test_collective(rank, operation_name, test_func):
    """Helper to test a collective operation"""
    try:
        test_func()
        print(f"[Rank {rank}] ✓ {operation_name} SUCCESS")
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ {operation_name} FAILED: {e}")
        return False


def main():
    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()
    
    print(f"\n[Rank {rank}] XPU Collectives Debug Test")
    print(f"[Rank {rank}] Testing which operations work on XPU with CCL")
    print(f"[Rank {rank}] World size: {world_size}")
    
    # Setup distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    if rank == 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
    
    master_addr = mpi_comm.bcast(os.environ.get('MASTER_ADDR'), root=0)
    master_port = mpi_comm.bcast(os.environ.get('MASTER_PORT'), root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    mpi_comm.Barrier()
    
    # Determine device and backend
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
    
    results = {}
    
    # Test different tensor types
    for dtype_name, dtype in [('float32', torch.float32), ('float16', torch.float16)]:
        print(f"\n[Rank {rank}] === Testing with {dtype_name} ===")
        
        # Test 1: all_reduce
        def test_allreduce():
            t = torch.tensor([float(rank)], device=device, dtype=dtype)
            dist.all_reduce(t)
        results[f'all_reduce_{dtype_name}'] = test_collective(rank, f"all_reduce ({dtype_name})", test_allreduce)
        
        # Test 2: broadcast
        def test_broadcast():
            t = torch.tensor([42.0 if rank == 0 else 0.0], device=device, dtype=dtype)
            dist.broadcast(t, src=0)
        results[f'broadcast_{dtype_name}'] = test_collective(rank, f"broadcast ({dtype_name})", test_broadcast)
        
        # Test 3: send/recv (sync)
        def test_send_recv():
            t = torch.tensor([float(rank)], device=device, dtype=dtype)
            if rank == 0:
                dist.send(t, dst=1)
            else:
                dist.recv(t, src=0)
        if world_size > 1:
            results[f'send_recv_{dtype_name}'] = test_collective(rank, f"send/recv ({dtype_name})", test_send_recv)
        
        # Test 4: isend/irecv (async)
        def test_isend_irecv():
            t = torch.tensor([float(rank)], device=device, dtype=dtype)
            if rank == 0:
                req = dist.isend(t, dst=1)
                req.wait()
            else:
                req = dist.irecv(t, src=0)
                req.wait()
        if world_size > 1:
            results[f'isend_irecv_{dtype_name}'] = test_collective(rank, f"isend/irecv ({dtype_name})", test_isend_irecv)
        
        # Test 5: all_to_all_single
        def test_alltoall():
            input_t = torch.arange(world_size, device=device, dtype=dtype)
            output_t = torch.empty_like(input_t)
            dist.all_to_all_single(output_t, input_t)
        results[f'alltoall_{dtype_name}'] = test_collective(rank, f"all_to_all_single ({dtype_name})", test_alltoall)
        
        # Test 6: all_gather
        def test_allgather():
            t = torch.tensor([float(rank)], device=device, dtype=dtype)
            output = torch.empty(world_size, device=device, dtype=dtype)
            dist.all_gather_into_tensor(output, t)
        results[f'allgather_{dtype_name}'] = test_collective(rank, f"all_gather ({dtype_name})", test_allgather)
        
        # Test 7: barrier
        def test_barrier():
            dist.barrier()
        results[f'barrier_{dtype_name}'] = test_collective(rank, f"barrier ({dtype_name})", test_barrier)
    
    # Test with dummy all_reduce first
    print(f"\n[Rank {rank}] === Testing with dummy all_reduce first ===")
    
    # Do dummy all_reduce
    dummy = torch.tensor([1.0], device=device, dtype=torch.float32)
    try:
        dist.all_reduce(dummy)
        print(f"[Rank {rank}] Dummy all_reduce succeeded")
        
        # Now test all_to_all again
        def test_alltoall_after_dummy():
            input_t = torch.arange(world_size, device=device, dtype=torch.float32)
            output_t = torch.empty_like(input_t)
            dist.all_to_all_single(output_t, input_t)
        results['alltoall_after_dummy'] = test_collective(rank, "all_to_all after dummy", test_alltoall_after_dummy)
    except Exception as e:
        print(f"[Rank {rank}] Dummy all_reduce failed: {e}")
    
    # Synchronize
    dist.barrier()
    
    # Print summary on rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print("SUMMARY OF RESULTS:")
        print(f"{'='*60}")
        
        working = []
        failing = []
        
        for op, success in results.items():
            if success:
                working.append(op)
            else:
                failing.append(op)
        
        print("\nWORKING operations:")
        for op in working:
            print(f"  ✓ {op}")
        
        print("\nFAILING operations:")
        for op in failing:
            print(f"  ✗ {op}")
        
        print(f"\nTotal: {len(working)} working, {len(failing)} failing")
    
    # Cleanup
    dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())