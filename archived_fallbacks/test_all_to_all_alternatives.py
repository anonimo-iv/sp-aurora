#!/usr/bin/env python3
"""
Test script for all-to-all alternative implementations
Tests each method to see which ones work with CCL backend on Intel GPU
"""

import torch
import torch.distributed as dist
import os
import sys
from mpi4py import MPI
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import oneCCL bindings
import oneccl_bindings_for_pytorch

# Import our fallback implementations
from ring_flash_attn.comm.all_to_all_fallback import (
    all_to_all_single_native,
    all_to_all_using_allgather,
    all_to_all_using_send_recv,
    all_to_all_with_cpu_fallback,
    all_to_all_fallback
)


def setup_distributed():
    """Initialize distributed environment"""
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    
    if size > 1 and rank == 0:
        import socket
        os.environ['MASTER_ADDR'] = socket.gethostname()
    
    if size > 1:
        master_addr = mpi_comm.bcast(os.environ['MASTER_ADDR'], root=0)
        os.environ['MASTER_ADDR'] = master_addr
        mpi_comm.Barrier()
    
    dist.init_process_group(backend='ccl', init_method='env://')
    
    return rank, size


def test_native_all_to_all(tensor, rank):
    """Test native PyTorch all_to_all_single"""
    print(f"[Rank {rank}] Testing native all_to_all_single...", flush=True)
    try:
        start = time.time()
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor)
        end = time.time()
        print(f"[Rank {rank}] ✓ Native all_to_all_single succeeded in {end-start:.3f}s", flush=True)
        return True, end - start
    except Exception as e:
        print(f"[Rank {rank}] ✗ Native all_to_all_single failed: {e}", flush=True)
        return False, None


def test_allgather_method(tensor, rank):
    """Test allgather-based implementation"""
    print(f"[Rank {rank}] Testing allgather-based method...", flush=True)
    try:
        start = time.time()
        # Test with typical Ulysses dimensions
        output = all_to_all_using_allgather(tensor, scatter_idx=2, gather_idx=1)
        end = time.time()
        print(f"[Rank {rank}] ✓ Allgather method succeeded in {end-start:.3f}s", flush=True)
        return True, end - start
    except Exception as e:
        print(f"[Rank {rank}] ✗ Allgather method failed: {e}", flush=True)
        return False, None


def test_send_recv_method(tensor, rank):
    """Test send/recv-based implementation"""
    print(f"[Rank {rank}] Testing send/recv-based method...", flush=True)
    try:
        start = time.time()
        output = all_to_all_using_send_recv(tensor, scatter_idx=2, gather_idx=1)
        end = time.time()
        print(f"[Rank {rank}] ✓ Send/recv method succeeded in {end-start:.3f}s", flush=True)
        return True, end - start
    except Exception as e:
        print(f"[Rank {rank}] ✗ Send/recv method failed: {e}", flush=True)
        return False, None


def test_cpu_fallback_method(tensor, rank):
    """Test CPU fallback method"""
    print(f"[Rank {rank}] Testing CPU fallback method...", flush=True)
    try:
        start = time.time()
        output = all_to_all_with_cpu_fallback(tensor, scatter_idx=2, gather_idx=1)
        end = time.time()
        print(f"[Rank {rank}] ✓ CPU fallback method succeeded in {end-start:.3f}s", flush=True)
        return True, end - start
    except Exception as e:
        print(f"[Rank {rank}] ✗ CPU fallback method failed: {e}", flush=True)
        return False, None


def test_auto_fallback(tensor, rank):
    """Test automatic fallback selection"""
    print(f"[Rank {rank}] Testing automatic fallback selection...", flush=True)
    try:
        start = time.time()
        output = all_to_all_fallback(tensor, scatter_idx=2, gather_idx=1)
        end = time.time()
        print(f"[Rank {rank}] ✓ Auto fallback succeeded in {end-start:.3f}s", flush=True)
        return True, end - start
    except Exception as e:
        print(f"[Rank {rank}] ✗ Auto fallback failed: {e}", flush=True)
        return False, None


def test_correctness(rank, world_size, device):
    """Test correctness of working methods"""
    print(f"\n[Rank {rank}] Testing correctness of implementations...", flush=True)
    
    # Create test tensor with known pattern
    # Shape: (world_size, 4, 8, 2) to match Ulysses pattern
    test_data = torch.arange(world_size * 4 * 8 * 2, dtype=torch.float32).reshape(world_size, 4, 8, 2)
    test_data = test_data.to(device)
    
    # Each rank gets a different slice
    local_data = test_data[rank].unsqueeze(0)  # Shape: (1, 4, 8, 2)
    
    # Test each working method
    methods = {
        'allgather': lambda t: all_to_all_using_allgather(t, scatter_idx=2, gather_idx=1),
        'send_recv': lambda t: all_to_all_using_send_recv(t, scatter_idx=2, gather_idx=1),
        'cpu': lambda t: all_to_all_with_cpu_fallback(t, scatter_idx=2, gather_idx=1)
    }
    
    results = {}
    for name, method in methods.items():
        try:
            output = method(local_data.clone())
            results[name] = output
            print(f"[Rank {rank}] {name} output shape: {output.shape}", flush=True)
        except Exception as e:
            print(f"[Rank {rank}] {name} failed in correctness test: {e}", flush=True)
    
    # Compare results if multiple methods worked
    if len(results) > 1:
        reference_name, reference = list(results.items())[0]
        for name, output in list(results.items())[1:]:
            if torch.allclose(reference, output, rtol=1e-5, atol=1e-5):
                print(f"[Rank {rank}] ✓ {name} matches {reference_name}", flush=True)
            else:
                print(f"[Rank {rank}] ✗ {name} does NOT match {reference_name}", flush=True)


def main():
    # Setup distributed
    rank, world_size = setup_distributed()
    
    print(f"\n{'='*60}", flush=True)
    print(f"[Rank {rank}] Testing All-to-All Alternatives", flush=True)
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
    
    # Create test tensors matching Ulysses dimensions
    # Shape: (P, seq_len/P, bs, num_heads/P, head_dim) for forward pass
    test_tensor_small = torch.randn(world_size, 4, 2, 8, 64, device=device, dtype=torch.float16)
    test_tensor_large = torch.randn(world_size, 512, 2, 8, 64, device=device, dtype=torch.float16)
    
    print(f"[Rank {rank}] Testing with small tensor {test_tensor_small.shape}", flush=True)
    
    # Test each method
    results = {}
    
    # 1. Native all_to_all_single
    success, time_taken = test_native_all_to_all(test_tensor_small, rank)
    results['native'] = (success, time_taken)
    
    # 2. Allgather method
    success, time_taken = test_allgather_method(test_tensor_small, rank)
    results['allgather'] = (success, time_taken)
    
    # 3. Send/recv method
    success, time_taken = test_send_recv_method(test_tensor_small, rank)
    results['send_recv'] = (success, time_taken)
    
    # 4. CPU fallback
    success, time_taken = test_cpu_fallback_method(test_tensor_small, rank)
    results['cpu_fallback'] = (success, time_taken)
    
    # 5. Auto fallback
    success, time_taken = test_auto_fallback(test_tensor_small, rank)
    results['auto'] = (success, time_taken)
    
    # Test with larger tensor for performance comparison
    print(f"\n[Rank {rank}] Testing with large tensor {test_tensor_large.shape}", flush=True)
    working_methods = [name for name, (success, _) in results.items() if success and name != 'native']
    
    for method_name in working_methods:
        if method_name == 'allgather':
            _, time_taken = test_allgather_method(test_tensor_large, rank)
        elif method_name == 'send_recv':
            _, time_taken = test_send_recv_method(test_tensor_large, rank)
        elif method_name == 'cpu_fallback':
            _, time_taken = test_cpu_fallback_method(test_tensor_large, rank)
        print(f"[Rank {rank}] {method_name} on large tensor: {time_taken:.3f}s", flush=True)
    
    # Test correctness
    if world_size > 1:
        test_correctness(rank, world_size, device)
    
    # Summary
    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print("SUMMARY:", flush=True)
        print(f"{'='*60}", flush=True)
        working = [name for name, (success, _) in results.items() if success]
        failed = [name for name, (success, _) in results.items() if not success]
        print(f"Working methods: {', '.join(working)}", flush=True)
        print(f"Failed methods: {', '.join(failed)}", flush=True)
        
        # Recommend best method
        if 'native' in working:
            print("\nRecommendation: Use native all_to_all_single", flush=True)
        elif 'allgather' in working:
            print("\nRecommendation: Use allgather method (set ULYSSES_ALLTOALL_METHOD=allgather)", flush=True)
        elif 'send_recv' in working:
            print("\nRecommendation: Use send/recv method (set ULYSSES_ALLTOALL_METHOD=send_recv)", flush=True)
        else:
            print("\nRecommendation: Use CPU fallback (set ULYSSES_ALLTOALL_METHOD=cpu)", flush=True)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()