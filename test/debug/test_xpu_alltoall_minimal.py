#!/usr/bin/env python3
"""
Minimal test to understand all-to-all behavior on XPU with CCL backend.
Tests different approaches to find what works.

Usage:
    mpirun -n 2 python test_xpu_alltoall_minimal.py
"""

import torch
import torch.distributed as dist
import os
import sys
import datetime
from mpi4py import MPI

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


def test_direct_alltoall(rank, world_size, device):
    """Test direct all_to_all_single on XPU"""
    print(f"\n[Rank {rank}] Test 1: Direct all_to_all_single on {device}")
    
    try:
        # Create test tensor
        input_tensor = torch.arange(4 * world_size, dtype=torch.float32, device=device).reshape(world_size, 4)
        input_chunk = input_tensor[rank]  # Each rank gets its chunk
        
        # Flatten for all_to_all_single
        input_flat = input_chunk.repeat(world_size)
        output_flat = torch.empty_like(input_flat)
        
        print(f"[Rank {rank}] Input: {input_flat}")
        dist.all_to_all_single(output_flat, input_flat)
        print(f"[Rank {rank}] ✓ Direct all_to_all_single SUCCESS!")
        print(f"[Rank {rank}] Output: {output_flat}")
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ Direct all_to_all_single failed: {e}")
        return False


def test_alltoall_with_dummy_allreduce(rank, world_size, device):
    """Test all_to_all_single after dummy all_reduce (like Ring does)"""
    print(f"\n[Rank {rank}] Test 2: all_to_all_single with dummy all_reduce first")
    
    try:
        # Do dummy all_reduce first (like Ring)
        dummy_tensor = torch.tensor([1.0], device=device, dtype=torch.float32)
        print(f"[Rank {rank}] Performing dummy all_reduce...")
        dist.all_reduce(dummy_tensor)
        print(f"[Rank {rank}] Dummy all_reduce result: {dummy_tensor.item()}")
        
        # Now try all_to_all_single
        input_tensor = torch.arange(4 * world_size, dtype=torch.float32, device=device).reshape(world_size, 4)
        input_chunk = input_tensor[rank]
        
        input_flat = input_chunk.repeat(world_size)
        output_flat = torch.empty_like(input_flat)
        
        print(f"[Rank {rank}] Input: {input_flat}")
        dist.all_to_all_single(output_flat, input_flat)
        print(f"[Rank {rank}] ✓ all_to_all_single after dummy all_reduce SUCCESS!")
        print(f"[Rank {rank}] Output: {output_flat}")
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ all_to_all_single after dummy all_reduce failed: {e}")
        return False


def test_alltoall_using_isend_irecv(rank, world_size, device):
    """Implement all-to-all using isend/irecv (async P2P)"""
    print(f"\n[Rank {rank}] Test 3: all-to-all using isend/irecv")
    
    try:
        # Create test data
        input_tensor = torch.arange(4 * world_size, dtype=torch.float32, device=device).reshape(world_size, 4)
        chunks_to_send = [input_tensor[i].clone() for i in range(world_size)]
        chunks_received = [torch.empty(4, dtype=torch.float32, device=device) for _ in range(world_size)]
        
        # Send and receive chunks using isend/irecv
        send_reqs = []
        recv_reqs = []
        
        for i in range(world_size):
            if i == rank:
                # Copy own chunk
                chunks_received[i].copy_(chunks_to_send[i])
            else:
                # Use rank parity ordering to avoid deadlock
                if rank < i:
                    # Send first, then receive
                    send_req = dist.isend(chunks_to_send[i], dst=i)
                    recv_req = dist.irecv(chunks_received[i], src=i)
                else:
                    # Receive first, then send
                    recv_req = dist.irecv(chunks_received[i], src=i)
                    send_req = dist.isend(chunks_to_send[i], dst=i)
                
                send_reqs.append(send_req)
                recv_reqs.append(recv_req)
        
        # Wait for all operations
        for req in send_reqs:
            req.wait()
        for req in recv_reqs:
            req.wait()
        
        output = torch.cat(chunks_received)
        print(f"[Rank {rank}] ✓ all-to-all using isend/irecv SUCCESS!")
        print(f"[Rank {rank}] Output: {output}")
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ all-to-all using isend/irecv failed: {e}")
        return False


def test_4d_tensor_alltoall(rank, world_size, device):
    """Test all-to-all with 4D tensors like Ulysses uses"""
    print(f"\n[Rank {rank}] Test 4: 4D tensor all-to-all (Ulysses pattern)")
    
    try:
        # Create 4D tensor: (batch, seq_len/P, num_heads, head_dim)
        batch_size = 2
        seq_len_per_rank = 4
        num_heads = 8
        head_dim = 64
        
        input_4d = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim, 
                               device=device, dtype=torch.float16)
        
        # Reshape for all-to-all: need to scatter on heads dimension
        # Target: (batch, seq_len, num_heads/P, head_dim)
        shard_heads = num_heads // world_size
        
        # Reshape to (batch, seq_len/P, world_size, num_heads/P, head_dim)
        input_reshaped = input_4d.reshape(batch_size, seq_len_per_rank, world_size, shard_heads, head_dim)
        
        # Transpose to put world_size first: (world_size, seq_len/P, batch, num_heads/P, head_dim)
        input_t = input_reshaped.permute(2, 1, 0, 3, 4).contiguous()
        
        # Flatten for all_to_all_single
        input_flat = input_t.view(world_size, -1)
        output_flat = torch.empty_like(input_flat)
        
        # Try with dummy all_reduce first
        dummy = torch.tensor([1.0], device=device, dtype=torch.float32)
        dist.all_reduce(dummy)
        
        dist.all_to_all_single(output_flat, input_flat)
        
        # Reshape back
        output_t = output_flat.view(world_size, seq_len_per_rank, batch_size, shard_heads, head_dim)
        seq_len_total = seq_len_per_rank * world_size
        output_4d = output_t.reshape(seq_len_total, batch_size, shard_heads, head_dim)
        output_4d = output_4d.permute(1, 0, 2, 3).contiguous()
        
        print(f"[Rank {rank}] ✓ 4D tensor all-to-all SUCCESS!")
        print(f"[Rank {rank}] Input shape: {input_4d.shape}, Output shape: {output_4d.shape}")
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ 4D tensor all-to-all failed: {e}")
        return False


def main():
    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()
    
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] XPU All-to-All Minimal Test")
    print(f"[Rank {rank}] World size: {world_size}")
    print(f"{'='*60}")
    
    # Setup distributed environment
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    if world_size > 1:
        # Set master address/port
        if rank == 0:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
        
        # Broadcast master info
        master_addr = mpi_comm.bcast(os.environ.get('MASTER_ADDR'), root=0)
        master_port = mpi_comm.bcast(os.environ.get('MASTER_PORT'), root=0)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
    
    # Synchronize before init
    mpi_comm.Barrier()
    
    # Determine device and backend
    if INTEL_GPU_AVAILABLE:
        device = torch.device(f'xpu:{rank % torch.xpu.device_count()}')
        torch.xpu.set_device(device)
        backend = 'ccl'
        print(f"[Rank {rank}] Using XPU device: {device}, backend: {backend}")
    else:
        device = torch.device('cpu')
        backend = 'gloo'
        print(f"[Rank {rank}] Intel GPU not available, using CPU with gloo")
    
    # Initialize process group
    if world_size > 1:
        try:
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(seconds=300)
            )
            print(f"[Rank {rank}] Process group initialized")
        except Exception as e:
            print(f"[Rank {rank}] Failed to init process group: {e}")
            return 1
    
    # Run tests
    results = {}
    
    # Test 1: Direct all_to_all
    results['direct'] = test_direct_alltoall(rank, world_size, device)
    
    # Test 2: With dummy all_reduce
    results['with_dummy'] = test_alltoall_with_dummy_allreduce(rank, world_size, device)
    
    # Test 3: Using isend/irecv
    results['isend_irecv'] = test_alltoall_using_isend_irecv(rank, world_size, device)
    
    # Test 4: 4D tensor pattern
    results['4d_tensor'] = test_4d_tensor_alltoall(rank, world_size, device)
    
    # Synchronize before summary
    if dist.is_initialized():
        dist.barrier()
    
    # Print summary
    if rank == 0:
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"{'='*60}")
        for test_name, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{test_name}: {status}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS:")
        if results['with_dummy'] and not results['direct']:
            print("- Dummy all_reduce enables all-to-all! Use this approach.")
        if results['isend_irecv']:
            print("- Async P2P (isend/irecv) works as fallback.")
        if not any(results.values()):
            print("- No approach worked. Use CPU fallback or gloo backend.")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())