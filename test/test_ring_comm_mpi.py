#!/usr/bin/env python3
"""
Simple test for ring communication pattern using MPI backend
"""

import os
import sys
import torch
import torch.distributed as dist

# Import Intel GPU modules
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

def test_ring_communication():
    """Test the exact ring communication pattern used in intel_ring_flash_attn"""
    
    # For MPI, we need to set up the environment variables
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Set environment variables for PyTorch
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    if not dist.is_initialized():
        print(f"[Rank {rank}] Initializing process group...")
        dist.init_process_group(backend='ccl')
    
    device = 'xpu'
    
    print(f"[Rank {rank}] World size: {world_size}, Device: {device}")
    
    if world_size != 2:
        print(f"[Rank {rank}] This test requires exactly 2 processes")
        return
    
    # Create test tensors
    k = torch.ones((2, 4, 8, 16), device=device) * rank  # batch=2, heads=4, seq=8, dim=16
    v = torch.ones((2, 4, 8, 16), device=device) * (rank + 10)
    
    print(f"[Rank {rank}] Initial k sum: {k.sum().item()}, v sum: {v.sum().item()}")
    
    # Ring communication setup
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size
    
    print(f"[Rank {rank}] Will send to {send_rank}, receive from {recv_rank}")
    
    # Initialize communicators with a collective operation (required by PyTorch)
    dummy = torch.tensor([1.0], device=device, dtype=torch.float32)
    dist.broadcast(dummy, src=0)
    print(f"[Rank {rank}] Communicators initialized with broadcast, dummy={dummy.item()}")
    
    # Test 1: Simple blocking send/recv
    print(f"\n[Rank {rank}] Test 1: Simple blocking send/recv")
    recv_k = torch.empty_like(k)
    
    if rank == 0:
        print(f"[Rank {rank}] Sending k...")
        dist.send(k, dst=send_rank)
        print(f"[Rank {rank}] Receiving k...")
        dist.recv(recv_k, src=recv_rank)
    else:
        print(f"[Rank {rank}] Receiving k...")
        dist.recv(recv_k, src=recv_rank)
        print(f"[Rank {rank}] Sending k...")
        dist.send(k, dst=send_rank)
    
    print(f"[Rank {rank}] Received k sum: {recv_k.sum().item()}")
    
    # Test 2: Non-blocking isend/irecv
    print(f"\n[Rank {rank}] Test 2: Non-blocking isend/irecv")
    recv_v = torch.empty_like(v)
    
    print(f"[Rank {rank}] Starting isend and irecv...")
    send_req = dist.isend(v, dst=send_rank)
    recv_req = dist.irecv(recv_v, src=recv_rank)
    
    print(f"[Rank {rank}] Waiting for send...")
    send_req.wait()
    print(f"[Rank {rank}] Waiting for recv...")
    recv_req.wait()
    
    print(f"[Rank {rank}] Received v sum: {recv_v.sum().item()}")
    
    # Test 3: P2POp pattern (like IntelRingComm)
    print(f"\n[Rank {rank}] Test 3: P2POp pattern")
    next_k = torch.empty_like(k)
    next_v = torch.empty_like(v)
    
    ops = []
    send_op_k = dist.P2POp(dist.isend, k, send_rank)
    recv_op_k = dist.P2POp(dist.irecv, next_k, recv_rank)
    send_op_v = dist.P2POp(dist.isend, v, send_rank)
    recv_op_v = dist.P2POp(dist.irecv, next_v, recv_rank)
    
    ops.extend([send_op_k, recv_op_k, send_op_v, recv_op_v])
    
    print(f"[Rank {rank}] Calling batch_isend_irecv with {len(ops)} ops")
    reqs = dist.batch_isend_irecv(ops)
    
    print(f"[Rank {rank}] Waiting for {len(reqs)} requests...")
    for i, req in enumerate(reqs):
        print(f"[Rank {rank}] Waiting for request {i}...")
        req.wait()
        print(f"[Rank {rank}] Request {i} done")
    
    print(f"[Rank {rank}] Received next_k sum: {next_k.sum().item()}, next_v sum: {next_v.sum().item()}")
    
    print(f"\n[Rank {rank}] ✅ All tests completed successfully!")

def main():
    # Check Intel GPU
    if not torch.xpu.is_available():
        print("❌ Intel GPU not available")
        return 1
    
    print(f"Process PID: {os.getpid()}")
    print(f"✅ Intel GPU available: {torch.xpu.device_count()} devices")
    
    try:
        test_ring_communication()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())