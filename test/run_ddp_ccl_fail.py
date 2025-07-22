#!/usr/bin/env python3
"""
Simple test for ring communication pattern that's hanging
"""

import os
import sys
import socket
import datetime
import torch
import torch.distributed as dist
from mpi4py import MPI

# Import Intel GPU modules
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

def test_ring_communication():
    """Test the exact ring communication pattern used in intel_ring_flash_attn"""
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'xpu:{rank % 12}'  # Use modulo for multi-GPU nodes
    
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
    
    # CCL backend requires a collective operation before P2P communication
    print(f"[Rank {rank}] Performing dummy all_gather to initialize CCL communicators...")
    dummy_tensor = torch.tensor([float(rank)], device=device, dtype=torch.float32)
    gather_list = [torch.zeros_like(dummy_tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, dummy_tensor)
    print(f"[Rank {rank}] CCL communicators initialized, gathered values: {[t.item() for t in gather_list]}")
    
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
    # MPI initialization
    # MPI.COMM_WORLD.Barrier()
    
    # Set up environment variables from MPI
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', MPI.COMM_WORLD.Get_rank()))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', MPI.COMM_WORLD.Get_size()))
    mpi_world_size = MPI.COMM_WORLD.Get_size()
    mpi_my_rank = MPI.COMM_WORLD.Get_rank()
    
    # Set up master address and port
    if mpi_my_rank == 0:
        master_addr = socket.gethostname()
        sock = socket.socket()
        sock.bind(('', 0))
        master_port = sock.getsockname()[1]
        sock.close()
    else:
        master_addr = None
        master_port = None
    
    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    
    # Check Intel GPU
    if not torch.xpu.is_available():
        print("❌ Intel GPU not available")
        return 1
    
    print(f"Process PID: {os.getpid()}")
    print(f"✅ Intel GPU available: {torch.xpu.device_count()} devices")
    
    # Initialize distributed process group
    MPI.COMM_WORLD.Barrier()
    dist.init_process_group(backend="ccl", init_method='env://', 
                          world_size=mpi_world_size, rank=mpi_my_rank, 
                          timeout=datetime.timedelta(seconds=3600))
    MPI.COMM_WORLD.Barrier()
    
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