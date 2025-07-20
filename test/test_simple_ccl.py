#!/usr/bin/env python3
"""
Simple test to verify CCL P2P communication works
"""

import os
import torch
import torch.distributed as dist
import intel_extension_for_pytorch as ipex

# Import oneccl bindings to register CCL backend
try:
    import oneccl_bindings_for_pytorch
except ImportError:
    print("oneCCL bindings not available")
    exit(1)

# Set CCL environment
os.environ['CCL_BACKEND'] = 'native'
os.environ['CCL_ATL_TRANSPORT'] = 'ofi'
os.environ['FI_PROVIDER'] = 'cxi'
os.environ['CCL_ZE_IPC_EXCHANGE'] = 'drmfd'
os.environ['CCL_ZE_ENABLE'] = '1'
os.environ['CCL_LOG_LEVEL'] = 'warn'  # Reduce verbosity

def test_simple_p2p():
    """Test simple P2P communication"""
    if not dist.is_initialized():
        dist.init_process_group(backend='ccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'xpu:{rank}' if torch.xpu.device_count() > 1 else 'xpu'
    
    print(f"[Rank {rank}] Device: {device}")
    
    # Create test tensor
    tensor = torch.full((10,), rank, device=device, dtype=torch.float32)
    print(f"[Rank {rank}] Initial tensor: {tensor}")
    
    # Simple ring communication
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size
    
    recv_tensor = torch.empty_like(tensor)
    
    print(f"[Rank {rank}] Sending to {send_rank}, receiving from {recv_rank}")
    
    # Try different P2P methods
    
    # Method 1: Direct send/recv (blocking)
    if rank == 0:
        dist.send(tensor, dst=send_rank)
        dist.recv(recv_tensor, src=recv_rank)
    else:
        dist.recv(recv_tensor, src=recv_rank)
        dist.send(tensor, dst=send_rank)
    
    print(f"[Rank {rank}] Received tensor: {recv_tensor}")
    
    # Verify
    expected = recv_rank
    if torch.all(recv_tensor == expected):
        print(f"[Rank {rank}] ✅ P2P test passed!")
    else:
        print(f"[Rank {rank}] ❌ P2P test failed! Expected {expected}, got {recv_tensor}")

if __name__ == "__main__":
    test_simple_p2p()
    
    if dist.is_initialized():
        dist.destroy_process_group()