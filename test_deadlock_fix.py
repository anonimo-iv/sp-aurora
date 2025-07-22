#!/usr/bin/env python3
"""
Test script to verify the deadlock fix in ring communication
"""

import torch
import torch.distributed as dist
from ring_flash_attn.intel_utils import IntelRingComm
import os
import sys

def test_ring_comm():
    """Test ring communication with the deadlock fix"""
    if not dist.is_initialized():
        print("Error: Distributed not initialized. Run with mpiexec or torchrun")
        return False
        
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] Starting ring communication test with {world_size} processes")
    
    # Create ring communicator
    comm = IntelRingComm(None)
    
    # Create test tensor
    device = 'xpu' if torch.xpu.is_available() else 'cpu'
    tensor_size = 1024
    test_tensor = torch.ones(tensor_size, dtype=torch.float32, device=device) * rank
    
    print(f"[Rank {rank}] Sending tensor with value {rank}")
    
    # Test the fixed send_recv method
    try:
        recv_tensor = comm.send_recv(test_tensor)
        
        # Wait for communication to complete
        comm.wait()
        
        expected_value = float((rank - 1 + world_size) % world_size)
        actual_value = recv_tensor[0].item()
        
        print(f"[Rank {rank}] Received tensor with value {actual_value}, expected {expected_value}")
        
        if abs(actual_value - expected_value) < 1e-6:
            print(f"[Rank {rank}] ✅ Test PASSED!")
            return True
        else:
            print(f"[Rank {rank}] ❌ Test FAILED!")
            return False
            
    except Exception as e:
        print(f"[Rank {rank}] ❌ Test FAILED with exception: {e}")
        return False

if __name__ == "__main__":
    # Check if running with MPI
    if 'PMI_RANK' in os.environ:
        # Running with mpiexec
        from test.run_ddp_ccl import device  # This will initialize distributed
    else:
        # Initialize distributed if not already done
        if not dist.is_initialized():
            print("Please run with: mpiexec -n 2 python test_deadlock_fix.py")
            sys.exit(1)
    
    success = test_ring_comm()
    
    # Synchronize before exit
    dist.barrier()
    
    if dist.get_rank() == 0:
        if success:
            print("\n✅ All tests passed! The deadlock fix is working correctly.")
        else:
            print("\n❌ Tests failed!")