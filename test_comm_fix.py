#!/usr/bin/env python3

import os
import torch
import torch.distributed as dist

# Import oneCCL bindings first
try:
    import oneccl_bindings_for_pytorch
    print("✅ oneCCL bindings loaded")
except ImportError as e:
    print(f"❌ oneCCL bindings not available: {e}")

def test_ccl_communication():
    """Test basic CCL communication"""
    # Initialize distributed
    dist.init_process_group(backend='ccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] Initialized CCL backend, world_size={world_size}")
    
    # Test basic all-gather
    device = f'xpu:{rank}' if torch.xpu.device_count() > 1 else 'xpu'
    tensor = torch.ones(2, 2, device=device, dtype=torch.float16) * rank
    
    print(f"[Rank {rank}] Created tensor on {device}: {tensor}")
    
    # All-gather test
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    print(f"[Rank {rank}] All-gather successful!")
    for i, t in enumerate(tensor_list):
        print(f"[Rank {rank}] Received from rank {i}: {t}")
    
    # Cleanup
    dist.destroy_process_group()
    return True

if __name__ == "__main__":
    try:
        result = test_ccl_communication()
        print(f"✅ Communication test passed!")
    except Exception as e:
        print(f"❌ Communication test failed: {e}")
        import traceback
        traceback.print_exc()